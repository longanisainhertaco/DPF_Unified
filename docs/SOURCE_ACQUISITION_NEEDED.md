# Source Acquisition Needed

Updated: 2026-05-12

This document lists external papers and method references that are not exact
local files in the DPF-Unified tree and still matter for scientific closure,
implementation review, or validation-method review.

## Source-of-Truth Rule

We are still following the source-of-truth rule:

- External links and publisher pages are acquisition targets only.
- They are not evidence for physics, validation, requirements closure, or
  readiness claims.
- A source becomes usable only after the correct document is acquired, added to
  the local project, reviewed into `KnowledgeReference/`, hashed, and mapped to
  typed target records or verified digitization packets.
- If a paper cannot be acquired, the affected validation tier remains blocked.

Textbooks and monographs are method references unless a specific validated
target is extracted from them. They can support implementation review,
diagnostic-method design, numerical-method verification, and acceptance
criteria. They do not substitute for same-device experimental papers.

## Team Acquisition Handoff Export

On 2026-05-11, the current acquisition queue was exported to
`docs/SOURCE_ACQUISITION_TEAM_HANDOFF_2026_05_11.xlsx` for team handoff. The
workbook contains:

- `Acquisition Needed`: 91 actionable paper, book, data-sheet, dataset, and
  process-guidance rows with priority, authors/leads, DOI or search route,
  acquisition links, local status, validation/module gap, and required intake
  action.
- `Already Local`: 10 local sources that should not be reacquired unless a
  cleaner official copy or newer edition is specifically needed.
- `Intake Checklist`: the minimum post-acquisition metadata/review steps before
  anything can become local validation or method evidence.

This spreadsheet is an acquisition-management artifact only. It does not
promote any external citation, web page, dataset, or textbook into scientific
evidence.

## User Intake Promotion - 2026-05-11

The 30 PDFs provided from `/Users/anthonyzamora/Downloads` were copied into
`downloaded_books_papers/Research Papers/2026-05-11-user-ingest/` and promoted
through `scripts/promote_research_papers_to_kr.py --apply`.

Promotion result: 91 intake PDFs scanned, 91 unique SHA-256 payloads, 32 new
`KnowledgeReference/` Markdown/JSON pairs written, 59 existing source-level
records skipped, 0 extraction failures, and 0 duplicate deletions. The two
extra promoted records beyond the 30-file user batch were the existing intake
copies of Schmidt et al. 2014 (`1169854.pdf`) and the 2019 NRL Plasma
Formulary.

Durable manifest:
`docs/RESEARCH_PAPERS_KR_PROMOTION_2026_05_11.md` and
`docs/RESEARCH_PAPERS_KR_PROMOTION_2026_05_11.json`.

All records from this batch are `text_parity_extracted_review_needed` with
`validation_status="source_available_not_target_extracted"`. They are now
searchable local source text, but figures, tables, plotted curves, numeric
targets, waveform points, and validation claims are still blocked until
separately reviewed and target-extracted.

Newly local source areas from this batch:

- Toro 2009 full method textbook for Riemann solvers and finite-volume review.
- PF-1000 interferometry/phase/density/field/energy papers: Zielinska 2011
  CTTP DOI `10.1002/ctpp.201000047`, Kubes 2009, Kubes/Klir 2012 energy,
  Kubes 2012 magnetic-probe/neutron/interferometry correlation, Kubes 2013,
  Krauz 2012, Kortanek 2014, Munzar 2021, Cikhardtova 2015, and Szydlowski
  2004.
- Neutron diagnostics and mechanism papers: Klir 2011 detector, Klir 2011 APL,
  Krasa 2008 anisotropy, Jednorog 2015 radioindium, Catenacci 2020, Springham
  2021, and Vikhrev 2007.
- Physics/model-form/source-packet leads: Shumlak 1995/2001 sheared-flow
  stabilization, Buneman 1959 anomalous resistivity, Lotz 1967 ionization,
  Seaton 1959 radiative recombination OCR candidate, Del Zanna 2015 CHIANTI,
  Puetterich 2019 impurity tolerance, Lee/Saw 2008 scaling papers, Lee et al.
  2016 PF1000 radiative cooling, and Stepniewski 2004 PF-1000 MHD modeling.

Title/metadata cleanup still needed: some publisher PDFs exposed weak metadata
titles during automatic extraction, including Seaton 1959, Shumlak 2001,
Kortanek 2014, Kubes 2009, Kubes/Klir 2012 energy, Lee/Saw 2008 neutron
scaling, Lee et al. 2016, Springham 2021, Stepniewski 2004, Szydlowski 2004,
Cikhardtova 2015, and Toro 2009. Use the source PDF path, DOI, SHA-256, and
first-page text in the KR record as the authority for cleanup; do not treat
current H1 metadata as reviewed bibliographic truth until corrected.

## Broader Local PDF Inventory - 2026-05-11

The active promotion run above intentionally scanned only
`downloaded_books_papers/Research Papers`, which currently has 91 unique
SHA-256 payloads. A broader inventory confirms that this is only a narrow
curated intake slice:

- Project PDF-like files outside `KnowledgeReference/`: 1,159 files, 583 unique
  SHA-256 payloads.
- `/Users/anthonyzamora/Downloads` PDF-like files scanned to depth 2: 139
  files, 130 unique SHA-256 payloads.
- Combined project-plus-Downloads inventory: 1,298 PDF-like files, 651 unique
  SHA-256 payloads.

Durable inventory:
`docs/PDF_SOURCE_INVENTORY_2026_05_11.md` and
`docs/PDF_SOURCE_INVENTORY_2026_05_11.json`.

Guardrail: do not bulk-promote the broader 651-unique inventory. It includes
stale archive material, legacy simulator docs, generated plots, vendor/backend
manuals, duplicates, and unrelated Downloads material. Triage
`archive_reference_OLD/references/papers` into the active intake only after a
source is relevant to a named module/source-packet blocker.

## Textbook Chunking - 2026-05-11

Book-length KR Markdown should remain readable. The promotion utility now
writes sources with at least 120 extracted pages as a top-level Markdown index
plus page-range chunks under `KnowledgeReference/chunks/`; JSON still preserves
the full page list.

Applied now:

- Toro 2009 was rewritten as `KnowledgeReference/toro2009-433cd861.md`, a
  30-chunk index with 25-page chunks under
  `KnowledgeReference/chunks/toro2009-433cd861/`.
- Chunking report:
  `docs/KR_TEXTBOOK_CHUNKING_2026_05_11.md` and
  `docs/KR_TEXTBOOK_CHUNKING_2026_05_11.json`.

Existing large KR records were not bulk-rewritten because older source files
may already be cited by line number in findings, docs, or tests. Rechunk them
only when a source is being actively reviewed or when its current Markdown
layout blocks reliable review.

## Source-Critical Fidelity Review - 2026-05-11

The 91 active-intake KR records were second-pass checked after user review to
ensure figure captions, table captions/matrices, formula-like lines, numeric
target contexts, and uncertainty contexts were not silently summarized or
dropped by the first text-only extraction.

Durable audit:
`docs/KR_SOURCE_FIDELITY_AUDIT_2026_05_11.md` and
`docs/KR_SOURCE_FIDELITY_AUDIT_2026_05_11.json`.

Result:

- 91 KR Markdown/JSON records updated.
- 90 records had recovered secondary-extraction items copied into
  `source_fidelity_review`.
- 10,767 recovered items were copied from the second-pass extraction.
- Totals now tracked across the active intake: 2,012 figure captions,
  255 table captions, 345 extracted table matrices, 14,554 formula-like lines,
  9,533 numeric target contexts, 2,143 uncertainty contexts, and 19,784 PDF
  image blocks.

Guardrail: this closes copy-fidelity for the reviewed intake, not validation
acceptance. Plotted curves, visual geometry, and quantitative pass/fail targets
still require explicit target extraction before code or test thresholds may
cite them.

## User Intake Promotion - 2026-05-12

The new user-supplied local PDF batch was staged from the provided paths into
`downloaded_books_papers/Research Papers/2026-05-12-user-ingest/`.

Durable intake inventory:
`docs/USER_PDF_INTAKE_2026_05_12.md`,
`docs/USER_PDF_INTAKE_2026_05_12.json`, and
`docs/USER_PDF_INTAKE_2026_05_12.csv`.

Result:

- 39 supplied paths were readable.
- 35 unique SHA-256 payloads were staged.
- 4 duplicate input paths were detected and not copied as separate canonical
  records.
- 0 files were missing and 0 read failures occurred.

Promotion was intentionally filtered to DPF/plasma/numerics/math-method
sources. Durable promotion manifest:
`docs/USER_PDF_KR_PROMOTION_2026_05_12.md` and
`docs/USER_PDF_KR_PROMOTION_2026_05_12.json`.

Promotion result:

- 28 selected physics/method sources after manual demotion of the out-of-scope
  `symons1994.pdf` JSTOR social-science review.
- 28 new `KnowledgeReference/` Markdown/JSON pairs.
- 0 selected sources skipped as already represented.
- Trunk 1975 was promoted as a distinct source after validation caught a false
  match against an unrelated Kortanek 2014 KR record with the same generic IOP
  cover-page title but a different SHA-256.
- 7 stage-only PDFs kept outside physics authority:
  `apostolou2020.pdf`, `symons1994.pdf`, plus five AI/ML support PDFs.
- 6 book-length records chunked into 126 page-range Markdown chunks.
- 28/28 promoted records passed text-parity checks.

Source-fidelity review was also applied to the 28 promoted records.
Durable fidelity audit:
`docs/USER_PDF_KR_SOURCE_FIDELITY_AUDIT_2026_05_12.md` and
`docs/USER_PDF_KR_SOURCE_FIDELITY_AUDIT_2026_05_12.json`.

Fidelity result:

- 28 KR records updated.
- 27 records had recovered secondary-extraction items.
- 11,376 recovered source-critical items were copied into
  `source_fidelity_review`.
- Totals detected: 1,698 figure captions, 293 table-caption hits, 68 extracted
  table matrices, 25,298 formula-like lines, 4,423 numeric target contexts,
  1,666 uncertainty contexts, and 1,433 PDF image blocks.

Guardrail: this closes source availability and copy-fidelity for the selected
May 12 intake. It does not accept plotted curves, formulas, tables, targets,
uncertainty values, or scientific validation claims. Those still require typed
target extraction and, for digitized figures/tables, independent accepted
review.

May 12 target-triage backlog:
`docs/USER_PDF_MAY12_TARGET_TRIAGE_2026_05_12.md` and
`docs/USER_PDF_MAY12_TARGET_TRIAGE_2026_05_12.json`.

The triage report classifies 28 promoted records into 5
target-extraction candidates and 23 method/context references. P1 target
review candidates are the dense plasma focus expansion discharge paper,
Kasperczuk 2002, Kubes 2020, and Trunk 1975. P2 target review candidate is
Lindemuth 1982. Alexiou 2002 is a spectroscopy/method reference, Sadowski 2008
is review/source-map context, and `symons1994.pdf` is stage-only after
first-page review showed it is an out-of-scope JSTOR social-science review.
This is a backlog only; every candidate still needs source-line review, typed
target extraction, unit normalization, uncertainty handling, and digitization
review where plotted data are involved.

May 12 source-validation report:
`docs/USER_PDF_MAY12_SOURCE_VALIDATION_2026_05_12.md` and
`docs/USER_PDF_MAY12_SOURCE_VALIDATION_2026_05_12.json`.

The validation report checks local source identity, SHA consistency, KR mapping,
text parity, source-fidelity mapping, stage-only status, and target-candidate
classification. Result: 28 promoted source records checked, 7 stage-only
records checked, 5 source-validated target-extraction candidates, 23
source-validated method/context records, and 0 validation failures. It also
records the Trunk 1975 false-match repair and confirms the Kortanek 2014
source-fidelity annotation was repaired back to the 2026-05-11 Kortanek source.

## Promoted Local Method Sources

These method references were previously acquisition candidates but have now
been converted into paired `KnowledgeReference/` Markdown/JSON records and
passed schema plus PDF-text parity checks.

| Source | Local KR status | Usable scope | Guardrail |
| --- | --- | --- | --- |
| Randall J. LeVeque, *Finite Volume Methods for Hyperbolic Problems*, Cambridge University Press, 2002 | Promoted to `KnowledgeReference/finite-volume-methods-for-hyperbolic-problems.md` and `KnowledgeReference/finite-volume-methods-for-hyperbolic-problems.json`; original PDF SHA-256 `b3adec0d3616dbde57a5522cfce1861890887d7c03a2232d2136cb94c9bac1d5`; schema and text-parity checks passed on 2026-05-09 | Finite-volume conservation-law method authority, CFL/shock-capturing/limiter/convergence review, and generic Tier 3 numerical-method verification support | Not DPF experimental validation evidence; does not close PF-1000 same-scope Tier 4 spatial validation, Tier 5 neutron validation, or predictive scientific readiness. |
| Eleuterio F. Toro, *Riemann Solvers and Numerical Methods for Fluid Dynamics*, 3rd ed., Springer, 2009 | Promoted to `KnowledgeReference/toro2009-433cd861.md` and `KnowledgeReference/toro2009-433cd861.json`; original PDF SHA-256 `433cd86174534f8d28702a173c247f29b0028c8153526b2282951ba9c458405e`; text-parity checks passed on 2026-05-11; Markdown is chunked into 30 page-range files under `KnowledgeReference/chunks/toro2009-433cd861/` | Riemann-solver, approximate-solver, TVD/limiter, shock-tube, and finite-volume method review for implementation verification | Method authority only. It does not validate DPF physics, PF-1000 target data, current waveforms, spatial diagnostics, or neutron predictions. Bibliographic title metadata and method-target mapping still need cleanup. |

## Textbooks and Monographs to Acquire

These are not current Tier 5 experimental-validation blockers in the same way
as the missing PF-1000/DPF papers below. They should be acquired when the
corresponding method area moves from scaffolded or partially supported to
source-backed closure.

| Priority | Source | Acquisition link | Local status | Acquire before |
| --- | --- | --- | --- | --- |
| P1-method | I. H. Hutchinson, *Principles of Plasma Diagnostics*, 2nd ed., Cambridge University Press, 2002 | <https://www.cambridge.org/core/books/principles-of-plasma-diagnostics/93578947341481B0F9169D598D535E76> | exact local book not found | Closing diagnostic-method authority for interferometry, scattering/Thomson design, fast-ion/fusion-product diagnostics, detector uncertainty, or Tier 5 diagnostic acceptance criteria. |
| P2-method | Jeffrey P. Freidberg, *Ideal MHD*, Cambridge University Press, 2014 | <https://www.cambridge.org/core/books/ideal-mhd/contents/D5A56CC603F0FFDC574F38FAA68DD3C3> | exact local book not found | Making strong claims about ideal-MHD applicability, MHD equilibrium, or pinch stability limits. |
| P2-method | Hans Goedbloed, Rony Keppens, Stefaan Poedts, *Magnetohydrodynamics of Laboratory and Astrophysical Plasmas*, Cambridge University Press, 2019 | <https://www.cambridge.org/core/books/magnetohydrodynamics-of-laboratory-and-astrophysical-plasmas/8E0AEB8F1298B97204D5B254DE8877F0> | exact local book not found | Reviewing advanced cylindrical/toroidal MHD, flows, dissipation, and computational-MHD assumptions. |
| P2-method | C. K. Birdsall and A. B. Langdon, *Plasma Physics via Computer Simulation*, CRC Press/Routledge, 1991/2004 | <https://www.routledge.com/Plasma-Physics-via-Computer-Simulation/Birdsall-Langdon-Langdon/p/book/9780750310253> | exact local book not found | Promoting PIC or kinetic-validation scaffolds beyond placeholder status. |
| P3-method | Hans R. Griem, *Principles of Plasma Spectroscopy*, Cambridge University Press, 1997 | <https://www.cambridge.org/core/books/principles-of-plasma-spectroscopy/preface/107670FAAC73228F679C693CA8003F49> | exact local book not found | Adding or validating spectroscopic density, temperature, Stark broadening, or line-radiation diagnostics. |
| P3-method | George B. Rybicki and Alan P. Lightman, *Radiative Processes in Astrophysics*, Wiley/Wiley-VCH | <https://www.wiley-vch.de/de/fachgebiete/naturwissenschaften/physik-11ph/astronomie-u-astrophysik-11ph1/radiative-processes-in-astrophysics-978-3-527-41449-9> | local partial candidate found outside KR with frontmatter and Chapter 1 material: `archive_reference_OLD/references/papers/textbooks/rybicki-lightman-1979-radiative-processes.pdf`; SHA-256 `fcff04d2c6c1c77855192cd107ad144497cc7637706a66278658af1a5f23a08d`; full source still not found | Revisiting bremsstrahlung, radiative-transfer, or radiation-loss implementation beyond the current local formula checks. |
| Optional | Stefano Atzeni and Juergen Meyer-ter-Vehn, *The Physics of Inertial Fusion*, Oxford University Press, 2004 | <https://academic.oup.com/book/27812> | exact local book not found | Expanding high-energy-density, radiation/EOS, burn, or inertial-fusion context. This is useful context but not DPF-specific experimental evidence. |

If budget or access is constrained, acquire in this order for the current work:
Hutchinson, then either Freidberg or Goedbloed depending on whether the next
Track A closure is ideal-MHD limits or broader MHD/numerical method review.
LeVeque and Toro no longer need acquisition, but both still need target/test
mapping where they are used as method authority.

Current queue source:
`scientific_closure_source_acquisition_queue()` and
`docs/SCIENTIFIC_CLOSURE_SOURCE_QUEUE.md`.

Current local exact-PDF audit status:
the prior six-paper "not found" audit has been partially superseded by the
2026-05-11 user intake batch. Klir 2011 TOF detector calibration,
Szydlowski/Sadowski/Scholz/PF-1000 2004 fast ions/neutrons, Catenacci 2020,
Springham 2021, and Cikhardtova 2015 now have local KR text records. Jednorog
2017 PF-1000 neutron monitoring still needs exact-source acquisition unless a
local copy is later identified.

Current A14 digitization status:
`docs/A14_CROP_BOUNDARY_REVIEW_2026_05_11.md` now records crop-boundary QA for
the 36 local crop candidates: 21 figure crops are ready for draft
axis/numeric extraction, 9 are manual-review diagram/image crops, 0 need crop
adjustment before extraction, and 6 table crops have draft packets but remain
review-blocked. This does not change acquisition status or validation
acceptance; every A14 item still has `accepted_for_validation=false`. The
first three figure calibration scaffolds now exist in
`KnowledgeReference/digitization/a14-2026-05-11-axis-calibration-draft-packets.json`
for Cikhardtova 2015 Fig. 6, Klir 2011 Fig. 2, and Springham 2021 Fig. 5,
and the first numeric Springham Fig. 5 mono-energetic curve draft now exists in
`KnowledgeReference/digitization/a14-2026-05-11-springham-fig5-monoenergetic-draft-packet.json`,
with draft round-trip RMS residual `0.002049609754498783 px`; it remains
review-blocked and is not validation evidence. The reviewer-facing handoff
`docs/A14_INDEPENDENT_REVIEW_HANDOFF_2026_05_11.md` now lists 9 reviewable
draft packets and 3 context-only axis scaffolds, all still
`accepted_for_validation=false`; the verification gate now checks declared
local source-PDF hashes before any accepted review can pass.
Cikhardtova Fig. 6 now has a blocker report at
`docs/A14_CIKHARDTOVA_FIG6_EXTRACTION_BLOCKER_2026_05_11.md`; it needs manual
or vector-assisted curve separation before numeric arrays can be drafted.
The generated A14 backlog is in
`docs/A14_REMAINING_EXTRACTION_BACKLOG_2026_05_11.md` and currently reports
18 ready-not-started crops, 9 manual-review crops, 1 blocked crop, and 0
accepted validation items.

Extended local-machine search status:
`docs/LOCAL_SOURCE_SEARCH_2026_05_09.md` searched likely local source pools
outside `KnowledgeReference/`, including Downloads, DPF-U2, old paper archives,
and converted text stores. It found no exact local copy of the six paper
targets. It found LeVeque as a local method-source candidate, Toro as a local
reading sample/excerpt, and Rybicki-Lightman as a local partial candidate with
frontmatter and Chapter 1 material. `docs/LOCAL_METHOD_SOURCE_REVIEW_2026_05_09.md`
captures the intake review. LeVeque has since been promoted to
`KnowledgeReference/finite-volume-methods-for-hyperbolic-problems.md` and can
be used only for the method scope listed above. Toro has since been promoted to
`KnowledgeReference/toro2009-433cd861.md` for method-scope review.
Rybicki-Lightman remains a full-source acquisition candidate.

## Web And Google Scholar Review - 2026-05-09

Direct automated access to `scholar.google.com` returned HTTP 403 during this
review. To keep the review reproducible without scraping Scholar, each item
below records a Google Scholar title query plus publisher, institutional,
repository, or indexed scholarly pages that were reachable from normal web
search. Scholar hit counts and "cited by" counts are intentionally not recorded.

These web pages are acquisition leads only. They do not change the local
source-of-truth state until the exact document is acquired, hashed, reviewed
into `KnowledgeReference/`, and mapped into target/digitization records.

| Source | Google Scholar query | Verified web pages found | Acquisition action |
| --- | --- | --- | --- |
| Klir et al. 2011 TOF detector calibration | <https://scholar.google.com/scholar?q=%22Fusion+neutron+detector+for+time-of-flight+measurements+in+z-pinch+and+plasma+focus+experiments%22> | PubMed metadata; DOI/AIP/Silverchair full-text route; IPPLM 2011 publication listing | Local KR text now promoted: `KnowledgeReference/fusion-neutron-detector-for-time-of-flight-measurements-in-z-pinch-and-plasma-focus-214fbdae.md`; typed target extraction started as `klir_2011_tof_detector_response_targets()` and pages rendered in `docs/TARGET_EXTRACTION_DIGITIZATION_2026_05_11.md`; four unreviewed crop candidates now cover Figs. 1-4, but detector response curves, geometry mapping, uncertainty, and review still open. |
| Szydlowski/Sadowski/Scholz/PF-1000 team 2004 fast ions/neutrons | <https://scholar.google.com/scholar?q=%22Measurements+of+fast+ions+and+neutrons+emitted+from+PF-1000+plasma+focus+device%22> | ScienceDirect abstract/landing page; cited in open review/reference pages | Local KR text now promoted: `KnowledgeReference/doi-10-1016-j-vacuum-2004-07-040-6de67a98.md`; typed target extraction started as `pf1000_szydlowski_fast_ion_neutron_targets()` and pages rendered in `docs/TARGET_EXTRACTION_DIGITIZATION_2026_05_11.md`; five unreviewed crop candidates now cover Figs. 1-5, but OCR-suspect units, spectra, anisotropy, and CR-39 figures still need axis extraction, residual checks, and independent review. |
| Catenacci et al. 2020 neutron time-energy tomography | <https://scholar.google.com/scholar?q=%22Tomographic+Reconstruction+of+the+Neutron+Time-Energy+Spectrum+from+a+Dense+Plasma+Focus%22> | PNNL publication record; IEEE DOI; OSTI related-index page | Local KR text now promoted: `KnowledgeReference/tomographic-reconstruction-of-the-neutron-time-energy-spectrum-from-a-dense-plasma-focus-b78f1154.md`; typed target extraction started as `nnss_dpf_neutron_time_energy_tomography_targets()` and pages rendered in `docs/TARGET_EXTRACTION_DIGITIZATION_2026_05_11.md`; twelve unreviewed crop candidates now cover Figs. 1-8 and Tables I-IV; Tables I-IV now have draft packets in `KnowledgeReference/digitization/a14-2026-05-11-table-draft-packets.json`, but plots/tomography arrays, residual checks, and independent review remain open. |
| Springham et al. 2021 Zr/Be activation | <https://scholar.google.com/scholar?q=%22Plasma+focus+neutron+energy+and+anisotropy+measurements+using+zirconium-beryllium+pair+activation+detectors%22> | ScienceDirect landing page; IAEA CRP page listing the DOI | Local KR text now promoted: `KnowledgeReference/nuclear-inst-and-methods-in-physics-research-a-988-2021-164830-bc8edab3.md`; typed target extraction started as `nx3_springham_zrbe_activation_targets()` and pages rendered in `docs/TARGET_EXTRACTION_DIGITIZATION_2026_05_11.md`; nine unreviewed crop candidates now cover Figs. 1-7 and Tables 1-2; Tables 1-2 now have draft packets in `KnowledgeReference/digitization/a14-2026-05-11-table-draft-packets.json`, but pressure-sweep curves, MCNP response, uncertainty, and review remain open. |
| Jednorog et al. 2017 PF-1000 neutron monitoring | <https://scholar.google.com/scholar?q=%22A+new+concept+of+fusion+neutron+monitoring+for+PF-1000+device%22> | Sciendo open article page; Nukleonika public PDF link already identified | Acquire the public PDF, hash it, and review into KR; this looks like the quickest paper acquisition among the current P2 blockers. |
| Cikhardtova et al. 2015 linear-density timing | <https://scholar.google.com/scholar?q=%22Temporal+distribution+of+linear+densities+of+the+plasma+column+in+a+plasma+focus+discharge%22> | Sciendo open article page; CTU publication pages; CTU thesis appendix listing | Local KR text now promoted: `KnowledgeReference/cikhardtova-plazma-indd-9dfed6c0.md`; typed target extraction started as `pf1000_cikhardtova_linear_density_motion_targets()` and pages rendered in `docs/TARGET_EXTRACTION_DIGITIZATION_2026_05_11.md`; six unreviewed crop candidates now cover Figs. 1-6, but axis/table extraction, residual checks, and independent review are still open. |

Community/discovery pages reviewed:

- ResearchGate and Academia-style pages were found for several exact or related
  papers. Treat those as discovery leads only unless the upload is clearly
  author-provided and license-compatible.
- CTU FEE, IPPLM, PNNL, OSTI, IAEA/INIS, PubMed, Sciendo, ScienceDirect, MDPI,
  Nukleonika, J-GLOBAL, and ICDMP pages are useful verified metadata or
  acquisition leads. They still do not substitute for local KR ingestion.

## Physics-Gap-Driven Search - 2026-05-09

This second web pass was driven by the physics we still need to learn and
validate, not only by the titles already in the acquisition queue. It searched
for sources that could close or sharpen these remaining validation gaps:

- Akel/PF-1000 S1/S2 waveform and current-dip acceptance.
- Tier 2 phase timing: axial, radial, pinch, stagnation, and current-derivative
  timing with uncertainty.
- Tier 4 spatial validation: density, magnetic/EM, and temperature evidence in
  a same-scope packet.
- Tier 5 neutron validation: scalar yield, timing, spectrum, anisotropy,
  detector/activation response, mechanism separation, and uncertainty.
- Circuit-field coupling: current-sheath structure, magnetic probes,
  inductance, `dL/dt`, back-EMF, Poynting/energy balance, and handoff timing.
- Physics-fidelity limits: 2T/resistive MHD, ionization, radiation cooling,
  Hall/kinetic effects, beam-target coupling, instability/reconnection, and
  model-form uncertainty.

Search result:

| Physics gap | Best acquisition or local-review leads | Why it matters | Current action |
| --- | --- | --- | --- |
| Phase plus density timing | Zielinska/Paduch/Scholz 2011; Kubes et al. 2009; Malir et al. 2022 | These are the strongest leads for 16-frame interferometry, pinch evolution, implosion velocity, phase timing, and density history. | Zielinska 2011 and Kubes 2009 now have local KR text records; extract frame timing, density profiles, velocity, and uncertainty. Malir 2022 remains an acquisition/review lead unless already located elsewhere. |
| Magnetic/EM spatial closure | Kubes et al. 2012 magnetic-probe/neutron/interferometry correlation; Krauz et al. 2012; Mitrofanov et al. 2014; Munzar et al. 2021 | These directly address magnetic probes, current-sheath structure, azimuthal/axial magnetic fields, and field coupling to density/neutron evidence. | Kubes 2012, Krauz 2012, and Munzar 2021 now have local KR text records; extract probe geometry, B-field/current profiles, timing, and uncertainty. Mitrofanov 2014 remains an acquisition lead. |
| Temperature and spectroscopy | Jakubowska et al. 2011; Skladnik-Sadowska et al. 2011; local Zaloga et al. 2015 KR record | Tier 4 cannot close with density-only evidence; spectroscopy sources may provide temperature/effective-temperature and Stark-broadening uncertainty context. | Skladnik-Sadowska 2011 now has a local KR text record; extract spectroscopy method and uncertainty. Jakubowska 2011 remains an acquisition lead; use local Zaloga 2015 only for its gas-puff DPF-1000U scope unless same-scope compatibility is proven. |
| Neutron anisotropy/spectrum/mechanism | Krasa et al. 2008; Klir et al. 2011 APL; Klir et al. 2012 PPCF; Szydlowski/Sadowski/Scholz 2004; Catenacci 2020; Springham 2021; Rezac et al. 2012; Jednorog 2017; Jednorog 2015 radioindium | Tier 5 needs more than scalar yield: anisotropy, spectrum, detector response, activation response, mechanism separation, and uncertainty. | Krasa 2008, both Klir 2011 papers, Szydlowski 2004, Catenacci 2020, Springham 2021, and Jednorog 2015 now have local KR text records; extract detector/activation geometry, spectra, anisotropy, mechanism separation, and uncertainty. Rezac 2012 and Jednorog 2017 remain acquisition leads. |
| Energy/circuit coupling | Kubes/Klir et al. 2012 energy transformations; Kortanek/Kubes et al. 2014 energy balance; Malir et al. 2022 | These are the strongest leads for field/current/energy coupling, sheath-current fraction, inductance and kinetic-energy estimates. | Kubes/Klir 2012 and Kortanek/Kubes 2014 now have local KR text records; extract electrical waveform definitions, energy terms, inductance, current fraction, and uncertainty. Malir 2022 remains an acquisition/review lead. |
| Physics-fidelity and model-form limits | Stepniewski 2004 PF-1000 MHD modelling; Schmidt et al. 2014 fully kinetic MJ DPF; Lee/Saw/Akel/Kubes/Paduch 2016 radiative cooling; local 2019 PF-1000 evolution KR | These define or bound what reduced MHD, Lee, resistive MHD, kinetic/PIC, radiation, and reconnection claims can support. | Stepniewski 2004, Schmidt 2014, and Lee et al. 2016 now have local KR text records. Treat them as model-form and scope-limit evidence after extraction; do not use as same-scope experimental validation unless the source contains matching observables. |

New or re-ranked physics-gap leads from this pass:

| Priority | Source | Acquisition / metadata link | Local status | Primary validation gap |
| --- | --- | --- | --- | --- |
| P1-phase-spatial | E. Zielinska, M. Paduch, M. Scholz, "Sixteen-frame interferometer for a study of a pinch dynamics in PF-1000 device," *Contributions to Plasma Physics* 51, 279-283, 2011 | <https://doi.org/10.1002/ctpp.201000047>; IPPLM listing <https://www.ifpilm.pl/en/ipplmpublications/articles-in-scientific-journals/186-articles-in-scientific-journals-ipplm-2011> | local KR text promoted: `KnowledgeReference/sixteenframe-interferometer-for-a-study-of-a-pinch-dynamics-in-pf1000-device-f8dc9d1b.md`; target extraction still needed | Tier 2 phase timing, Tier 4 density timing, interferometry uncertainty. |
| P1-phase-neutron | P. Kubes et al., "Interferometric Study of Pinch Phase in Plasma-Focus Discharge at the Time of Neutron Production," *IEEE Transactions on Plasma Science* 37, 2191-2196, 2009 | <https://doi.org/10.1109/TPS.2009.2030576>; CTU listing <https://fel.cvut.cz/cs/fakulta/lide/780-karel-rezac/publikace> | local KR text promoted: `KnowledgeReference/ieee-transactions-on-plasma-science-vol-37-no-11-november-2009-dee3911f.md`; metadata title cleanup and target extraction still needed | Pinch-frame timing, density/neutron correlation, phase validation. |
| P1-field-neutron | P. Kubes et al., "Correlation of magnetic probe and neutron signals with interferometry figures on the plasma focus discharge," *Plasma Physics and Controlled Fusion* 54, 105023, 2012 | <https://doi.org/10.1088/0741-3335/54/10/105023>; INIS/CTU metadata | local KR text promoted: `KnowledgeReference/correlation-of-magnetic-probe-and-neutron-signals-with-interferometry-figures-on-the-plasm-7fd84db7.md`; target extraction still needed | Field-coupling, magnetic-probe timing, neutron/interferometry correlation. |
| P1-current-sheath | V. Krauz et al., "Experimental study of the structure of the plasma-current sheath on the PF-1000 facility," *Plasma Physics and Controlled Fusion* 54, 025010, 2012 | <https://doi.org/10.1088/0741-3335/54/2/025010> | local KR text promoted: `KnowledgeReference/experimental-study-of-the-structure-of-the-plasma-current-sheath-on-the-pf-1000-facility-705bcc83.md`; target extraction still needed | Current-sheath density/current separation, magnetic/EM spatial closure. |
| P1-current-sheath | K. N. Mitrofanov et al., "Study of the fine structure of the plasma current sheath and magnetic fields in the axial region of the PF-1000 facility," *Plasma Physics Reports* 40, 623-639, 2014 | <https://doi.org/10.1134/S1063780X14070071>; CTU metadata | referenced by local Malir 2024 KR; exact source not promoted | Shock-wave/magnetic-piston separation, current transport, magnetic field timing. |
| P1-current-energy | J. Malir et al., "Dynamics of implosion phase of modified plasma focus studied via laser interferometry and electrical measurements," *Physics of Plasmas* 29, 092706, 2022 | <https://doi.org/10.1063/5.0098124>; CTU/IPPLM metadata | cited by local KR records; exact source not promoted | Implosion velocity, imploding mass, sheath-current fraction, inductance and kinetic-energy coupling. |
| P1-neutron-anisotropy | J. Krasa et al., "Anisotropy of the emission of DD-fusion neutrons caused by the plasma-focus vessel," *Plasma Physics and Controlled Fusion* 50, 125006, 2008 | <https://doi.org/10.1088/0741-3335/50/12/125006>; CTU metadata | local KR text promoted: `KnowledgeReference/anisotropy-of-the-emission-of-dd-fusion-neutrons-caused-by-the-plasma-focus-vessel-527cc533.md`; target extraction still needed | Vessel-scattering correction, neutron anisotropy, TOF-spectrum interpretation. |
| P1-neutron-response | S. Jednorog et al., "Radioindium and determination of neutron radial asymmetry for the PF-1000 plasma focus device," *Journal of Radioanalytical and Nuclear Chemistry* 303, 941-947, 2015 | <https://doi.org/10.1007/s10967-014-3444-z>; AGH BaDAP metadata | local KR text promoted: `KnowledgeReference/radioindium-and-determination-of-neutron-radial-asymmetry-46bcae32.md`; target extraction still needed | Radial neutron asymmetry, activation geometry, Monte Carlo activation factors. |
| P2-neutron-mechanism | D. Klir et al., "Experimental evidence of thermonuclear neutrons in a modified plasma focus," *Applied Physics Letters* 98, 071501, 2011 | <https://doi.org/10.1063/1.3555447>; Starfos metadata | local KR text promoted: `KnowledgeReference/experimental-evidence-of-thermonuclear-neutrons-in-a-modified-plasma-focus-ad181105.md`; target extraction still needed | Thermonuclear-vs-beam-target mechanism separation and neutron spectrum width context. |
| P2-neutron-spectrum | P. Kubes et al., "Determination of Deuteron Energy Distribution From Neutron Diagnostics in a Plasma-Focus Device," *IEEE Transactions on Plasma Science* 37, 83-87, 2009 | CTU/Starfos metadata; DOI not found in this pass | exact source not promoted | Deuteron energy distribution from multi-direction neutron TOF diagnostics. |
| P2-energy-coupling | P. Kubes et al., "Energy Transformations in Column of Plasma Focus Discharges with Megaampere Currents," *IEEE Transactions on Plasma Science* 40, 481-486, 2012 | <https://doi.org/10.1109/TPS.2011.2178866>; CTU metadata | local KR text promoted: `KnowledgeReference/ieee-transactions-on-plasma-science-vol-40-no-2-february-2012-19056f65.md`; metadata title cleanup and target extraction still needed | Energy partition, current/voltage coupling, neutron/x-ray/interferometry energy balance. |
| P2-field-map | V. Munzar et al., "Mapping of azimuthal B-fields in Z-pinch plasmas using Z-pinch-driven ion deflectometry," *Physics of Plasmas* 28, 062702, 2021 | <https://doi.org/10.1063/5.0040515>; CTU metadata | local KR text promoted: `KnowledgeReference/mapping-of-azimuthal-b-fields-in-z-pinch-plasmas-using-z-pinch-driven-ion-deflectometry-c041d3f7.md`; target extraction still needed | Azimuthal B-field mapping, ion-deflectometry method, magnetic-field validation design. |
| P2-temperature | K. Jakubowska et al., "Optical emission spectroscopy of plasma streams in PF-1000 experiments," *Nukleonika* 56, 125-129, 2011 | <https://www.nukleonika.pl/www/back/full/vol56_2011/v56n2p125f.pdf> | exact source not promoted | Stark-broadening electron density, excitation temperature, spectroscopy uncertainty. |
| P2-temperature | E. Skladnik-Sadowska et al., "Optical spectroscopy of free-propagating plasma and its interaction with tungsten targets in PF-1000 facility," *Contributions to Plasma Physics* 51, 2011 | <https://doi.org/10.1002/ctpp.201000046>; IPPLM listing | local KR text promoted: `KnowledgeReference/optical-spectroscopy-of-freepropagating-plasma-and-its-interaction-with-tungsten-targets-i-3a20181e.md`; target extraction still needed | Spectroscopy method, free-stream plasma parameters, target-interaction context. |
| P2-model-form | W. Stepniewski, "MHD numerical modelling of the plasma focus phenomena," *Vacuum* 76, 51-55, 2004 | <https://doi.org/10.1016/j.vacuum.2004.05.019>; ScienceDirect metadata | local KR text promoted: `KnowledgeReference/doi-10-1016-j-vacuum-2004-05-019-f931cb0b.md`; metadata title cleanup and target extraction still needed | PF-1000 nonideal MHD, ionization, transport, anomalous resistivity, model-form limits. |
| P2-kinetic | A. Schmidt et al., "Fully kinetic simulations of megajoule-scale dense plasma focus," *Physics of Plasmas* 21, 102703, 2014 | <https://doi.org/10.1063/1.4897192>; INIS metadata | local KR text record promoted: `KnowledgeReference/fully-kinetic-simulations-of-megajoule-scale-dense-plasma-focus-3f439245.md` / `.json`; target extraction and figure/table review still needed | Kinetic/PIC limits, neutron spectra, anisotropy, spot size, time history, model-form uncertainty. |
| P2-radiation | S. Lee et al., "Conditions for Radiative Cooling and Collapse in the Plasma Focus Illustrated With Numerical Experiments on PF1000," *IEEE Transactions on Plasma Science* 44, 165-173, 2016 | <https://doi.org/10.1109/TPS.2015.2497269>; CTU metadata | local KR text promoted: `KnowledgeReference/this-article-has-been-accepted-for-inclusion-in-a-future-issue-of-this-journal-content-is-d1825800.md`; metadata title cleanup and target extraction still needed | Radiation-cooling scope limits, high-Z/puff gas guardrails, model-form closure. |

Local KR records that this pass flagged for extraction/review before acquiring
more adjacent material:

| Local KR record | Why revisit it | Guardrail |
| --- | --- | --- |
| `KnowledgeReference/evolution-of-a-pinch-column-during-the-acceleration-of-fast-electrons-and-deuterons-in-a-plasma.md` | Already contains PF-1000 evolution, neutron diagnostics, interferometry, soft X-ray, optical spectroscopy, magnetic probes, electron/ion measurements, and instability/reconnection interpretation. | Useful for physics-fidelity and mechanism context; do not treat as Akel 16 kV shot evidence or as a complete same-scope Tier 4/5 packet without extraction and scope matching. |
| `KnowledgeReference/investigation-of-the-optical-spectra-emitted-from-plasma-streams-is-of-primary-importance-for.md` | Already contains DPF-1000U gas-puff optical spectroscopy and electron-density context. | Useful for spectroscopy method and temperature/density context only in its gas-puff DPF-1000U scope. |
| `KnowledgeReference/malir-2024-interferometry-dpf.md` | Already contains PF-1000 density profiles versus MHD and a bibliography pointing to many missing same-scope diagnostics. | Use its coded density-profile targets where already extracted; acquire primary references before using cited material for new phase, field, or neutron targets. |

## Module-Coverage Source Search - 2026-05-09

This third web pass was driven by code surfaces that were not fully covered by
the PF-1000/Track A validation queue. It looked for acquisition leads for
modules that are scaffolded or partially implemented but still need exact local
source review before we can use them for source-backed design, predictive, or
validation claims.

These are still external acquisition leads only. They do not change the local
scientific status of any module until the exact document or dataset is acquired,
hashed, reviewed into `KnowledgeReference/`, and linked to tests, target
records, or validation certificates.

| Module area | Unconfirmed code surface | Acquisition / metadata leads found | Required action before closure |
| --- | --- | --- | --- |
| `src/dpf/atomic/ionization.py` | Saha/CR ionization is implemented with Lotz ionization, Seaton radiative recombination, Burgess/dielectronic context, and NIST ionization potentials, but source table/formula review is incomplete. | Lotz 1967 is now local as `KnowledgeReference/an-empirical-formula-for-the-electron-impact-ionization-cross-section-b5fde85c.md`; Seaton 1959 is local as `KnowledgeReference/1959mnras-119-81s-586d5f74.md` but needs OCR/text recovery; Burgess/Seaton coronal-ionization context `10.1093/mnras/127.5.355`; NIST ASD/SRDB 78 `10.18434/T4W30F`. | OCR/review Seaton, extract Lotz formula/table provenance, acquire Burgess/NIST authoritative exports, and add table/hash provenance for Cu/W/H/D. |
| `src/dpf/radiation/line_radiation.py` | Line cooling remains explicitly empirical; current piecewise power laws are not traceable to Post 1977 or a verified table source. | Del Zanna 2015 CHIANTI is now local as `KnowledgeReference/chianti-an-atomic-database-for-emission-lines-version-8-26ccc010.md`; Puetterich 2019 impurity-tolerance source is local as `KnowledgeReference/determination-of-the-tolerable-impurity-concentrations-in-a-fusion-reactor-38957579.md`; Post et al. 1977, ADAS/Summers 2004, Dere et al. 2009, and exact CHIANTI data exports remain leads. | Replace or calibrate the empirical fits from reviewed tabular data; record units, density/temperature limits, charge-state model, interpolation rules, and regression tolerances. |
| `src/dpf/atomic/ablation.py` | Electrode ablation is a constant-efficiency source with threshold support, but efficiency values and shielding/fluence limits remain empirical. | Vikhrev 2007 is now local as `KnowledgeReference/neutron-generation-from-z-pinches-6fb2ee87.md`; Bruzzone & Aranchuk 2003 J. Phys. D, Lee & Serban 1996 DPF modelling/ablation, and related DPF EUV/electrode loading literature remain leads. | Review whether Vikhrev has applicable erosion/ablation content; otherwise acquire exact ablation/erosion sources and extract material, pulse length, fluence/current-density ranges, shielding assumptions, and uncertainty. |
| `src/dpf/turbulence/anomalous.py` | Anomalous resistivity thresholds and alpha ranges are implemented but remain model-form assumptions for DPF unless KR-backed limits are added. | Buneman 1959 is now local as `KnowledgeReference/dissipation-of-currents-in-ionized-media-ad2e2910.md`; Shumlak stability papers are local for shear-flow limits, but Davidson & Gladd 1975, finite-beta LHDI, Haines 2011 DPF review, and Krall & Trivelpiece remain acquisition leads. | Extract threshold definitions and limits from Buneman where applicable; acquire remaining instability sources/textbooks; keep outputs as model-form/diagnostic until source-backed. |
| `src/dpf/diagnostics/scaling_laws.py` | Neutron, energy, Bennett, and pinch-radius scaling outputs are diagnostic estimates and not solver validation. | Lee & Saw 2008 neutron scaling is now local as `KnowledgeReference/original-paper-dbfa1caa.md`; Lee/Saw APL pinch-current limitation is local as `KnowledgeReference/pinch-current-limitation-effect-in-plasma-focus-61bd14af.md`; Soto et al. 2010 and other scaling reviews remain leads. | Clean metadata titles, extract current definitions, fitted ranges, saturation limits, calibration devices, and uncertainty; prevent use as validation targets unless same-scope data are acquired. |
| `src/dpf/diagnostics/pb11_yield.py` | p-B11 reactivity table is implemented, but Nevins/Swain/Rider/Becker source tables are not in the verified corpus; DPF feasibility remains unsupported. | Nevins & Swain 2000 `10.1088/0029-5515/40/4/310`; Rider 1997 `10.1063/1.872556`; Sikora & Weller 2016 open article `10.1007/s10894-016-0069-y`; Becker et al. 1987 S-factor lead. | Acquire reactivity/cross-section tables and uncertainty; add a KR-backed reactivity dataset; keep p-B11 output as `reactivity_table_estimate`, not a DPF feasibility validation. |
| `src/dpf/diagnostics/thomson_scattering.py` | Synthetic Thomson scattering uses Salpeter/Faddeeva form but diagnostic-method authority and DPF applicability are not fully local. | Salpeter 1960 `10.1103/PhysRev.120.1528`; Salpeter magnetic-field extension `10.1103/PhysRev.122.1663`; Hutchinson diagnostic chapter `10.1017/CBO9780511613630.009`; Sheffield/Froula/Glenzer/Luhmann book `10.1016/C2009-0-20048-1`. | Acquire Hutchinson and/or Sheffield; extract geometry, collective/noncollective regimes, detector response, fitting uncertainty, and DPF applicability guardrails. |
| `src/dpf/diagnostics/xray_imaging.py` and X-ray diagnostics | Synthetic pinhole images use NRL-backed bremsstrahlung and a simple filter response; line radiation and detector/filter transfer are not source-backed. | Shan et al. 2004 DPF X-ray diagnostics lead; NIST pinhole-camera RSI metadata lead; GORGON/synthetic gated X-ray image literature; Rybicki-Lightman remains partial/local only. | Acquire DPF X-ray diagnostic and detector/filter response sources; verify Abel geometry, filter transmission, detector energy window, and line-radiation coupling assumptions. |
| `src/dpf/diagnostics/instability.py` and `src/dpf/diagnostics/shear_stabilization.py` | m=0/Kadomtsev, tearing, and sheared-flow margins are diagnostic estimates; source review is incomplete for Z-pinch applicability and thresholds. | Shumlak & Hartman 1995 is now local as `KnowledgeReference/sheared-flow-stabilization-of-the-m-8e0bd47a.md`; Shumlak et al. 2001 is local as `KnowledgeReference/volume-87-number-20-fd194438.md`; Kruskal-Schwarzschild 1954, Kadomtsev 1966, Furth/Killeen/Rosenbluth 1963, and Biskamp remain leads. | Clean metadata titles, document mode definitions, wall/profile assumptions, shear thresholds, and DPF applicability limits before using as validation or design closure. |
| `src/dpf/experimental/civ_breakdown.py` | CIV/Paschen breakdown uses Alfven/Brenning/Danielsson/Haerendel-style theory plus gas coefficients; DPF breakdown applicability is not fully sourced. | Brenning 1992 CIV review lead; Danielsson 1970 `10.1063/1.1693235`; critical-velocity review/lab-experiment leads; Meek & Craggs/Lieberman-Lichtenberg gas-coefficient sources; NIST ASD ionization data. | Acquire CIV and gas-breakdown sources; extract coefficients, units, pressure-distance ranges, magnetization conditions, and DPF-specific breakdown guardrails. |
| `src/dpf/experimental/pic/hybrid.py` and kinetic scaffolds | PIC/hybrid collision kernel is implemented and tested, but kinetic-DPF validation and beam slowing-down/source terms are not source-backed locally. | Nanbu 1997 `10.1103/PhysRevE.55.4642`; Perez et al. 2012 `10.1063/1.4742167`; Schmidt/Tang/Welch PRL 2012 `10.1103/PhysRevLett.109.205003`; Schmidt et al. 2014 `10.1063/1.4897192`; Birdsall/Langdon textbook. | Schmidt et al. 2014 is now a local KR text record; remaining PIC method papers/textbook and PRL 2012 still need acquisition or target extraction before validation use. |
| `src/dpf/sheath/bohm.py` | Bohm/sheath support relies on textbook/formulary authority and needs explicit KR-linked guardrails before broader sheath-model claims. | NRL formulary is local; Lieberman & Lichtenberg textbook; Riemann/Bohm sheath review candidates. | Acquire sheath textbook/review material if this module becomes closure-critical; otherwise keep as method-support/diagnostic utility. |
| `src/dpf/verification/sedov_cylindrical.py` and `src/dpf/validation/sedov_exact.py` | Cylindrical Sedov verification has local Kamm/Timmes support, but the gamma=5/3 cylindrical normalization still needs explicit extraction or quadrature. | Local Kamm & Timmes source should be mined first; Taylor 1950 `10.1098/rspa.1950.0049`; Sedov/Taylor-von-Neumann historical leads; Lin/Sakurai cylindrical-blast leads. | Prefer local Kamm/Timmes extraction and alpha quadrature before new acquisition; acquire historical sources only if primary-authority traceability is required. |
| `src/dpf/athena_wrapper/`, `src/dpf/athenak_wrapper/`, and backend dispatch | Athena/AthenaK backends are infrastructure/method surfaces; they need external code-method authority and local backend verification before trustable comparison claims. | Athena method paper `10.1086/588755`; Stone & Gardiner unsplit Godunov `10.1016/j.newast.2008.06.003`; Athena++ method paper `10.3847/1538-4365/ab929b`; AthenaK 2026 lead `10.3847/1538-4365/ae3717`. | Acquire method papers and exact external-code version records; add backend build/version manifests, canonical test problems, and comparison certificates before using backend agreement as validation evidence. |
| `src/dpf/ai/` and surrogate/optimization surfaces | ML/surrogate modules are not physics-source gaps by themselves, but need training-data provenance, split discipline, UQ, and model-card style credibility records. | Use local validation datasets first; add modern ML-UQ/model-risk references only when these modules are promoted into requirements. | Do not use surrogate outputs for scientific closure until dataset hashes, target provenance, uncertainty, and out-of-domain rejection are implemented. |

Immediate priority from this module pass:

1. Review newly local Del Zanna/CHIANTI and Puetterich records, then still
   acquire Post/ADAS/Dere or exact data exports before upgrading line radiation
   beyond `empirical_cooling_estimate`.
2. Review newly local Lotz and OCR/recover Seaton, then still acquire
   Burgess/NIST authoritative data before making CR or impurity-charge-state
   claims stronger than method scaffolding.
3. Use newly local Schmidt 2014 for kinetic model-form review after target
   extraction; still acquire Nanbu/Perez and Schmidt/Tang/Welch PRL 2012 before
   promoting PIC or hybrid beam diagnostics.
4. Acquire Nevins/Swain/Sikora-Weller p-B11 reactivity data before treating
   p-B11 outputs as more than a table estimate.
5. Mine local Kamm/Timmes and backend verification material before adding new
   Sedov/Athena acquisition work.

## WALRUS, The Well, And MHD Dataset Search - 2026-05-09

This fourth web/local pass covered ML-surrogate and public MHD dataset sources
that are adjacent to, but not equivalent to, DPF validation evidence.

Local audit details are recorded in
`docs/WALRUS_MHD_TRAINING_DATA_REVIEW_2026_05_09.md`.

| Source or dataset | Acquisition / metadata link | Current local status | Usable scope | Guardrail |
| --- | --- | --- | --- | --- |
| McCabe et al., "Walrus: A Cross-Domain Foundation Model for Continuum Dynamics" | arXiv `2511.15684`; GitHub <https://github.com/PolymathicAI/walrus>; Hugging Face model card | local `walrus/` checkout and `models/walrus-pretrained/walrus.pt` exist, both ignored; exact model hash/version record not promoted | ML architecture, model loading, fine-tuning workflow, benchmark context | Not DPF physics evidence; record model hash, license, checkpoint provenance, and inference tests before relying on it. |
| Ohana et al., "The Well: a Large-Scale Collection of Diverse Physics Simulations for Machine Learning" | NeurIPS 2024 DOI `10.52202/079017-1430`; arXiv `2412.00568`; docs <https://polymathic-ai.org/the_well/> | local `docs/WELL_FORMAT_SPECIFICATION.md` and Well exporter/validator exist; exact The Well paper is not yet KR-promoted | data-format and ML benchmark authority | Supports Well-format engineering only, not DPF validation. |
| The Well `MHD_64` / `MHD_256` | <https://polymathic-ai.org/the_well/datasets/MHD_64/> and `MHD_256` page | exact public dataset files not found locally in this pass | public MHD pretraining/benchmark data: density, velocity, magnetic field, periodic Cartesian isothermal MHD turbulence | Not DPF: no circuit, no electrodes, no cylindrical sheath, no radiation/neutrons, no PF-1000 scope. |
| Burkhart et al. 2020, "The Catalogue for Astrophysical Turbulence Simulations (CATS)" | DOI `10.3847/1538-4357/abc484`; OSTI metadata <https://www.osti.gov/pages/biblio/1762230> | exact paper not promoted to KR | source lineage for public astrophysical MHD turbulence data | Astrophysical turbulence benchmark only; not DPF validation. |
| NASA CFD V&V tutorial and ASME/FDA credibility-framework leads | NASA <https://www.grc.nasa.gov/WWW/wind/valid/tutorial/overview.html>; ASME V&V 40 / FDA credibility pages | not promoted to KR | context-of-use, credibility, and V&V method framing for surrogate/data use | Does not validate any dataset; use only for requirements and credibility process. |

Local WALRUS data verdict:

- Tracked `docs/walrus_training_*.json` files are Lee-model current/yield
  waveform sweeps, not volumetric MHD and not experimental validation.
- Ignored `training_data/dpf_v2/*.h5` and `training_data/dpf_batch_50/*.h5`
  are not defensible training data as-is: the audit found missing manifests,
  missing energy-conservation fields, non-finite circuit scalars, suspicious
  float32-limit field values, geometry metadata mismatches, and sampled all-zero
  magnetic fields.
- `docs/walrus_training_pf1000.h5` is a Lee-model current-waveform HDF5 file
  and fails the current validator because `boundary_conditions` is missing.

Required action before using any local WALRUS/DPF data beyond exploratory
software work:

1. Add strict dataset validation for scalars, saturated fields, geometry
   consistency, conservation diagnostics, and all-zero critical fields.
2. Add dataset manifests with hashes, generator commit, command, backend,
   config, dependency versions, hardware, seed, split IDs, and validator status.
3. Regenerate clean DPF HDF5 data from a vetted backend after numerical
   verification and run-manifest behavior are stable.
4. Keep public The Well MHD pretraining/benchmark data separate from DPF
   validation tiers.

## Highest Priority Blocking Acquisition

### 1. Klir et al. (2011)

- Title: "Fusion neutron detector for time-of-flight measurements in z-pinch
  and plasma focus experiments"
- DOI: `10.1063/1.3559548`
- Acquisition link: <https://doi.org/10.1063/1.3559548>
- Useful public metadata page: <https://pubmed.ncbi.nlm.nih.gov/21456735/>
- Local status: `not_found_as_exact_local_pdf`
- Priority: P1
- Needed for:
  - neutron detector timing and sensitivity calibration
  - neutron yield closure
  - neutron detector-response closure
  - propagated uncertainty for Tier 5
- Blocks:
  - `tier_5_neutron_validation`
  - `high_fidelity_readiness`
  - uncertainty coverage for neutron validation
- Required extraction after acquisition:
  - detector temporal response
  - neutron sensitivity or calibration procedure
  - absolute-yield calibration context
  - timing/sensitivity uncertainty terms

## Priority 2 Blocking Acquisitions

### 2. Sadowski/Scholz/PF-1000 team (2004)

- Title: "Measurements of fast ions and neutrons emitted from PF-1000 plasma
  focus device"
- DOI: `10.1016/j.vacuum.2004.07.040`
- Acquisition link: <https://doi.org/10.1016/j.vacuum.2004.07.040>
- Publisher landing page:
  <https://www.sciencedirect.com/science/article/abs/pii/S0042207X04003355>
- Local status: `not_found_as_exact_local_pdf`
- Priority: P2
- Needed for:
  - PF-1000 neutron spectra
  - neutron angular anisotropy
  - fast-ion diagnostic context
- Blocks:
  - `tier_5_neutron_validation`
- Required extraction after acquisition:
  - neutron spectrum data or digitizable figures
  - angular detector geometry and anisotropy values
  - fast-ion diagnostic placement and uncertainty context
  - source-scope details for PF-1000 shot/configuration matching

### 3. Catenacci et al. (2020)

- Title: "Tomographic Reconstruction of the Neutron Time-Energy Spectrum from a
  Dense Plasma Focus"
- DOI: `10.1109/TPS.2020.3012104`
- Acquisition link: <https://doi.org/10.1109/TPS.2020.3012104>
- Public metadata page:
  <https://www.pnnl.gov/publications/tomographic-reconstruction-neutron-time-energy-spectrum-dense-plasma-focus>
- Local status: `not_found_as_exact_local_pdf`
- Priority: P2
- Needed for:
  - DPF neutron time-energy spectrum reconstruction method
  - scatter-background subtraction method
  - validation design for neutron time/spectrum outputs
- Blocks:
  - `tier_5_neutron_validation`
- Required extraction after acquisition:
  - reconstruction method constraints
  - detector geometry and time-of-flight setup
  - spectrum/time-bin uncertainty or resolution
  - any digitizable time-energy examples suitable for method validation

### 4. Springham et al. (2021)

- Title: "Plasma focus neutron energy and anisotropy measurements using
  zirconium-beryllium pair activation detectors"
- DOI: `10.1016/j.nima.2020.164830`
- Acquisition link: <https://doi.org/10.1016/j.nima.2020.164830>
- Publisher landing page:
  <https://www.sciencedirect.com/science/article/abs/pii/S0168900220312274>
- Local status: `not_found_as_exact_local_pdf`
- Priority: P2
- Needed for:
  - activation-detector method for neutron energy and anisotropy
  - angular neutron fluence and energy anisotropy methodology
- Blocks:
  - `tier_5_neutron_validation`
- Required extraction after acquisition:
  - activation detector pair geometry
  - energy and fluence anisotropy definitions
  - MCNP or calibration relationship used for detector response
  - uncertainty terms for activation-derived neutron measurements

### 5. Jednorog et al. (2017)

- Title: "A new concept of fusion neutron monitoring for PF-1000 device"
- DOI: `10.1515/nuka-2017-0003`
- Acquisition link: <https://doi.org/10.1515/nuka-2017-0003>
- Publisher landing page: <https://sciendo.com/article/10.1515/nuka-2017-0003>
- Public PDF link:
  <https://www.nukleonika.pl/www/back/full/vol62_2017/v62n1p17f.pdf>
- Local status: `not_found_as_exact_local_pdf`
- Priority: P2
- Needed for:
  - PF-1000 neutron monitoring and activation response
  - silver activation counter and yttrium-monitor comparison context
  - MCNP activation coefficient context
- Blocks:
  - `tier_5_neutron_validation`
- Required extraction after acquisition:
  - activation monitor geometry and isotope/channel details
  - silver activation counter comparison data
  - MCNP activation coefficient assumptions
  - response calibration and uncertainty terms

## Secondary Missing Acquisition Candidate

This source is not one of the current blocking P1/P2 acquisition items because
the source queue currently marks `spatial_density` as complete in the selected
PF-1000 scope from other local sources. It is still missing as an exact local
PDF and may be worth acquiring for redundancy or stronger spatial-density
closure.

### 6. Cikhardtova et al. (2015)

- Title: "Temporal distribution of linear densities of the plasma column in a
  plasma focus discharge"
- DOI: `10.1515/nuka-2015-0065`
- Acquisition link: <https://doi.org/10.1515/nuka-2015-0065>
- Publisher landing page:
  <https://sciendo.com/es/article/10.1515/nuka-2015-0065>
- Local status: `not_found_as_exact_local_pdf`
- Priority: secondary
- Useful for:
  - PF-1000 interferometric linear-density time history
  - plasma-column velocity and spatial-density timing context
- Potential extraction after acquisition:
  - interferometry frame timing
  - linear-density profiles or digitizable figures
  - radial and axial velocity estimates
  - shot/configuration scope and uncertainty context

## Additional Verified Acquisition Candidates From Web Review

The sources below were identified from Google Scholar title-query review,
publisher/institutional metadata pages, open-access journal pages, IAEA/INIS,
OSTI, CTU FEE, IPPLM, Nukleonika, ICDMP, and other scholarly indexes. They are
not yet promoted evidence. Their local status is only a fast title/DOI scan
against the current repo, not the full exact-PDF audit already performed for the
six primary blockers above.

### 7. Rezac et al. (2026)

- Title: "Silver activation counter: Detector with large dynamic range for
  measurement of fast-neutron bursts"
- DOI: `10.1016/j.nima.2025.171054`
- Acquisition link: <https://doi.org/10.1016/j.nima.2025.171054>
- Publisher landing page:
  <https://www.sciencedirect.com/science/article/abs/pii/S0168900225008563>
- Verified metadata pages:
  - CTU FEE publication pages
  - J-GLOBAL article metadata
- Local status: `not_found_in_KR_title_doi_scan_2026_05_09`
- Priority: P2-method / P2-neutron-diagnostic
- Useful for:
  - modern silver-activation counter design
  - fast-neutron burst dynamic range
  - OpenMC-backed response corrections
  - neutron-yield uncertainty budget design
- Required extraction after acquisition:
  - detector geometry, moderator/activation material assumptions, calibration
    procedure, OpenMC correction coefficients, dynamic range, pulse-duration
    limits, and yield uncertainty terms.

### 8. Rezac, Klir, Kubes, Kravarik (2012)

- Title: "Improvement of time-of-flight methods for reconstruction of neutron
  energy spectra from D(d,n)3He fusion reactions"
- DOI: `10.1088/0741-3335/54/10/105011`
- Acquisition link: <https://doi.org/10.1088/0741-3335/54/10/105011>
- Verified metadata pages:
  - CTU FEE publication pages
  - OSTI nToF review related-index page
- Local status: `not_found_in_KR_title_doi_scan_2026_05_09`
- Priority: P2-method / P2-neutron-spectrum
- Useful for:
  - PF-1000 neutron time-of-flight spectral reconstruction
  - opposite-direction detector layout
  - scatter reduction and time/energy resolution methodology
- Required extraction after acquisition:
  - detector-distance layout, reconstruction algorithm, resolution, scatter
    handling, source-scope details, and uncertainty limits.

### 9. Klir/Kubes/PF-1000 Team (2012)

- Title: "Search for thermonuclear neutrons in a mega-ampere plasma focus"
- DOI: `10.1088/0741-3335/54/1/015001`
- Acquisition link: <https://doi.org/10.1088/0741-3335/54/1/015001>
- Verified metadata pages:
  - CTU FEE publication pages
  - IAEA TECDOC reference lists
- Local status: `not_found_in_KR_title_doi_scan_2026_05_09`
- Priority: P2-neutron-mechanism
- Useful for:
  - PF-1000 neutron mechanism separation
  - thermonuclear versus nonthermal neutron interpretation
  - time-of-flight signal context at mega-ampere current
- Required extraction after acquisition:
  - shot/configuration scope, neutron pulse timing, TOF detector geometry,
    energy spectra, inferred ion-temperature or mechanism bounds, and stated
    uncertainties.

### 10. Krauz et al. (2012)

- Title: "Experimental study of the structure of the plasma-current sheath on
  the PF-1000 facility"
- DOI: `10.1088/0741-3335/54/2/025010`
- Acquisition link: <https://doi.org/10.1088/0741-3335/54/2/025010>
- Verified metadata pages:
  - local Malir 2024 KR reference list cites the paper
  - IAEA TECDOC and CTU/IPPLM-linked reference pages
- Local status: `referenced_in_KR_but_exact_source_not_promoted`
- Priority: P2-current-coupling / P2-spatial
- Useful for:
  - current-sheath structure on PF-1000
  - magnetic-probe comparison with interferometry
  - current coupling and sheath transport into the axial region
- Required extraction after acquisition:
  - probe geometry, radial current/density profiles, sheath timing, current
    transport estimates, high-yield shot conditions, and uncertainty/context
    limits.

### 11. Kubes/Klir/PF-1000 Team (2013)

- Title: "Scenario of pinch evolution in a plasma focus discharge"
- DOI: `10.1088/0741-3335/55/3/035011`
- Acquisition link: <https://doi.org/10.1088/0741-3335/55/3/035011>
- Verified metadata pages:
  - CTU FEE publication pages
  - IAEA/INIS metadata page
- Local status: `not_found_in_KR_title_doi_scan_2026_05_09`
- Priority: P2-phase / P2-field-coupling
- Useful for:
  - PF-1000 pinch-evolution sequence
  - magnetic-probe, interferometry, and neutron-diagnostic correlation
  - toroidal/helical/plasmoidal current structure context
- Required extraction after acquisition:
  - phase timing, magnetic probe observables, interferometry frames, neutron
    timing, inferred toroidal-current estimates, and configuration scope.

### 12. Kortanek/Kubes/PF-1000 Team (2014)

- Title: "Current flow and energy balance during the evolution of instabilities
  in the plasma focus"
- DOI: `10.1088/0031-8949/2014/T161/014044`
- Acquisition link: <https://doi.org/10.1088/0031-8949/2014/T161/014044>
- Verified metadata pages:
  - CTU FEE publication pages
  - IPPLM 2014 publication listing
- Local status: `not_found_in_KR_title_doi_scan_2026_05_09`
- Priority: P2-energy-coupling / P2-instability
- Useful for:
  - PF-1000 energy release during constriction expansion
  - current, voltage, `dI/dt`, inductance, and neutron-pulse correlation
  - deuteron-acceleration energy budget review
- Required extraction after acquisition:
  - electrical waveform definitions, inductance and `dL/dt` calculations,
    neutron pulse timing, energy-balance equations, and uncertainty terms.

### 13. Scholz et al. (2012)

- Title: "Progress in MJ plasma focus research at IPPLM"
- DOI: not found in the web review
- Acquisition link:
  <https://www.nukleonika.pl/www/back/full/vol57_2012/v57n2p183f.pdf>
- Verified metadata pages:
  - Nukleonika issue page
  - CTU FEE publication pages
- Local status: `not_found_in_KR_title_scan_2026_05_09`
- Priority: P3-context / P3-redundancy
- Useful for:
  - PF-1000 magnetic-probe, 16-frame interferometry, silver activation, and
    PMT neutron-probe overview
  - neutron-yield scaling context
- Required extraction after acquisition:
  - device/configuration summary, diagnostic table, current measurements,
    density-evolution statements, neutron-yield scaling context, and figure or
    table candidates.

### 14. Auluck et al. (2021)

- Title: "Update on the Scientific Status of the Plasma Focus"
- DOI: `10.3390/plasma4030033`
- Acquisition link: <https://doi.org/10.3390/plasma4030033>
- Publisher page: <https://www.mdpi.com/2571-6182/4/3/33>
- Verified metadata pages:
  - MDPI open-access article page
  - OUCI indexed metadata
  - ResearchGate discovery page
- Local status: `not_found_as_reviewed_KR_record`
- Priority: P3-review / P3-source-map
- Useful for:
  - DPF literature map
  - diagnostic and mechanism terminology review
  - identifying additional PF-1000 source papers before source acquisition
- Required extraction after acquisition:
  - bibliography map only; do not extract validation targets from the review
    unless the underlying primary source is also acquired and reviewed.

### 15. Bernard et al. (1998)

- Title: "Scientific status of plasma focus research"
- DOI: not found in the web review
- Acquisition link:
  <https://icdmp.pl/basicpapers/scientificstatus>
- Verified metadata pages:
  - ICDMP basic-paper page
  - cited by Auluck et al. 2021 as the predecessor review
- Local status: `not_found_as_reviewed_KR_record`
- Priority: P3-historical-review / P3-source-map
- Useful for:
  - historical DPF review and terminology baseline
  - source discovery for older neutron, ion, and diagnostic work
- Required extraction after acquisition:
  - bibliography map and historical context only; do not use as direct
    validation evidence without the primary source.

## Already Local - Do Not Reacquire Unless Better Copies Are Needed

These papers are already represented by parity-verified local
`KnowledgeReference/` records. They still may need local digitization,
target extraction, or independent review, but they are not current acquisition
targets.

| Source | DOI | Local KR record | Remaining local work |
| --- | --- | --- | --- |
| Akel, Kubes, Paduch, Lee (2021), "Comparison of measured and computed neutron yield from PF1000 plasma focus device operated with deuterium gas" | `10.1016/j.radphyschem.2021.109633` | `KnowledgeReference/radiation-physics-and-chemistry-188-2021-109633.md` | Akel Fig. 1 remains `blocked_by_review`; Figs. 2-6 still need verified digitization/review. |
| Gribkov et al. (2007), PF-1000 Part I | `10.1088/0022-3727/40/7/021` | `KnowledgeReference/gribkov-2007-pf1000-jphysd-part2.md` | Extract any remaining same-scope waveform, phase, and spatial targets not yet coded. |
| Gribkov et al. (2007), PF-1000 Part II | `10.1088/0022-3727/40/12/008` | `KnowledgeReference/scholz-2007-pf1000-part2-jphysd.md` | Extract any remaining fast-ion, neutron, and uncertainty observables not yet coded. |
| Schmidt et al. (2022), MJOLNIR high/low performance discharges | `10.1063/5.0089121` | `KnowledgeReference/goyon-2022-mjolnir-high-low.md` | Use only coded targets or verified digitized observables for MJOLNIR comparison. |
| Malir et al. (2024), PF-1000 density/interferometry vs MHD | `10.1063/5.0193268` | `KnowledgeReference/malir-2024-interferometry-dpf.md` | Use coded density-profile targets and uncertainty records; digitize only for extra arrays. |
| Goyon et al. (2025), MA-class DPF neutron-generation dynamics | `10.1063/5.0253547` | `KnowledgeReference/neutron-generation-dynamics-inside-a-ma-class-dense-plasma-focus-z-pinch.md` | Use coded neutron timing, temperature, and detector-response targets; digitize only for additional arrays. |

## Already Local Textbook/Formulary Coverage

These are already represented locally and should not be reacquired unless a
newer official edition or cleaner copy is specifically needed.

| Source | Local KR record | Current use |
| --- | --- | --- |
| NRL Plasma Formulary 2018 | `KnowledgeReference/plasma-formulary.md` | Constants, unit checks, plasma parameters, transport/radiation formula cross-checks. |
| F. F. Chen, *Introduction to Plasma Physics and Controlled Fusion*, 3rd ed. | `KnowledgeReference/plasma-physics-and-controlled.md` | General plasma fundamentals, fluid equations, waves, MHD basics, diffusion/resistivity. |
| *Fundamentals of Plasma Physics* local chapter set, 1986 | `KnowledgeReference/preface-1986-fundamentals-of-plasma-physics.md` and related `chapter-*` records | General plasma fundamentals, kinetic/fluid/MHD and transport background. |

## Intake Checklist After Acquisition

For each acquired paper:

1. Preserve the exact source document and record its SHA-256.
2. Create or update the reviewed `KnowledgeReference/*.md` record.
3. Record DOI, title, authors, publication venue, local source path, and hash.
4. Extract source-backed target records in `src/dpf/validation/kr_targets.py`
   only after local review.
5. For figures or tables, create a digitization packet with:
   source hash, figure/table ID, page, axes or table schema, units, arrays,
   overlay residuals for figures, and independent accepted review metadata.
6. Rerun:
   - `scientific_closure_source_acquisition_queue()`
   - `kr_validation_same_scope_target_report()`
   - `scientific_accuracy_gap_report()`
   - predictive/high-fidelity readiness checks

Until these steps are complete, the affected validation tier remains blocked.
