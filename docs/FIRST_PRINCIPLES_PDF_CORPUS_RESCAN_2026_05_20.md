# First-Principles PDF Corpus Rescan - 2026-05-20

Purpose: identify additional local PDF material that may help close
first-principles DPF simulator blockers while the implementation team works on
Sprint 3R/Sprint 4 tasks.

This document is a discovery and extraction-priority artifact. Raw PDFs are not
scientific authority. A PDF becomes usable by runtime code only after it is
promoted into `KnowledgeReference/`, reviewed for source fidelity, target
extracted with units/scope/status, and covered by fail-closed tests.

## Method

Observed local inventory:

- `rg --files --hidden --no-ignore -g '*.pdf' -g '*.PDF'` across
  `/Users/anthonyzamora/dpf-unified` and `/Users/anthonyzamora/Downloads` found
  1724 PDF paths.
- Focused roots were rescanned:
  - `archive_reference_OLD/references/papers`
  - `downloaded_books_papers`
  - `/Users/anthonyzamora/Downloads`
- The focused rescan saw 1565 PDF paths and 847 unique SHA-256 payloads.
- Text extraction used `pdftotext` over the first 40 pages per unique PDF, with
  cached text under `tmp/pdfs/corpus_rescan_2026_05_20/text/`.
- Candidate scoring used blocker-domain terms for PF-1000 geometry, startup
  and breakdown, transport closure, radiation/EOS, ablation/impurity, neutron
  mechanism, same-scope comparator evidence, and whole-shot numerics.
- The generated candidate index is
  `tmp/pdfs/corpus_rescan_2026_05_20/corpus_rescan_candidates.json`.

Domain-hit counts from the focused pass:

| Domain bucket | Hit count |
| --- | ---: |
| Whole-shot numerics | 630 |
| PF-1000 geometry | 604 |
| Transport closure | 536 |
| Atomic/radiation/EOS | 414 |
| Same-scope comparator terms | 348 |
| Ablation/impurity | 254 |
| Neutron kinetic | 178 |
| Startup/breakdown | 168 |

Limitations:

- This pass is high-recall discovery, not final source extraction.
- Page 41+ content and image-only tables may still hide useful data.
- Broad keyword hits are leads only. Spot reading decides whether a record is
  worth promotion or target extraction.

## Verdict

Yes, there is additional useful material on disk.

The biggest immediate win is not inventing new physics. It is converting two
classes of local material into source-truth records:

1. Raw PDFs that are high-value enough to promote into `KnowledgeReference/`.
2. Already-promoted KR records that are still shallow and need target
   extraction into typed, line-cited simulator inputs.

This rescan did not find same-scope PF-1000/Akel 16 kV `V(t)`, `T_e/T_i`, X-ray,
neutron spectrum, or anisotropy data. Those channels remain blocked unless the
validation scope changes or new same-scope measurements are obtained.

## P0 Raw PDF Promotion Candidates

2026-05-20 status: P0 promotion is complete. The two raw candidates below were
staged into
`downloaded_books_papers/Research Papers/2026-05-20-corpus-rescan/` and promoted
as fail-closed `KnowledgeReference/` text-parity records. The promotion ledger is
`docs/CORPUS_RESCAN_KR_PROMOTION_2026_05_20.md`.

| Priority | PDF | Why it matters | Required next action |
| --- | --- | --- | --- |
| P0 | `/Users/anthonyzamora/Downloads/plasma-04-00033.pdf` | Auluck et al. 2021, "Update on the Scientific Status of the Plasma Focus," DOI `10.3390/plasma4030033`. The spot read shows dense coverage of DPF phenomenology, beam-target versus thermonuclear interpretation, PF-1000 material, startup/sheath formation, diagnostics, ion acceleration, neutron spectra, and modeling limits. | Promoted to `KnowledgeReference/update-on-the-scientific-status-of-the-plasma-focus-1385adeb.md` with 9 page-range chunks. Next: source-fidelity review and target extraction only for blocker-relevant claims. |
| P0 | `/Users/anthonyzamora/Downloads/bernard1977.pdf` | Bernard et al. 1977, "The Dense Plasma Focus - A High Intensity Neutron Source." The spot read shows neutron time/space/spectrum diagnostics, beam-target versus thermonuclear discrimination, density/temperature diagnostics, anisotropy, and detector method material. | Promoted to `KnowledgeReference/the-dense-plasma-focus-a-high-intensity-neutron-source-f0a3910d.md`. Next: target-extract neutron mechanism and diagnostic requirements. |

Expected extraction products:

- DPF phase taxonomy: breakdown, liftoff, axial rundown, radial collapse, pinch,
  disruption, neutron-emitting phase.
- Neutron mechanism taxonomy: thermonuclear, beam-target, trapped fast-ion, and
  detector-response separation.
- Diagnostic requirements: time-of-flight, activation, anisotropy, spectrum,
  interferometry, Thomson scattering, density/temperature context.
- Modeling constraints: where MHD is acceptable, where kinetic/PIC treatment is
  required, and where historical models are explicitly non-ab-initio.

## P1 Raw PDF Promotion Candidates

| Priority | PDF | Why it matters | Scope boundary |
| --- | --- | --- | --- |
| P1 | `/Users/anthonyzamora/Downloads/102708_1_online.pdf` | Lerner et al. 2017, "Confined ion energy >200 keV and increased fusion yield in a DPF with monolithic tungsten electrodes and pre-ionization," DOI `10.1063/1.4989859`. Spot read shows impurity reduction, preionization, electrode material, mean ion energy, and yield reproducibility claims. | FF-1/Focus Fusion scope, not PF-1000/Akel validation. Use only for impurity/preionization/electrode-material modeling candidates. |
| P1 | `archive_reference_OLD/references/papers/core-dpf/offermann-2021-mjolnir-dpf.pdf` | MJOLNIR DPF radiography source already appears in project reference ledgers. It is relevant to MJOLNIR model-coverage blockers, restrike/current-diversion investigation, and radiographic experimental context. | MJOLNIR scope, not PF-1000/Akel. Check existing KR ingest alias before duplicating. |
| P1 | `archive_reference_OLD/references/papers/core-dpf/schmidt-2017-dpf-design-parameter.pdf` | HPO/DPF design-parameter source appears relevant to high-pressure operation, liftoff, drive-parameter scaling, and source-configuration review. | Design/scaling context only until target-extracted. |

## P2 Method-Only Candidates

| PDF | Use | Reason not higher priority |
| --- | --- | --- |
| `/Users/anthonyzamora/Downloads/Plasma-Physics-and-Engineering.pdf` | General plasma engineering background, transport sanity checks, notation cross-checks. | Not DPF-specific authority unless a named equation/data gap maps directly to it. |
| `/Users/anthonyzamora/Downloads/033901_1_5.0182654.pdf` | X-pinch SPH numerics with EOS, transport, and radiation terms. | X-pinch, not DPF. Useful only as numerical-method background. |
| `archive_reference_OLD/references/papers/core-dpf/2023_Focus_Fusion_Overview_of_Progress_Towards_p-B11_Fusion_with_the_Dense_Plasma_Fo.pdf` | pB11/Focus Fusion context, impurity/electrode and repetition-rate context. | Not DD PF-1000/Akel first demonstrator scope. |
| `archive_reference_OLD/references/papers/core-dpf/2025_Double_3 MJ_dense_plasma_focus_for_thermonuclear_drive_inertial_confinement_fusi.pdf` | High-energy DPF concept context. | Concept/design scope; not immediate whole-shot DD closure. |

## Already-KR Target-Extraction Queue

These are already in `KnowledgeReference/` and should not be duplicated as raw
ingestion. They need target-extraction packets with exact line ranges, units,
scope labels, and fail-closed status.

| Priority | KR record | Extract next |
| --- | --- | --- |
| P0 | `KnowledgeReference/the-dense-plasma-focus-a-versatile-dense-pinch-for-diverse-applications.md` | Startup phase sequence; insulator breakdown and liftoff; high-pressure filamentation; preionization; restrike/knife-edge qualitative requirements; final-pinch MHD-to-PIC boundary; neutron mechanism evidence schema. |
| P0 | `KnowledgeReference/malir-2024-interferometry-dpf.md` | PF-1000 density profiles, shock reflection context, initial/boundary values, and explicit 1D limitations requiring 2D/3D treatment. This is cross-scope PF-1000 evidence, not Akel 16 kV validation. |
| P1 | `KnowledgeReference/effect-of-current-sheath-initiation-on-the-radial-collapse-and-energetic-particle-accelera-b2e95b88.md` | DPF-specific Paschen-law caveats, insulator/liftoff scaling, pressure regimes, partial-ionization and filamentation bounds. |
| P1 | `KnowledgeReference/measurement-of-electric-flux-emission-a-new-diagnostic-for-the-dense-plasma-focus-a-b-blagoev12aa-v-4.md` | Formation symmetry diagnostic, D-dot probe requirements, training-shot effects, PIC-MCC/radiative multiscale constraints. |
| P1 | `KnowledgeReference/beresnyak_2018_dpf_hawk_simulations.md` | 3D MHD+circuit coupling, Hall/Spitzer method context, stochastic particle acceleration schema. Keep as HAWK/local-injection method context, not neutral-fill startup authority. |
| P1 | `KnowledgeReference/beresnyak_2022_pulsed_power_ideal_mhd.md` | Ideal-MHD pulsed-power simulation assumptions and limits; useful for negative-control tests around overclaiming ideal MHD. |

## Negative Findings That Should Stay Blocked

This pass did not find:

- Same-scope PF-1000/Akel 16 kV terminal voltage waveform `V(t)`.
- Same-scope PF-1000/Akel 16 kV electron and ion temperature histories.
- Same-scope PF-1000/Akel 16 kV X-ray waveform.
- Same-scope PF-1000/Akel 16 kV neutron spectrum and anisotropy.
- A D2 Townsend-alpha table or D2 Paschen A/B constants ready for a
  first-principles startup BVP.
- A dedicated current-restrike equation suitable for runtime closure.
- A local Brysk 1973 Doppler-broadening source in the focused rescan.

Implication:

- More source ingestion can improve the simulator foundation.
- It cannot complete the same-scope Akel 16 kV certificate unless those missing
  channels are obtained, digitized from a reviewed same-scope source, or the
  validation target is explicitly re-scoped.

## Index Hygiene Finding

The rescan exposed a source-truth index reconciliation issue:

- `AD1100306_SimulationsofabDensebbPlasmabFocusonaHigh-Impedance.pdf` was
  initially scored as "not in KR," but it is represented by
  `KnowledgeReference/beresnyak_2018_dpf_hawk_simulations.md`.
- `AD1194691_bSimulatingbaPulsed-Power-DrivenPlasmawithIdealbMHDb.pdf` was
  initially scored as "not in KR," but it is represented by
  `KnowledgeReference/beresnyak_2022_pulsed_power_ideal_mhd.md`.

Required improvement:

- Add a source-alias or SHA-based reconciliation check to
  `docs/FIRST_PRINCIPLES_SOURCE_TRUTH_INDEX.*` generation so raw-path aliases do
  not produce false "not in KR" findings.

## Recommended Parallel Work While The Team Works

1. Target-extract the newly promoted P0 records and the P0/P1 already-KR records
   listed above into a typed source
   target packet, without promoting validation state.
2. Run a deeper full-text and table/figure pass over the top 50 rescanned PDFs,
   not the full 847-PDF set, to keep the review auditable.
3. Repair or reacquire corrupted candidate PDFs only when they map to an active
   blocker. The one worth tracking is the corrupted electrode-heating PDF under
   `archive_reference_OLD/references/papers/core-dpf/`.
4. Add the source-index reconciliation check so duplicate aliases stop
   producing fake gaps.
5. Keep same-scope PF-1000/Akel 16 kV comparator channels blocked until a
   reviewed source is found or the validation scope changes.

## Handoff Instructions

The next team should return extraction packets, not prose summaries. Each packet
must include:

1. blocker ID;
2. old status;
3. proposed new status;
4. raw PDF path or KR path;
5. exact page/table/figure or KR line range;
6. extracted value/equation/requirement;
7. units and symbol map;
8. device and shot/configuration scope;
9. uncertainty or `uncertainty_not_supplied`;
10. runtime claim impact;
11. test that prevents raw or shallow evidence from being treated as accepted.

Any proposal that lacks those fields stays `source_available_not_target_extracted`
or `raw_pdf_candidate_not_authority`.
