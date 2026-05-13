# Scientific Closure Source Queue

Updated: 2026-05-12

This queue is not source-of-truth evidence. The source of truth remains local
files under `KnowledgeReference/`. These candidate links become usable only
after the user acquires the correct document, the document is added locally
under `KnowledgeReference/`, Codex reviews it under the KR-only rule, and any
digitized data passes one-for-one provenance verification.

For a user-facing acquisition checklist with direct DOI/publisher links, see
`docs/SOURCE_ACQUISITION_NEEDED.md`.

Current widest closure scope from the local KR target report:

- `validation_scope`: `pf1000_full_energy_2007_gribkov_scholz`
- `device`: `PF-1000`
- Readiness state: blocked. The local corpus has no complete same-scope packet
  across circuit waveform, phase timing, spatial validation, neutron timing,
  neutron spectrum, neutron anisotropy, detector response, field coupling,
  physics fidelity, and propagated uncertainty.

## Acquisition Process

1. AI researches candidate source documents and provides links.
2. User acquires the correct source document.
3. Document is added locally under `KnowledgeReference/`.
4. Codex reviews the local document under the KR-only rule.
5. If needed, digitization is performed with one-for-one verification.
6. Typed KR targets are extracted and same-scope closure is rerun.

Queue entries distinguish `local_sources_available` from
`candidate_sources_for_acquisition`. Akel 2021, Gribkov 2007 Parts I/II,
Schmidt 2022, Malir 2024, and Goyon 2025 are already represented by
parity-verified local `KnowledgeReference` markdown records. The 2026-05-11
supplemental intake also promoted Cikhardtova 2015, Szydlowski/Sadowski/Scholz
2004, Catenacci 2020, Springham 2021, and Klir 2011 into local
`source_fidelity_reviewed_target_extraction_needed` KR records, and the first
typed target/crop-candidate pass is now underway. Jednorog 2017 remains the only
still-missing exact PDF among that earlier six-paper queue.

The 2026-05-12 supplemental user intake staged 35 unique PDFs from 39 supplied
paths. It promoted 28 new DPF/plasma/numerics/math-method records into
`KnowledgeReference/`, repaired a false Trunk 1975/Kortanek 2014 generic-title
match, and left seven non-physics/AI-only support PDFs
staged but outside physics authority. The promoted/skipped set now has
source-fidelity annotations across 28 KR records, including figure-caption,
table-caption, formula-like, numeric-target-context, uncertainty-context, and
image-block indexes. This creates new source-review material only; no May 12
record closes a validation tier until typed targets or independently reviewed
digitization packets are created.

The first May 12 triage backlog is
`docs/USER_PDF_MAY12_TARGET_TRIAGE_2026_05_12.md` / `.json`. It currently
marks 5 target-extraction candidates: four P1 source-review candidates
(`10.1088@1742-6596@370@1@012059.pdf`, `kasperczuk2002.pdf`,
`kubes2020.pdf`, and `trunk1975.pdf`) and one P2 cleanup-first candidate
(`lindemuth1982.pdf`). Alexiou 2002 is a spectroscopy/method reference,
Sadowski 2008 is review/source-map context, and `symons1994.pdf` is stage-only
after first-page review showed it is an out-of-scope JSTOR social-science
review. The remaining promoted records are method, context, or materials
references until a specific source-backed validation need is mapped.

The source-level validation report is
`docs/USER_PDF_MAY12_SOURCE_VALIDATION_2026_05_12.md` / `.json`. It validates
28 promoted source records, 7 stage-only records, 5 target-extraction
candidates, 23 method/context records, and records 0 validation failures. This
is source-authority validation only; validation tiers remain blocked until typed
targets or independently reviewed digitization packets exist.

The validated-physics promotion plan is
`docs/VALIDATED_PHYSICS_PIPELINE_PLAN.md`. It defines the required pipeline from
source validation through source-line review, typed target extraction,
figure/table digitization, formula evidence, uncertainty propagation,
comparator binding, same-scope packet assembly, and certificate gates. This plan
does not accept any target, curve, table, formula, uncertainty value, or
validation threshold by itself.

## Machine-Readable Queue State

`scientific_closure_source_acquisition_queue()` now reports both blocker items
and a same-scope group-status matrix. The current PF-1000 full-energy closure
summary is:

- `blocker_count`: 10
- `priority_1_count`: 5
- `priority_2_count`: 5
- `local_digitization_or_target_extraction_count`: 10
- `user_acquisition_required_count`: 1
- `complete_group_count`: 2
- `partial_group_count`: 10
- `missing_group_count`: 0

The same-scope status matrix marks `phase_semantics` and `spatial_density` as
`complete_in_current_scope`; the other active closure groups remain
`partial_in_current_scope`. This means the queue is now primarily blocked by
title cleanup, target extraction, figure/table digitization, uncertainty, and
detector-response work. User acquisition is still required for Jednorog 2017
and for adjacent sources not yet covered by the candidate matrix.

Each blocker item now includes a `source_action`:

- `local_digitization_or_target_extraction` when local KR sources already exist
  but observables, uncertainty, or typed targets still need extraction.
- `user_acquisition_then_knowledge_reference_ingestion` when at least one needed
  source still requires user acquisition before KR review.

Each blocker item also names the validation tiers it blocks so Tier 2, Tier 4,
Tier 5, Akel S1/S2, and high-fidelity readiness cannot be accidentally
advanced from an incomplete queue.

## Digitization Verification Gate

Digitized figure or table data can support validation only when a packet
passes `digitization_verification_evidence()`. The packet must include:

- local `KnowledgeReference/` source path and matching SHA-256 hash
- figure image path and matching SHA-256 hash for figure extractions
- page and figure/table identifier
- x/y axis calibration points, units, and residuals for figures
- extracted series arrays with units and enough points for the target
- overlay residual evidence for figures
- at least one independent review and `review_status="accepted"`
- no `KnowledgeReference` path traversal or malformed review-count metadata

Passing this gate proves provenance and digitization quality only. It does not
validate a simulation result by itself.

## Local Akel 2021 Figure Digitization Queue

Akel et al. 2021 is already represented in `KnowledgeReference` at
`KnowledgeReference/radiation-physics-and-chemistry-188-2021-109633.md`, and
text parity against the local PDF passed. The remaining local work is figure
digitization, now tracked by `scientific_closure_digitization_queue()`.

Priority 1 current-waveform tasks:

- Fig. 1, PDF page 3, source lines 294-295: measured and computed current,
  shot 12581, 1.2 Torr. Current extraction status:
  `extracted_not_digitized`; cropped panel stored at
  `KnowledgeReference/figures/akel-2021-fig1-current-waveform-shot-12581.png`
  with SHA-256
  `4c574525f1de413e54cd02bd06aa35d549db700270281310a3809edc54ab255e`.
  The crop preserves axes, legend, and current traces. It is still not
  validation evidence until measured/computed series arrays, overlay residuals,
  and independent review are supplied.
  Draft vector extraction route:
  the current `pdftocairo` page-3 SVG separates a measured-current candidate
  as filled black paths `1987-2280` and a computed-current candidate as black
  stroke paths `1942-1975`; filled black paths `2345-2411` are legend glyphs
  in the white legend box and must be excluded. This is extraction metadata
  only, not an accepted digitization packet.
  Draft arrays are now stored at
  `KnowledgeReference/digitization/akel-2021-fig1-current-waveform-shot-12581-draft-packet.json`
  with SHA-256
  `abe4a283ee154f84f6061da8ea508d3871faf3b14dddb2d1cfc8a7a0a5f8e0e7`.
  The packet includes 294 measured-current candidate points and 34
  computed-current candidate points, but remains `draft_unreviewed`;
  `digitization_verification_evidence()` currently fails it only on
  independent-review acceptance checks. Internal vector round-trip overlay
  residuals are measured from the archived page-3 SVG at `0.213455189 px`
  RMS over 328 candidate points.
- Fig. 2, PDF page 3, source lines 296-297: measured and computed current,
  shot 12584, 1.2 Torr.
- Fig. 3, PDF page 3, source lines 298-299: measured and computed current,
  shot 12592, 1.05 Torr.
- Fig. 4, PDF page 3, source lines 300-301: measured and computed current,
  shot 12604, 1.05 Torr.

Priority 2 neutron-yield cross-check tasks:

- Fig. 5, PDF page 5, source line 916: measured and calculated neutron yields
  at 1.2 Torr.
- Fig. 6, PDF page 5, source line 917: measured and calculated neutron yields
  at 1.05 Torr.

The yield figures are priority 2 because Akel Tables 1 and 2 are already
typed as `pf1000_16kv_shot_table_2021_akel`. The figures remain useful as a
plot-level parity cross-check, but they do not replace detector-response,
neutron timing, spectrum, or anisotropy closure.

Future digitization packets should be evaluated with
`scientific_closure_digitization_status()`. A task is accepted only if the
packet passes `digitization_verification_evidence()` and also matches the
queue's task ID, source path, source hash, local PDF hash, source line window,
figure ID, page, and required series names.

For the Akel Fig. 1 current-waveform packet,
`pf1000_16kv_current_waveform_digitization_candidate_evidence()` reports the
current state as `blocked_by_review`, not `digitization_packet_missing`. It
still returns `passed=False`; the helper is data-readiness status only and
does not compare any simulation waveform against the draft trace.

Before an independent review decision is recorded, run:

```bash
python3 scripts/verify_akel_digitization_source_integrity.py --pretty
```

This pre-review guardrail checks the local PDF/markdown/JSON text parity, source
hashes, figure crop hash, archived page-3 SVG hash, draft packet hash, Fig. 1
caption line window, required series counts, and non-review digitization
failures. A passing report can still say `accepted_for_validation=false`; that
is expected while the remaining failures are only `independent_review_missing`
and `review_status_not_accepted`.

## Priority 1 Queue

### circuit_waveform

- Physics need: circuit current and voltage
- Current KR status: partial only in same-scope targets
- Current KR source: `KnowledgeReference/scholz-2007-pf1000-part2-jphysd.md`
- Required data:
  - `digitized_current_trace_points`
  - `direct_pinch_current_measurement`
  - `per_point_current_uncertainty`
  - `per_point_timing_uncertainty`
- Source leads and local status:
  - Akel, Kubes, Paduch, Lee (2021), "Comparison of measured and computed neutron yield from PF1000 plasma focus device operated with deuterium gas", DOI: https://doi.org/10.1016/j.radphyschem.2021.109633
    - Why: measured PF-1000 current wave shapes at 16 kV and neutron yield.
  - Gribkov et al. (2007), "Plasma dynamics in PF-1000 device under full-scale energy storage: I. Pinch dynamics, shock-wave diffraction, and inertial electrode", DOI: https://doi.org/10.1088/0022-3727/40/7/021
    - Why: full-energy PF-1000 electrical and phase diagnostics.

### phase_timing

- Physics need: rundown, radial collapse, and pinch phase timing
- Current KR status: partial only in same-scope targets
- Current KR source: `KnowledgeReference/gribkov-2007-pf1000-jphysd-part2.md`
- Required data:
  - `breakdown_to_rundown_absolute_timing`
  - `digitized_current_and_derivative_traces`
  - `per-shot phase timing uncertainty`
  - `radial_transit_start_and_end_times`
- Source leads and local status:
  - Gribkov et al. (2007), "Plasma dynamics in PF-1000 device under full-scale energy storage: I. Pinch dynamics, shock-wave diffraction, and inertial electrode", DOI: https://doi.org/10.1088/0022-3727/40/7/021
    - Why: breakdown, rundown, radial collapse, pinch, x-ray, and neutron timing context.
  - Schmidt et al. (2022), "Measuring characteristic differences between high- and low-performing discharges on the MJOLNIR DPF", DOI: https://doi.org/10.1063/5.0089121
    - Why: current traces, optical gates, framing-camera velocities, and restrike context.

### neutron_yield

- Physics need: absolute or shot-resolved neutron yield
- Current KR status: partial only in same-scope targets
- Current KR source: `KnowledgeReference/scholz-2007-pf1000-part2-jphysd.md`
- Required data:
  - `fast_ion_distribution_uncertainty`
  - `neutron_field_transport_or_room_scatter_response_model`
  - `yield_calibration_uncertainty`
- Source leads and local status:
  - Akel, Kubes, Paduch, Lee (2021), "Comparison of measured and computed neutron yield from PF1000 plasma focus device operated with deuterium gas", DOI: https://doi.org/10.1016/j.radphyschem.2021.109633
    - Why: shot-resolved PF-1000 scalar neutron yields and fitted current parameters.
  - Klir et al. (2011), "Fusion neutron detector for time-of-flight measurements in z-pinch and plasma focus experiments", DOI: https://doi.org/10.1063/1.3559548
    - Why: detector timing and sensitivity calibration needed for predictive yield closure.

### spatial_temperature

- Physics need: direct or bounded spatial temperature validation
- Current KR status: partial only in same-scope targets
- Current KR source: `KnowledgeReference/scholz-2007-pf1000-part2-jphysd.md`
- Required data:
  - `direct_experimental_temperature_diagnostic`
- Source leads and local status:
  - Gribkov et al. (2007), "Plasma dynamics in PF-1000 device under full-scale energy storage: I. Pinch dynamics, shock-wave diffraction, and inertial electrode", DOI: https://doi.org/10.1088/0022-3727/40/7/021
    - Why: PF-1000 pinch temperature estimates and diagnostic context.
  - Goyon et al. (2025), "Neutron generation dynamics inside a MA-class dense plasma focus Z-pinch", DOI: https://doi.org/10.1063/5.0253547
    - Why: MJOLNIR stagnation-temperature and neutron-mechanism comparison for physics closure.

### uncertainty

- Physics need: same-scope uncertainty budget
- Current KR status: partial only in same-scope targets
- Current KR source: `KnowledgeReference/scholz-2007-pf1000-part2-jphysd.md`
- Required data:
  - `fast_ion_distribution_uncertainty`
  - `same-shot density uncertainty`
- Source leads and local status:
  - Malir et al. (2024), "Comparison of density profiles measured via laser interferometry with MHD simulations during shock wave reflection on mega-ampere dense plasma focus", DOI: https://doi.org/10.1063/5.0193268
    - Why: published density-profile uncertainty and setup limitations.
  - Klir et al. (2011), "Fusion neutron detector for time-of-flight measurements in z-pinch and plasma focus experiments", DOI: https://doi.org/10.1063/1.3559548
    - Why: detector timing and sensitivity calibration uncertainty.

## Priority 2 Queue

### neutron_anisotropy

- Physics need: neutron angular anisotropy and beam-target constraint
- Current KR status: partial only in same-scope targets
- Current KR source: `KnowledgeReference/scholz-2007-pf1000-part2-jphysd.md`
- Required data:
  - `fast_ion_distribution_uncertainty`
- Source leads and local status:
  - Sadowski/Scholz/PF-1000 team (2004), "Measurements of fast ions and neutrons emitted from PF-1000 plasma focus device", DOI: https://doi.org/10.1016/j.vacuum.2004.07.040
    - Why: PF-1000 angular neutron anisotropy and fast-ion diagnostics.
  - Springham et al. (2021), "Plasma focus neutron energy and anisotropy measurements using zirconium-beryllium pair activation detectors", DOI: https://doi.org/10.1016/j.nima.2020.164830
    - Why: activation-detector method for neutron energy and anisotropy.

### neutron_detector_response

- Physics need: neutron detector or activation response
- Current KR status: partial only in same-scope targets
- Current KR source: `KnowledgeReference/scholz-2007-pf1000-part2-jphysd.md`
- Required data:
  - `neutron_field_transport_or_room_scatter_response_model`
- Source leads and local status:
  - Klir et al. (2011), "Fusion neutron detector for time-of-flight measurements in z-pinch and plasma focus experiments", DOI: https://doi.org/10.1063/1.3559548
    - Why: PF-1000-relevant TOF detector calibration and response.
  - Jednorog et al. (2017), "A new concept of fusion neutron monitoring for PF-1000 device", DOI: https://doi.org/10.1515/nuka-2017-0003
    - Why: PF-1000 activation monitoring concept and diagnostic response.

### neutron_spectrum

- Physics need: neutron energy spectrum
- Current KR status: partial only in same-scope targets
- Current KR source: `KnowledgeReference/scholz-2007-pf1000-part2-jphysd.md`
- Required data:
  - `digitized_neutron_spectra`
- Source leads and local status:
  - Sadowski/Scholz/PF-1000 team (2004), "Measurements of fast ions and neutrons emitted from PF-1000 plasma focus device", DOI: https://doi.org/10.1016/j.vacuum.2004.07.040
    - Why: PF-1000 neutron spectra, anisotropy, yield, and fast-ion measurements.
  - Catenacci et al. (2020), "Tomographic Reconstruction of the Neutron Time-Energy Spectrum from a Dense Plasma Focus", DOI: https://doi.org/10.1109/TPS.2020.3012104
    - Why: DPF neutron time-energy spectrum reconstruction method with scatter subtraction.

### neutron_timing

- Physics need: neutron time history
- Current KR status: partial only in same-scope targets
- Current KR source: `KnowledgeReference/scholz-2007-pf1000-part2-jphysd.md`
- Required data:
  - `digitized_neutron_pulse_traces`
- Source leads and local status:
  - Gribkov et al. (2007), "Plasma dynamics in the PF-1000 device under full-scale energy storage: II. Fast electron and ion characteristics versus neutron emission parameters and gun optimization perspectives", DOI: https://doi.org/10.1088/0022-3727/40/12/008
    - Why: PF-1000 fast ion/electron timing versus neutron emission.
  - Goyon et al. (2025), "Neutron generation dynamics inside a MA-class dense plasma focus Z-pinch", DOI: https://doi.org/10.1063/5.0253547
    - Why: neutron pulse-shape comparison and mechanism-separated timing.

### spatial_magnetic_or_em

- Physics need: spatial magnetic or electromagnetic field validation
- Current KR status: partial only in same-scope targets
- Current KR source: `KnowledgeReference/scholz-2007-pf1000-part2-jphysd.md`
- Required data:
  - `same-shot calibrated magnetic-field map`
- Source leads and local status:
  - Gribkov et al. (2007), "Plasma dynamics in PF-1000 device under full-scale energy storage: I. Pinch dynamics, shock-wave diffraction, and inertial electrode", DOI: https://doi.org/10.1088/0022-3727/40/7/021
    - Why: magnetic-probe/Faraday-current-pinch context for PF-1000.
  - Schmidt et al. (2022), "Measuring characteristic differences between high- and low-performing discharges on the MJOLNIR DPF", DOI: https://doi.org/10.1063/5.0089121
    - Why: restrike and parasitic-current evidence relevant to EM coupling closure.

## Next Ratchet After Acquisition

For each acquired document:

1. Add the `.md` source under `KnowledgeReference/` with stable naming.
2. Record document hash and source metadata.
3. Extract or explicitly reject candidate target data under KR-only review.
4. For figures, create a digitization packet and pass
   `digitization_verification_evidence()`.
5. Add typed target records in `src/dpf/validation/kr_targets.py`.
6. Rerun `kr_validation_same_scope_target_report()`,
   `scientific_accuracy_gap_report()`, predictive readiness, and
   high-fidelity readiness.

## Local PDF Audit

Updated: 2026-05-06

I checked local PDFs under the DPF-Unified tree by filename, PDF metadata,
DOI/title text extraction, and SHA-256 duplicate checks. A local PDF outside
`KnowledgeReference/` is still not source-of-truth evidence; it is only
available for ingestion.

### Exact Local Matches

| Candidate document | Local PDF status | Notes |
| --- | --- | --- |
| Akel, Kubes, Paduch, Lee (2021), DOI `10.1016/j.radphyschem.2021.109633` | Found outside KR at `archive_reference_OLD/references/papers/core-dpf/akel-2021-pf1000-neutron-yield.pdf`; duplicate same hash at `archive_reference_OLD/references/papers/archive/akel-2021-pf1000-neutron-yield.pdf` | Exact metadata match. Already represented in KR as `KnowledgeReference/radiation-physics-and-chemistry-188-2021-109633.md`; text parity passed for all 6 pages. |
| Gribkov et al. (2007) Part I, DOI `10.1088/0022-3727/40/7/021` | Found at `archive_reference_OLD/references/papers/core-dpf/gribkov-2007-pf1000-jphysd-part2.pdf`; KR markdown exists at `KnowledgeReference/gribkov-2007-pf1000-jphysd-part2.md` | Filename is misleading: content is Part I, not Part II. Treat canonical identity by title/DOI, not filename. |
| Gribkov et al. (2007) Part II, DOI `10.1088/0022-3727/40/12/008` | Found at `archive_reference_OLD/references/papers/core-dpf/scholz-2007-pf1000-part2-jphysd.pdf`; KR markdown exists at `KnowledgeReference/scholz-2007-pf1000-part2-jphysd.md` | Filename is misleading: first author is Gribkov and the paper is Part II. Treat canonical identity by title/DOI, not filename. |
| Schmidt et al. (2022), DOI `10.1063/5.0089121` | Found at `archive_reference_OLD/references/papers/core-dpf/goyon-2022-mjolnir-high-low.pdf`; KR markdown exists at `KnowledgeReference/goyon-2022-mjolnir-high-low.md` | Filename is author-misleading: first author is Schmidt. Also found LLNL accepted-manuscript/preprint form at `archive_reference_OLD/references/papers/core-dpf/petrov-2022-mjolnir-high-low-discharges.pdf` and KR markdown at `KnowledgeReference/petrov-2022-mjolnir-high-low-discharges.md`. |
| Malir et al. (2024), DOI `10.1063/5.0193268` | Found at `archive_reference_OLD/references/papers/core-dpf/malir-2024-interferometry-dpf.pdf`; KR markdown exists at `KnowledgeReference/malir-2024-interferometry-dpf.md` | Exact match already in KR markdown. |
| Goyon et al. (2025), DOI `10.1063/5.0253547` | Found at `archive_reference_OLD/references/papers/core-dpf/goyon-2025-ma-class-dpf-neutron.pdf` and `archive_reference_OLD/references/papers/core-dpf/Neutron_generation_dynamics_inside_a_MA-class_dens.pdf`; both have identical SHA-256; duplicate copies also exist under `archive_reference_OLD/references/papers/archive/` | Exact match. KR markdown exists under long title filenames: `KnowledgeReference/neutron-generation-dynamics-inside-a-ma-class-dense-plasma-focus-z-pinch.md` and `KnowledgeReference/neutron-generation-dynamics-inside-a-ma-class-dense-plasma-focus-z-pinch-5.md`. |

### Supplemental Intake Status

The earlier "not found" list is partially superseded by the 2026-05-11
supplemental intake. These sources now have local KR text records but still
need title cleanup, target-array completion, figure/table digitization, and
independent review before validation use. Figure/table, formula, numeric-target,
and uncertainty-copy fidelity was second-pass checked on 2026-05-11 by
`scripts/verify_kr_source_fidelity.py --apply`; recovered artifacts are now
stored in each same-stem KR JSON under `source_fidelity_review` and summarized
in the same-stem Markdown.

- Cikhardtova et al. (2015), DOI `10.1515/nuka-2015-0065`
- Szydlowski/Sadowski/Scholz/PF-1000 team (2004), DOI
  `10.1016/j.vacuum.2004.07.040`
- Catenacci et al. (2020), DOI `10.1109/TPS.2020.3012104`
- Springham et al. (2021), DOI `10.1016/j.nima.2020.164830`
- Klir et al. (2011), DOI `10.1063/1.3559548`

Target extraction and digitization start, 2026-05-11:

- New typed target records were started for Cikhardtova 2015,
  Szydlowski 2004, Klir 2011, Springham 2021, and Catenacci 2020 in
  `src/dpf/validation/kr_targets.py`.
- A dated starter report was written to
  `docs/TARGET_EXTRACTION_DIGITIZATION_2026_05_11.md` / `.json`.
- The first 23 cited pages were rendered as crop-pending workbench images under
  `KnowledgeReference/figures/target-extraction/2026-05-11/`.
- The first 36 unreviewed crop candidates were generated and hash-recorded:
  six for Cikhardtova 2015 Figs. 1-6, five for Szydlowski 2004 Figs. 1-5,
  four for Klir 2011 Figs. 1-4, nine for Springham 2021 Figs. 1-7 and
  Tables 1-2, and twelve for Catenacci 2020 Figs. 1-8 and Tables I-IV.
- Status boundary: these records are `target_record_started_page_rendered_crop_pending`.
  None of the rendered pages, crop candidates, or extracted scalar target
  records is accepted figure digitization evidence until a cropped figure/table
  packet passes `digitization_verification_evidence()` and independent review.

A14 table extraction draft pass, 2026-05-11:

- A draft table packet bundle was written to
  `KnowledgeReference/digitization/a14-2026-05-11-table-draft-packets.json`,
  with a companion review report at
  `docs/A14_TABLE_EXTRACTION_DRAFTS_2026_05_11.md`.
- The bundle contains six source-bound draft table packets: Springham 2021
  Tables 1-2 and Catenacci 2020 Tables I-IV.
- Each packet records local KR source hash, local PDF hash, crop-image hash,
  source line window, table rows, and numeric series. Each packet remains
  `draft_unreviewed` with `accepted_for_validation=false`.
- `digitization_verification_evidence()` currently fails every A14 table draft
  only on `independent_review_missing` and `review_status_not_accepted`.
  This is intentional; the draft tables cannot support validation until
  independent review metadata is bound to the current hashes and accepted.
- Review hardening update: `a14_table_extraction_draft_packets()` now adds a
  stable per-table item hash for review binding, and
  `digitization_verification_evidence()` checks table crop-image hashes before
  any accepted review can pass. The verifier also checks declared local
  `source_pdf_path`/`source_pdf_sha256` pairs and requires accepted review
  metadata to match `reviewed_source_pdf_sha256`.

A14 crop-boundary QA pass, 2026-05-11:

- A crop-boundary status report was written to
  `docs/A14_CROP_BOUNDARY_REVIEW_2026_05_11.md` / `.json`.
- The report covers all 36 A14 crop candidates and keeps
  `accepted_for_validation_count=0`.
- Visual QA status is now explicit after crop-rectangle cleanup: 21 figure
  crops are `boundary_ready_for_draft_extraction`, 9 diagram/image crops are
  `manual_review_required`, 0 crops are `crop_adjustment_needed`, and the 6
  table crops are `draft_extracted_review_blocked`.
- The first recommended axis-calibration targets are Cikhardtova 2015 Fig. 6,
  Klir 2011 Fig. 2, and Springham 2021 Fig. 5.
- Status boundary: this QA pass does not accept any figure/table digitization
  evidence. Boundary-ready means suitable for draft calibration/extraction
  only; validation use still requires calibrated arrays, residual evidence, and
  accepted independent review through `digitization_verification_evidence()`.

A14 axis-calibration draft pass, 2026-05-11:

- A draft figure-calibration bundle was written to
  `KnowledgeReference/digitization/a14-2026-05-11-axis-calibration-draft-packets.json`,
  with a companion report at
  `docs/A14_AXIS_CALIBRATION_DRAFTS_2026_05_11.md`.
- The bundle covers Cikhardtova 2015 Fig. 6, Klir 2011 Fig. 2, and Springham
  2021 Fig. 5.
- Each packet records local KR source hash, local PDF hash, crop-image hash,
  source line window, visible axis/range metadata, visible series labels, and
  extraction notes.
- Status boundary: these packets are not digitized data. They contain no
  curve arrays, no residuals, no independent review, and
  `accepted_for_validation=false`.

A14 Springham Fig. 5 numeric draft pass, 2026-05-11:

- A draft digitization packet was written to
  `KnowledgeReference/digitization/a14-2026-05-11-springham-fig5-monoenergetic-draft-packet.json`,
  with a companion report at
  `docs/A14_SPRINGHAM_FIG5_DIGITIZATION_DRAFT_2026_05_11.md`.
- The packet contains 14 candidate points for the visible blue
  mono-energetic-neutron curve in Springham 2021 Fig. 5, plus source hash,
  local PDF hash, crop hash, axis calibration, and pixel-pick metadata.
- Residual status: the draft round-trip residual check now reports RMS
  `0.002049609754498783 px` and max `0.0031865149536866814 px` by projecting
  the candidate values back through the Fig. 5 axis calibration and comparing
  against the recorded draft pixel picks.
- Status boundary: this packet is a numeric draft only. It still fails
  `digitization_verification_evidence()` on `independent_review_missing` and
  `review_status_not_accepted`, and remains `accepted_for_validation=false`.
- Review-gate hardening: tests now prove a synthetic accepted Springham Fig. 5
  packet can pass only when review metadata binds to the current packet,
  source, local PDF, and figure hashes. This does not accept the real packet.
- A companion Gaussian-curve draft packet was written to
  `KnowledgeReference/digitization/a14-2026-05-11-springham-fig5-gaussian-curves-draft-packet.json`,
  with a report at
  `docs/A14_SPRINGHAM_FIG5_GAUSSIAN_CURVES_DRAFT_2026_05_11.md`. It contains
  the visible black 200 keV FWHM and red 400 keV FWHM response curves only;
  hidden segments under plot annotations were not synthesized.

A14 independent-review handoff pass, 2026-05-11:

- A reviewer-facing handoff manifest was written to
  `docs/A14_INDEPENDENT_REVIEW_HANDOFF_2026_05_11.json`, with a companion
  Markdown report at `docs/A14_INDEPENDENT_REVIEW_HANDOFF_2026_05_11.md`.
- The manifest lists nine reviewable draft packets: six A14 table drafts, the
  Springham Fig. 5 mono-energetic numeric draft, the Springham Fig. 5
  Gaussian-curve numeric draft, and the Klir Fig. 2 timing-response draft. It
  also lists three axis-calibration scaffolds as context-only items.
- Status boundary: the handoff is review readiness only. It preserves
  `accepted_for_validation_count=0`; actual validation use still requires
  independent accepted review metadata bound to current packet/source/local-PDF
  and figure-or-crop hashes, plus a passing
  `digitization_verification_evidence()` result.

A14 Klir Fig. 2 timing-response draft pass, 2026-05-11:

- A draft digitization packet was written to
  `KnowledgeReference/digitization/a14-2026-05-11-klir-fig2-timing-response-draft-packet.json`,
  with a companion report at
  `docs/A14_KLIR_FIG2_TIMING_RESPONSE_DRAFT_2026_05_11.md`.
- The packet contains two visible curve-centerline series from Klir 2011
  Fig. 2: FWHM and rise time versus PMT voltage.
- Status boundary: the source caption states error bars indicate +/-2 sigma,
  but numeric error-bar extents are not extracted in this packet. It remains
  `accepted_for_validation=false` and review-blocked.

A14 Cikhardtova Fig. 6 extraction blocker, 2026-05-11:

- A blocker report was written to
  `docs/A14_CIKHARDTOVA_FIG6_EXTRACTION_BLOCKER_2026_05_11.json`, with a
  companion Markdown report at
  `docs/A14_CIKHARDTOVA_FIG6_EXTRACTION_BLOCKER_2026_05_11.md`.
- The report records five visible linear-density series from Cikhardtova 2015
  Fig. 6 but does not create numeric arrays because monochrome line styles
  overlap and nearly merge.
- Status boundary: manual or vector-assisted curve separation is required
  before any Cikhardtova Fig. 6 digitization packet can be submitted for
  independent review.

A14 remaining-extraction backlog, 2026-05-11:

- A generated backlog was written to
  `docs/A14_REMAINING_EXTRACTION_BACKLOG_2026_05_11.json`, with a companion
  Markdown report at `docs/A14_REMAINING_EXTRACTION_BACKLOG_2026_05_11.md`.
- Current generated counts: 36 crop candidates, 9 reviewable draft packets
  across 8 distinct crops, 18 ready-not-started crops, 9 manual-review crops,
  1 blocked crop, and 0 accepted validation items.
- Status boundary: the backlog is a planning artifact. It does not accept any
  A14 packet or crop for validation use.

Still not found as an exact local PDF in this queue:

- Jednorog et al. (2017), "A new concept of fusion neutron monitoring for
  PF-1000 device", DOI `10.1515/nuka-2017-0003`

### Parity-Verified KR Records

The exact local matches already have `KnowledgeReference` markdown/JSON pairs
and passed text parity against the local PDFs:

- Akel et al. 2021: `KnowledgeReference/radiation-physics-and-chemistry-188-2021-109633.md`, 6/6 pages, PDF SHA-256 `9a762bc36bc1f5c175a0ec8dc07b69c48ad956d0c6a382882daf4e24677dcb3b`
- Gribkov et al. 2007 Part I: `KnowledgeReference/gribkov-2007-pf1000-jphysd-part2.md`, 13/13 pages, PDF SHA-256 `7acfb46d1db6ee5894978f70e1372edda7efaa5171d8e7c3bdf0baf7025eff43`
- Gribkov et al. 2007 Part II: `KnowledgeReference/scholz-2007-pf1000-part2-jphysd.md`, 16/16 pages, PDF SHA-256 `c4d62f5015bc6040aa85070e43f3cb6e7e4a8329e5d2baf33fa4d38f828caa4f`
- Schmidt et al. 2022 MJOLNIR article: `KnowledgeReference/goyon-2022-mjolnir-high-low.md`, 29/29 pages, PDF SHA-256 `89877f5c880dcd9c4454925984398cf51984f95d2ff78ac4437f5f755e98fe6a`
- Schmidt/Goyon accepted-manuscript copy: `KnowledgeReference/petrov-2022-mjolnir-high-low-discharges.md`, 16/16 pages, PDF SHA-256 `d9674bd39b12c3a87e7549c540384f56722d739f5b85a693fab73c24b2d32623`
- Malir et al. 2024: `KnowledgeReference/malir-2024-interferometry-dpf.md`, 14/14 pages, PDF SHA-256 `fafc32261c9172702b1c8dfdc92bcc33b1a32aeeb4cb9680d535478191db46c9`
- Goyon et al. 2025 canonical KR record: `KnowledgeReference/neutron-generation-dynamics-inside-a-ma-class-dense-plasma-focus-z-pinch.md`, 10/10 pages, PDF SHA-256 `9c0bc58d72ced9c914914aabdab63937a2b9c7820950eb0fa2412be9fd9d0f8c`
- Goyon et al. 2025 short-name KR duplicate: `KnowledgeReference/neutron-generation-dynamics-inside-a-ma-class-dense-plasma-focus-z-pinch-5.md`, 10/10 pages, PDF SHA-256 `9c0bc58d72ced9c914914aabdab63937a2b9c7820950eb0fa2412be9fd9d0f8c`

### Immediate Next Action

No new `KnowledgeReference` markdown file was created in this pass because all
exact local PDF matches already had KR markdown/JSON pairs and passed text
parity. Akel et al. 2021 Tables 1 and 2 are now typed as
`pf1000_16kv_shot_table_2021_akel`, with scalar/yield comparison available
through `pf1000_16kv_akel_table_candidate_evidence()`. The next local-only
ratchet is verified digitization of Akel et al. 2021 waveform/yield figures
through `digitization_verification_evidence()`.
