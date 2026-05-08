# Scientific Closure Source Queue

Updated: 2026-05-07

This queue is not source-of-truth evidence. The source of truth remains local
files under `KnowledgeReference/`. These candidate links become usable only
after the user acquires the correct document, the document is added locally
under `KnowledgeReference/`, Codex reviews it under the KR-only rule, and any
digitized data passes one-for-one provenance verification.

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
parity-verified local `KnowledgeReference` markdown records. Cikhardtova 2015,
Sadowski/Scholz 2004, Catenacci 2020, Springham 2021, Klir 2011, and Jednorog
2017 were not found as exact local PDFs in the DPF-Unified tree and remain
user-acquisition candidates.

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

### Not Found As Exact Local PDFs

These candidates were checked by DOI/title text extraction and filename search.
No exact local PDF match was found in the DPF-Unified tree:

- Cikhardtova et al. (2015), "Temporal distribution of linear densities of the plasma column in a plasma focus discharge", DOI `10.1515/nuka-2015-0065`
- Sadowski/Scholz/PF-1000 team (2004), "Measurements of fast ions and neutrons emitted from PF-1000 plasma focus device", DOI `10.1016/j.vacuum.2004.07.040`
- Catenacci et al. (2020), "Tomographic Reconstruction of the Neutron Time-Energy Spectrum from a Dense Plasma Focus", DOI `10.1109/TPS.2020.3012104`
- Springham et al. (2021), "Plasma focus neutron energy and anisotropy measurements using zirconium-beryllium pair activation detectors", DOI `10.1016/j.nima.2020.164830`
- Klir et al. (2011), "Fusion neutron detector for time-of-flight measurements in z-pinch and plasma focus experiments", DOI `10.1063/1.3559548`
- Jednorog et al. (2017), "A new concept of fusion neutron monitoring for PF-1000 device", DOI `10.1515/nuka-2017-0003`

Several other PDFs contain these DOIs or title fragments only in reference
lists, including `2025_Numerical_studies_of_plasma_emission_in_a_mega_joule_plasma_focus_using_Lee_code.pdf`,
`AD1079881_DPF_optimization.pdf`, `2024_Deuteron_beam_fluence_emitted_from_dense_plasma_focus_Comparative_investigation.pdf`,
and `2 ICTP e-manual 2.pdf`. They are not exact matches for the candidate
documents.

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
