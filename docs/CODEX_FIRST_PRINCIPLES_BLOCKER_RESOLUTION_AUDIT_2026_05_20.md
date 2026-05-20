# Codex Audit - First-Principles Blocker Resolution Handoff - 2026-05-20

Audited deliverable:
`docs/FIRST_PRINCIPLES_BLOCKER_RESOLUTION_AUDIT_HANDOFF_2026_05_20.md`

Audit scope:

- Current repository: `/Users/anthonyzamora/dpf-unified`
- Current branch: `codex/corpus`
- Current audited HEAD: `7999265`
- Source authority: repo-local `KnowledgeReference/` only. On-disk PDFs outside
  KR are ingestion candidates, not accepted physics authority.
- Current periodic audit evidence:
  `/private/tmp/dpf-unified-audit-logs/20260520T051600Z/summary.md` reports
  10/10 gates passing at HEAD
  `7999265c9a25aa4362743804a61053240c9280b7`.

## Verdict

**Conditional accept as a research triage artifact. Do not accept it as an
authoritative Sprint 5 execution packet until the corrections below are made.**

The handoff is directionally useful and several high-value findings are
confirmed, especially Bennett 2017 startup relevance, PF-1000 cathode-cage
geometry evidence, the qualitative lower-hybrid anomalous-resistivity source,
and the unchanged comparator-scope problem. It is not clean enough to hand to
implementation as-is because several rows are stale relative to current KR
promotion/target-extraction state, some line citations are shorthand or
incomplete, and several "resolved" statements need to be narrowed to
source-availability or method-context status.

No first-principles runtime acceptance, whole-shot readiness, neutron authority,
startup BVP acceptance, transport closure acceptance, or validation certificate
is promoted by this audit.

## Confirmed Findings

### Bennett 2017 Startup Source

The on-disk PDF exists:

`archive_reference_OLD/references/papers/core-dpf/schmidt-2017-kinetic-dpf-breakdown.pdf`

The filename is misleading. PDF text confirms the source is:

N. Bennett et al., "Kinetic simulations of gas breakdown in the dense plasma
focus", Phys. Plasmas 24, 062705 (2017), DOI `10.1063/1.4985313`.

Confirmed useful source content from the local PDF text:

- fully kinetic collisional EM PIC treatment of DPF gas breakdown;
- neutral deuterium at 5.5 Torr and seed plasma density `1e7 cm^-3`;
- simulated plasma density about `1e15 cm^-3` along the insulator by 100 ns;
- ion-density contours at 100 ns, 500 ns, and 1 us;
- sheath current fraction statement at 1 us, not 500 ns;
- electric-field explosive-emission thresholds of 250 kV/cm generally and
  10 kV/cm at the knife-edge approximation;
- photoionization only changes electron density by 1.2 percent by 125 ns, so
  Bennett neglects photons.

Audit status: promote and target-extract Bennett 2017, but phrase its effect as
"closes source-availability and target-extraction blockers for startup channels"
rather than "resolves the startup BVP." It cannot by itself accept a PF-1000 or
Akel whole-shot startup BVP.

### PF-1000 Cathode-Cage Radius

The 200 mm hardware radius finding is source-supported, with the caveat that
PF-1000 has multiple configurations and the runtime must select configuration
scope explicitly.

Verified KR evidence:

- `KnowledgeReference/experimental-study-of-the-structure-of-the-plasma-current-sheath-on-the-pf-1000-facility-705bcc83.md:344-349`
  states 12 stainless-steel rods, 80 mm rod diameter, OE radius 200 mm, CE
  radius 115.5 mm, and 85 mm alumina insulator extension.
- `KnowledgeReference/gribkov-2007-pf1000-jphysd-part2.md:392-400`
  states 12 stainless-steel rods, each 40 mm diameter and 800 mm length,
  distributed around a 400 mm diameter circumference, and a second 24-rod
  configuration also on a 400 mm perimeter.
- `KnowledgeReference/chunks/update-on-the-scientific-status-of-the-plasma-focus-1385adeb/pages-0026-0050.md:822-849`
  summarizes PF-1000 with a 230 mm anode tube, 12 stainless-steel tubes on a
  400 mm diameter cathode, alumina insulator, and 1.4 m x 2.5 m stainless-steel
  chamber.

Audit status: the 200 mm hardware radius is supported for the cited PF-1000
hardware context. The Akel 160 mm value remains a different category, not a
hardware radius replacement.

### Stepniewski Hollow-Anode Bore

The current target-extraction state is correct:

- `docs/FIRST_PRINCIPLES_TARGET_EXTRACTIONS_2026_05_20.md:18,37-42`
  records the Stepniewski 0.015 m hollow radius as
  `target_extracted_modeling_context_requires_review`.
- `KnowledgeReference/doi-10-1016-j-vacuum-2004-05-019-f931cb0b.md:310-314`
  gives PF-1000 simulation parameters including inner electrode radius 0.12 m,
  outer electrode 0.18 m, hollow radius 0.015 m, and electrode length 0.60 m.

Audit status: keep blocked for hardware-scope runtime geometry until a
hardware-scope source or review verdict is attached.

### Current-Sheath / Liftoff Method Context

The current-sheath KR source does contain the mass-sweep and pressure-regime
material, but the handoff's cited line range is incomplete.

Verified KR evidence:

- `KnowledgeReference/effect-of-current-sheath-initiation-on-the-radial-collapse-and-energetic-particle-accelera-b2e95b88.md:597-601`
  contains the variable mass-sweeping factor and the example values at 2, 4,
  and 6 Torr.
- `KnowledgeReference/effect-of-current-sheath-initiation-on-the-radial-collapse-and-energetic-particle-accelera-b2e95b88.md:631-640`
  contains the Paschen-style low/medium/high pressure regimes and the caveat
  that Paschen-type physics is fragile for DPF breakdown.
- `KnowledgeReference/effect-of-current-sheath-initiation-on-the-radial-collapse-and-energetic-particle-accelera-b2e95b88.md:642-660`
  contains the Te about 4 eV assumption and `Liz(P)/Li = 2.4` startup scaling
  context.

Audit status: useful wrong-scope method context only. Update any target record
that claims `massf` so it cites `597-601`, not only `616-670`.

### Bernard 1977 Historical Source

Bernard 1977 is already promoted into KR:

`KnowledgeReference/the-dense-plasma-focus-a-high-intensity-neutron-source-f0a3910d.md`

Verified KR evidence:

- lines `455-461`: independent scattering measurement gives deuteron
  temperature 700 eV in the filament phase of a 500 kA machine;
- lines `976-1033`: scattering method used for deuteron temperature;
- lines `1185-1193`: neutron spectra at 0, 45, and 90 degrees for the same
  discharge, modeled with incident deuterons between 30 and 350 keV;
- lines `1546-1547`: U, I, and L versus time for the 1 MJ Frascati Mather-type
  DPF.

Audit status: valuable historical comparator/method evidence, not PF-1000
same-scope Te/Ti support.

### Qualitative DPF Anomalous Resistivity

The qualitative DPF lower-hybrid anomalous-resistivity claim is supported.

Verified KR evidence:

- `KnowledgeReference/comparisons-of-dense-plasma-focus-kinetic-simulations-with-experimental-measurements.md:46-49`
  states anomalous resistivity arises due to a kinetic instability near the
  lower-hybrid frequency.
- lines `156-171` connect observed/simulated 3-4 GHz fluctuations with lower
  hybrid frequency estimates for 10-40 T fields.
- lines `185-190` compare inferred anomalous resistivity of 56 or 16 mOhm-cm
  with classical resistivity around 20 microOhm-cm.

Audit status: qualitative source support only. A quantitative DPF-regime
closure still requires target extraction or external acquisition.

### Gribkov Part II Is Already Local KR

The handoff lists Gribkov et al. 2007 J. Phys. D 40:3592 as an external
acquisition for `NEUTRON-BLK-001`. That is stale.

Current repo evidence:

- `docs/SCIENTIFIC_CLOSURE_SOURCE_QUEUE.md:470-471` records Gribkov Part I and
  Part II as found locally and represented in KR.
- `KnowledgeReference/scholz-2007-pf1000-part2-jphysd.md:1-31` confirms the
  Part II identity: J. Phys. D 40 (2007) 3592-3607, DOI
  `10.1088/0022-3727/40/12/008`.
- `KnowledgeReference/scholz-2007-pf1000-part2-jphysd.md:318-323` begins the
  direct fast-deuteron measurement section.
- lines `445-460` give the five-counter neutron anisotropy setup/result with
  about `Y0/Y90 = 1.8`.
- lines `1138-1168` discuss neutron spectra from fast deuterons and the 100 keV
  beam-target angular distribution context.

Audit status: reclassify from `EXTERNAL_ACQUISITION_REQUIRED` to existing KR
target extraction/review. It still does not provide a complete accepted
distribution function for runtime authority.

## Blocking Corrections Required Before Sprint 5 Uses This Handoff

1. **Reconcile the HEAD narrative.** The handoff header says research-completion
   HEAD `022b774`; the executive summary references `da97ed2`; current audited
   repo state is `7999265` with a 10/10 periodic audit pass. The packet must
   state which commit it audits and which later commits supersede it.

2. **Fix the category counts.** The executive-summary table says 21 blockers,
   but the listed category counts sum to 23, and the external-acquisition table
   contains about 20 rows. Split "blocker count" from "source-acquisition row
   count."

3. **Reclassify Talebitaher.** The handoff lists a Talebitaher PhD subset as
   `KR_PROMOTION_RECOMMENDED`, but current repo state already has it promoted
   and target-extracted as
   `target_extracted_nx2_detector_anisotropy_context` in
   `docs/FIRST_PRINCIPLES_TARGET_EXTRACTIONS_2026_05_20.md:22` and
   `src/dpf/first_principles/source_targets.py`.

4. **Reclassify Bernard 1977.** The handoff lists Bernard 1977 in external
   acquisition, but `docs/CORPUS_RESCAN_KR_PROMOTION_2026_05_20.md:18-19`
   shows it is already promoted into KR. It should be "existing KR target
   extraction / historical comparator scope", not external acquisition.

5. **Reclassify Gribkov Part II.** Gribkov et al. 2007 J. Phys. D 40:3592 is
   already in KR at
   `KnowledgeReference/scholz-2007-pf1000-part2-jphysd.md`. It should not stay
   in the external-acquisition list.

6. **Fix current-sheath line ranges.** The `massf` formula is at lines
   `597-601`, not `615-670`. Lines `631-660` support the pressure regime and
   `Liz/Li` material. The handoff and any extraction packet should cite both
   ranges explicitly.

7. **Correct Bennett timing.** The handoff says the 71 percent current fraction
   is "by 500 ns"; the PDF text places that statement at 1 us. Keep the 100 ns,
   500 ns, and 1 us contour targets, but correct the current-fraction timing.

8. **Downgrade Braginskii table/equation certainty until OCR/render review.**
   The PDF exists and is high priority, but `pdftotext` did not expose Table 2
   or Eqs. 4.30-4.45 in this audit. Treat those as
   `PDF_PRESENT_NEEDS_RENDERED_PAGE_OR_OCR_VERIFICATION` before target
   extraction.

9. **Replace shorthand citations.** Rows such as
   `experimental-study-...-705bcc83.md` and
   `effect-of-current-sheath-initiation-...-b2e95b88.md` must use full
   `KnowledgeReference/...` paths. Shorthand is acceptable in prose only after a
   full path has already been introduced.

10. **Narrow "promotion resolves blocker" language.** Promotion only creates a
    KR text source. Target extraction creates typed source evidence. Runtime
    acceptance still requires scope review, closure implementation, numerical
    verification, same-scope comparison, UQ, and certificate gates.

## Required V2 Handoff Structure

The team should submit a corrected V2 packet with this exact structure:

1. Current audited HEAD and list of superseding commits checked.
2. One table per blocker, not per source, with columns:
   `blocker_id`, `current_repo_status`, `corrected_status`,
   `source_or_acquisition`, `exact_path_or_full_citation`, `line_or_page_range`,
   `scope_tag`, `runtime_claim_allowed`, `remaining_action`.
3. Separate source-acquisition table with columns:
   `priority`, `source`, `resolves_blockers`, `already_in_kr`, `on_disk_path`,
   `external_required`, `notes`.
4. All KR citations as full local paths plus lines.
5. Explicit scope tags:
   `pf1000_full_energy`, `pf1000_akel_16kv`, `nx2_wrong_scope`,
   `historical_mather_wrong_scope`, `generic_formulary`, or
   `external_candidate`.
6. For every "resolved" row, a final field named `accepted_runtime_claim` that
   must be `false` unless the corresponding code/test/certificate gate exists.

## Next Work For The Engineering Team

Priority 1 - corrections:

- Produce V2 of
  `docs/FIRST_PRINCIPLES_BLOCKER_RESOLUTION_AUDIT_HANDOFF_2026_05_20.md` with
  the blocking corrections above fixed.
- Add a small status appendix that lists all rows reclassified from the first
  handoff.

Priority 2 - source promotion / extraction:

- Promote and target-extract Bennett 2017 with startup targets for seed density,
  fill pressure, geometry context, pressure regimes, explosive-emission
  thresholds, density evolution at 100 ns / 500 ns / 1 us, electron temperature,
  current-fraction timing, breakdown-delay comparison, and photoionization
  bound.
- Render/OCR Braginskii 1965 pages before extracting exact coefficient tables
  and equations. Attach rendered-page evidence in the extraction note.
- Target-extract Gribkov Part II fast-ion and neutron anisotropy material from
  existing KR. Mark it PF-1000 full-energy only.
- If massf is promoted into a startup method target, update the source line
  range to include `597-601`.

Priority 3 - external acquisition:

- Keep these as true external MUST candidates unless the team proves local
  coverage: D2 Townsend/Paschen/LXCat data, D2 electron-neutral cross sections,
  molecular D2 ionization/recombination, secondary emission for the relevant
  surfaces, Brysk Doppler broadening, deuteron stopping tables, and quantitative
  DPF anomalous-resistivity formula sources.
- Keep PF-1000 insulator outer radius, insulator wall thickness, backplate
  radial extent, and backplate axial thickness on the facility-request path.

## Audit Result

The handoff is not rejected. It correctly found several important leads and
properly preserved the global acceptance boundary. It is also not clean enough
to drive implementation without a corrected V2 because at least three rows are
already superseded by current repo state, one line-range claim is wrong, one
Bennett timing claim is wrong, and Braginskii table/equation support has not yet
been machine-verifiable from the PDF text.

Sprint 5 should proceed only after the V2 correction packet is submitted or the
implementation team explicitly consumes this audit as the controlling errata.
