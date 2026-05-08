# CortexFindings

This file records the execution plan requested on 2026-05-05. `CodexFindings.md`
remains the running findings log; this file is the detailed plan and review
artifact named in the latest request.

## Goal

Reach a validated end-to-end predictive Dense Plasma Focus simulation tool using
only local `KnowledgeReference/` documents as scientific source authority.

## Plan Review

The current codebase has useful validation gates, but gates are not the same as
validated physics. The plan must therefore prioritize real target extraction,
production evidence generation, and scope control over adding more nominal
metadata. A result should become predictive only when it carries line-referenced
KR targets, same-scope comparisons, numerical verification, physics-fidelity
coverage, detector response, and propagated uncertainty for the claimed device
and observables.

The plan below is intentionally ordered. Each step should leave the tool more
truthful even if later steps are incomplete.

## Detailed Plan

0. KR Corpus Inventory And Exhaustive Review Control

   - Inventory every local `KnowledgeReference/` file.
   - Track DPF-named markdown files separately from the full broader plasma
     corpus.
   - Count a file as review-closed only when it contributes to a coded,
     line-referenced KR validation target or is explicitly rejected as
     non-extractable with reason.
   - Done when the unreviewed DPF-named source list is empty and target
     extraction or rejection status is auditable.

1. KR Target Authority Manifest

   - Enumerate every KR-backed validation target currently embedded in code.
   - Audit each target for local source file existence and in-bounds line ranges.
   - Expose a machine-readable manifest for app/API/report layers.
   - Done when every target used by validation helpers can be listed and source
     audited from local files.

2. KR Semantic Target Extraction

   - Expand target records from bibliographic citations into typed observable
     targets: device, shot, bank, fill, geometry, waveform, phase timing,
     density, magnetic/EM, temperature, neutron, detector, and uncertainty.
   - Add tests that assert source path, source lines, units, and validation
     scope for each extracted target.
   - Done when each predictive tier has at least one same-scope KR target packet
     with typed observables.

3. Tier 2 Phase Validation

   - Extract full axial, radial, and pinch timing targets for at least one
     KR-backed device/shot.
   - Promote ordinary production runs only when simulated phase history matches
     the same-scope target within explicit tolerance and uncertainty.
   - Done when tier 2 can pass from a real run without candidate-only fallbacks.

4. Tier 3 Production MHD Fidelity

   - Make production runs emit finite-volume method verification, cylindrical
     convergence, resistive diffusion convergence, circuit-coupled energy,
     backend parity, and MHD phase/scope-limit evidence.
   - Keep MLX/Metal runtime failures as blockers until resolved or bypassed by
     a validated non-MLX backend path.
   - Done when a production MHD result passes the numerical-fidelity audit.

5. Tier 4 Same-Scope Spatial Validation

   - Build one same-scope spatial validation packet with density, magnetic/EM,
     and temperature evidence from KR-backed diagnostics.
   - Reject cross-device or review-only component mixing.
   - Done when tier 4 can pass from a real same-scope target packet.

6. Tier 5 Neutron Validation

   - Generate or ingest mechanism-separated neutron timing, spectrum,
     anisotropy, detector/activation response, and yield uncertainty for one
     KR-backed scope.
   - Done when tier 5 and the neutron high-fidelity gap both pass from the same
     validation scope.

7. Physics-Fidelity Closure

   - Implement or explicitly bound EOS/conductivity, ionization,
     two-temperature physics, radiation transport/opacities, ablation/impurity
     mixing, Hall/FLR/kinetic/PIC effects, 3D instabilities, flashover/sheath
     initiation, restrike/anomalous resistance, and beam-target coupling.
   - Done when every required effect is validated or bounded out for the
     claimed observable scope.

8. Uncertainty Propagation

   - Propagate experimental, input, numerical, model-form, shot-to-shot, and
     acceptance-rule uncertainties into every supported validation tier.
   - Done when high-fidelity readiness is blocked by no UQ component.

9. End-to-End Predictive Demonstration

   - Run a complete same-device/same-scope DPF case through all gates.
   - Export reproducibility, source authority, predictive readiness,
     high-fidelity readiness, and scientific-accuracy gaps.
   - Done when an ordinary production run, not a synthetic packet, reports
     `high_fidelity_ready`.

## Current Execution Position

Step 0 and Step 2 are in progress. The corpus has now been counted and the
project can report that the full local source of truth has not been completely
review-closed. Current inventory: 827 total files, 398 markdown files, 396 JSON
files, and 54 DPF-named markdown files. Current coded targets cover 26 target
records from 22 unique KR sources. Including explicit review decisions, 27 of
the 54 DPF-named markdown files count as review-closed: 21 by coded targets and
6 by explicit decisions. Target extraction remains open:
`circuit_waveform`, `phase_timing`, and `spatial_temperature` are still partial,
27 DPF-named markdown files remain unreviewed, and no same-scope target set
passes. The current best available same-scope report is MJOLNIR by audit rank,
but it lacks circuit waveform, phase semantics, phase timing, spatial density,
spatial magnetic/EM, and uncertainty groups and has partial temperature. The
PF-1000 full-energy scope remains the broadest end-to-end source packet by
present groups: every required observable group is present in that scope, but
current waveform, phase timing, neutron timing/spectrum/anisotropy/detector
response, magnetic/EM, temperature, and uncertainty remain partial.

## Execution Log

### 2026-05-05: KR Target Authority Manifest

- Added a machine-readable manifest for every coded KR validation target.
- Added a source audit for the manifest that checks local `KnowledgeReference/`
  files and line ranges through the existing source-authority helper.
- Exported the manifest and audit from `dpf.validation`.
- Verification:
  - `python3 -m py_compile src/dpf/validation/kr_targets.py src/dpf/validation/__init__.py tests/test_kr_targets.py`
  - `python3 -m pytest tests/test_kr_targets.py::test_kr_validation_target_manifest_lists_coded_targets tests/test_kr_targets.py::test_kr_validation_target_source_audit_passes_for_local_targets -q` passed (`2 passed in 0.89s`)
  - `python3 -m pytest tests/test_quality_assessment.py tests/test_kr_targets.py -q` passed (`86 passed in 0.50s`)
  - `python3 -m pytest tests/test_mhd_physics_integration.py tests/test_mhd_numerical_fidelity.py tests/test_circuit_field_coupling.py tests/test_physics_fidelity.py tests/test_uncertainty_budget.py -q` passed (`87 passed, 3 skipped in 1.29s`)
  - `git diff --check` passed.
- Remaining limit: this audit proves target source authority, not semantic
  completeness. The next step is typed KR target extraction hardening.

### 2026-05-05: Typed KR Target Coverage Report

- Added a target coverage report that maps coded KR targets to the end-to-end
  observable groups needed for predictive validation.
- The report intentionally fails today because the target set is incomplete:
  `circuit_waveform` is missing, and `phase_timing` is partial.
- Exported the coverage report from `dpf.validation`.
- Verification:
  - `python3 -m py_compile src/dpf/validation/kr_targets.py src/dpf/validation/__init__.py tests/test_kr_targets.py`
  - `python3 -m pytest tests/test_kr_targets.py::test_kr_validation_target_coverage_report_lists_remaining_groups tests/test_kr_targets.py::test_kr_validation_target_source_audit_passes_for_local_targets -q` passed (`2 passed in 0.80s`)
  - `python3 -m pytest tests/test_quality_assessment.py tests/test_kr_targets.py -q` passed (`87 passed in 0.55s`)
  - `python3 -m pytest tests/test_mhd_physics_integration.py tests/test_mhd_numerical_fidelity.py tests/test_circuit_field_coupling.py tests/test_physics_fidelity.py tests/test_uncertainty_budget.py -q` passed (`87 passed, 3 skipped in 1.36s`)
  - `git diff --check` passed.
- Remaining limit: coverage presence does not mean same-scope closure or
  simulation agreement. It only makes the target extraction backlog explicit.

### 2026-05-05: PF-1000 Partial Circuit Waveform Target

- Added a typed PF-1000 16 kV current-waveform target from the Akel 2021 KR
  source.
- The target records measured-current context, shot/fill/bank context,
  peak-current range, shot-12581 peak/pinch current, and the fact that the fit
  is valid only until the end of the current dip.
- The target is intentionally partial because digitized current trace points and
  per-point timing/current uncertainty are not extracted into the target packet.
- The coverage report now marks `circuit_waveform` as `partial` instead of
  `missing`.
- Verification:
  - `python3 -m py_compile src/dpf/validation/kr_targets.py src/dpf/validation/__init__.py tests/test_kr_targets.py`
  - `python3 -m pytest tests/test_kr_targets.py::test_pf1000_current_waveform_target_metadata tests/test_kr_targets.py::test_kr_validation_target_manifest_lists_coded_targets tests/test_kr_targets.py::test_kr_validation_target_coverage_report_lists_remaining_groups tests/test_kr_targets.py::test_kr_validation_target_source_audit_passes_for_local_targets -q` passed (`4 passed in 2.62s`)
  - `python3 -m pytest tests/test_quality_assessment.py tests/test_kr_targets.py -q` passed (`88 passed in 0.52s`)
  - `python3 -m pytest tests/test_mhd_physics_integration.py tests/test_mhd_numerical_fidelity.py tests/test_circuit_field_coupling.py tests/test_physics_fidelity.py tests/test_uncertainty_budget.py -q` passed (`87 passed, 3 skipped in 1.43s`)
  - `git diff --check` passed.
- Remaining limit: this is not yet a full waveform target because the actual
  digitized current series and per-point uncertainty are not extracted.

### 2026-05-05: Lee-Course Full Phase-Timing Example Target

- Added a typed Lee/RADPF course example target for NX2 neon phase timing.
- The target records numeric axial end, radial start/end, radial duration,
  pinch start/end, pinch duration, radial shock axis time, and reflected-shock
  piston timing.
- The target is marked example-only for predictive purposes because it is a
  fitted worksheet example, not a same-shot deuterium experimental target with
  uncertainty.
- Verification:
  - `python3 -m py_compile src/dpf/validation/kr_targets.py src/dpf/validation/__init__.py tests/test_kr_targets.py`
  - `python3 -m pytest tests/test_kr_targets.py::test_lee_course_nx2_phase_timing_example_metadata tests/test_kr_targets.py::test_kr_validation_target_coverage_report_lists_remaining_groups tests/test_kr_targets.py::test_kr_validation_target_source_audit_passes_for_local_targets -q` passed (`3 passed in 2.62s`)
  - `python3 -m pytest tests/test_quality_assessment.py tests/test_kr_targets.py -q` passed (`89 passed in 0.66s`)
  - `python3 -m pytest tests/test_mhd_physics_integration.py tests/test_mhd_numerical_fidelity.py tests/test_circuit_field_coupling.py tests/test_physics_fidelity.py tests/test_uncertainty_budget.py -q` passed (`87 passed, 3 skipped in 1.44s`)
  - `git diff --check` passed.
- Remaining limit: full predictive tier 2 still needs same-device/same-shot
  deuterium phase timing targets with experimental uncertainty.

### 2026-05-05: App Exports KR Target Source And Coverage Reports

- App post-processing now exports `kr_validation_target_source_audit` and
  `kr_validation_target_coverage` beside predictive and high-fidelity readiness.
- The target source audit passes for the coded local targets.
- The target coverage report intentionally fails because target extraction is
  still incomplete, including partial `phase_timing`.
- Verification:
  - `python3 -m py_compile app_mhd.py tests/test_mhd_physics_integration.py`
  - `python3 -m pytest tests/test_mhd_physics_integration.py::test_predictive_readiness_exported_and_blocks_unvalidated_claims tests/test_kr_targets.py::test_kr_validation_target_coverage_report_lists_remaining_groups tests/test_kr_targets.py::test_kr_validation_target_source_audit_passes_for_local_targets -q` passed (`3 passed in 0.73s`)
  - `python3 -m pytest tests/test_quality_assessment.py tests/test_kr_targets.py -q` passed (`89 passed in 0.55s`)
  - `python3 -m pytest tests/test_mhd_physics_integration.py tests/test_mhd_numerical_fidelity.py tests/test_circuit_field_coupling.py tests/test_physics_fidelity.py tests/test_uncertainty_budget.py -q` passed (`87 passed, 3 skipped in 1.24s`)
  - `git diff --check` passed.
- Remaining limit: app export does not close target extraction. It makes the
  missing target groups visible on ordinary result payloads.

### 2026-05-05: KR Target Semantic Source-Window Audit

- Added `kr_validation_target_semantic_audit()` to check that every coded KR
  target's cited line windows contain expected domain markers for the extracted
  observable.
- Exported the semantic audit from `dpf.validation` and app post-processing, so
  ordinary result payloads now include source-file validity, semantic
  source-window plausibility, and target coverage.
- Adjusted the PF-1000 Malir density marker to match the cited-window language:
  the target line windows use `interferometer` / `interferometric` diagnostic
  language rather than the title-form word `interferometry`.
- Verification:
  - `python3 -m py_compile app_mhd.py src/dpf/validation/kr_targets.py src/dpf/validation/__init__.py tests/test_kr_targets.py tests/test_mhd_physics_integration.py`
  - `python3 -m pytest tests/test_mhd_physics_integration.py::test_predictive_readiness_exported_and_blocks_unvalidated_claims tests/test_kr_targets.py::test_kr_validation_target_semantic_audit_passes_for_coded_targets -q` passed (`2 passed in 1.19s`)
  - `python3 -m pytest tests/test_quality_assessment.py tests/test_kr_targets.py -q` passed (`90 passed in 0.50s`)
  - `python3 -m pytest tests/test_mhd_physics_integration.py tests/test_mhd_numerical_fidelity.py tests/test_circuit_field_coupling.py tests/test_physics_fidelity.py tests/test_uncertainty_budget.py -q` passed (`87 passed, 3 skipped in 1.68s`)
  - `git diff --check` passed.
- Remaining limit: this is a lightweight semantic guard. It does not replace
  human review of extracted values, same-shot completeness, digitized waveform
  data, or uncertainty-bearing simulation-to-experiment validation.

### 2026-05-05: KR Target Coverage Becomes A High-Fidelity Gap

- Added a `kr_target_coverage` area to `scientific_accuracy_gap_report()`.
- High-fidelity readiness now requires both a passing KR target coverage report
  and a passing KR semantic source-window audit.
- The default app result still reports this gap as `partial` because the coded
  target set has partial `circuit_waveform`, `phase_timing`, and
  `spatial_temperature` coverage.
- Verification:
  - `python3 -m py_compile src/dpf/validation/quality_assessment.py tests/test_quality_assessment.py tests/test_mhd_physics_integration.py`
  - `python3 -m pytest tests/test_quality_assessment.py::TestQualityAssessment::test_scientific_accuracy_gap_report_lists_remaining_work tests/test_quality_assessment.py::TestQualityAssessment::test_high_fidelity_readiness_can_pass_with_complete_evidence_packet tests/test_mhd_physics_integration.py::test_predictive_readiness_exported_and_blocks_unvalidated_claims -q` passed (`3 passed in 0.73s`)
  - `python3 -m pytest tests/test_quality_assessment.py tests/test_kr_targets.py -q` passed (`90 passed in 0.46s`)
  - `python3 -m pytest tests/test_mhd_physics_integration.py tests/test_mhd_numerical_fidelity.py tests/test_circuit_field_coupling.py tests/test_physics_fidelity.py tests/test_uncertainty_budget.py -q` passed (`87 passed, 3 skipped in 1.25s`)
  - `git diff --check` passed.
- Remaining limit: this closes a readiness-gate loophole. It does not yet
  supply the missing digitized waveform, same-shot phase timing, or same-device
  spatial temperature targets.

### 2026-05-05: Same-Scope KR Target Coverage Audit

- Added `kr_validation_same_scope_target_report()` to distinguish cross-device
  target availability from one compatible validation scope.
- App post-processing now exports `kr_validation_same_scope_targets`.
- High-fidelity readiness now requires target coverage, semantic source-window
  audit, and same-scope target coverage to pass.
- The best available scope is currently MJOLNIR neutron timing/detector response,
  but it is still missing circuit waveform, phase timing, spatial density,
  spatial magnetic/EM, spatial temperature, and uncertainty target groups.
- Verification:
  - `python3 -m py_compile app_mhd.py src/dpf/validation/kr_targets.py src/dpf/validation/__init__.py src/dpf/validation/quality_assessment.py tests/test_kr_targets.py tests/test_quality_assessment.py tests/test_mhd_physics_integration.py`
  - `python3 -m pytest tests/test_kr_targets.py::test_kr_validation_same_scope_target_report_requires_one_scope tests/test_quality_assessment.py::TestQualityAssessment::test_high_fidelity_readiness_can_pass_with_complete_evidence_packet tests/test_mhd_physics_integration.py::test_predictive_readiness_exported_and_blocks_unvalidated_claims -q` passed (`3 passed in 3.03s`)
  - `python3 -m pytest tests/test_quality_assessment.py tests/test_kr_targets.py -q` passed (`91 passed in 0.49s`)
  - `python3 -m pytest tests/test_mhd_physics_integration.py tests/test_mhd_numerical_fidelity.py tests/test_circuit_field_coupling.py tests/test_physics_fidelity.py tests/test_uncertainty_budget.py -q` passed (`87 passed, 3 skipped in 1.32s`)
  - `git diff --check` passed.
- Remaining limit: this audit tightens the definition of an end-to-end target
  set. It does not create the missing same-scope experimental waveform, phase,
  spatial, or uncertainty targets.

### 2026-05-05: MJOLNIR Stagnation Temperature Target Context

- Added `mjolnir_stagnation_temperature_targets()` from the MJOLNIR KR paper.
- The target is tied to `mjolnir_neutron_timing_2025_goyon`, so same-scope
  audits now see MJOLNIR neutron timing/detector response plus partial
  spatial-temperature context.
- The target records the KR stagnation-temperature scaling reference, the
  `(Te + Ti) / 2` definition, the several-keV context, and explicit missing
  items for full tier 4: direct experimental temperature diagnostic,
  experimental uncertainty, and same-scope density/magnetic-field targets.
- Verification:
  - `python3 -m py_compile src/dpf/validation/kr_targets.py src/dpf/validation/__init__.py tests/test_kr_targets.py`
  - `python3 -m pytest tests/test_kr_targets.py::test_mjolnir_stagnation_temperature_target_is_partial_context tests/test_kr_targets.py::test_kr_validation_target_semantic_audit_passes_for_coded_targets tests/test_kr_targets.py::test_kr_validation_target_coverage_report_lists_remaining_groups tests/test_kr_targets.py::test_kr_validation_same_scope_target_report_requires_one_scope -q` passed (`4 passed in 0.50s`)
  - `python3 -m pytest tests/test_quality_assessment.py tests/test_kr_targets.py -q` passed (`92 passed in 0.58s`)
  - `python3 -m pytest tests/test_mhd_physics_integration.py tests/test_mhd_numerical_fidelity.py tests/test_circuit_field_coupling.py tests/test_physics_fidelity.py tests/test_uncertainty_budget.py -q` passed (`87 passed, 3 skipped in 1.46s`)
  - `git diff --check` passed.
- Remaining limit: this is temperature context from shock-theory/MHD-kinetic
  analysis, not a direct experimental temperature diagnostic. It cannot close
  tier 4 by itself.

### 2026-05-05: Corpus Review Status Saved And Audited

- Saved the explicit status that the complete `KnowledgeReference/` corpus has
  not yet been line-by-line review-closed.
- Added `kr_corpus_inventory()` and `kr_corpus_review_status()`.
- Current local inventory:
  - total files: 827
  - markdown files: 398
  - JSON files: 396
  - DPF-named markdown files: 54
- Current review-closed status under the coded-target rule at initial creation:
  - coded KR target records: 11
  - unique coded KR target source files: 7
  - DPF-named markdown files represented by coded targets: 6 of 54
  - unreviewed DPF-named markdown files: 48
- App post-processing now exports `kr_corpus_review_status`.
- Verification:
  - `python3 -m py_compile src/dpf/validation/kr_corpus.py src/dpf/validation/__init__.py app_mhd.py tests/test_kr_corpus.py tests/test_mhd_physics_integration.py`
  - `python3 -m pytest tests/test_kr_corpus.py tests/test_mhd_physics_integration.py::test_predictive_readiness_exported_and_blocks_unvalidated_claims -q` passed (`3 passed in 1.14s`)
- Remaining limit: this is an inventory and progress audit. It does not itself
  extract the remaining data. The next extraction ratchet is to review the
  unreviewed DPF-named markdown files for waveform, phase, spatial, neutron,
  and uncertainty targets or mark them explicitly non-extractable.

### 2026-05-05: Unreviewed DPF Source Triage Queue

- Added `kr_unreviewed_dpf_source_triage()` to rank the 48 unreviewed DPF-named
  markdown files by observable keyword categories.
- Triage category counts among the 48 unreviewed files:
  - circuit waveform candidates: 30
  - phase timing candidates: 31
  - spatial density candidates: 17
  - spatial magnetic/EM candidates: 33
  - spatial temperature candidates: 42
  - neutron validation candidates: 42
  - uncertainty candidates: 18
- Current highest-priority sources by category breadth:
  - `KnowledgeReference/focus-fusion-overview-of-progress-towards-p-b11-fusion-with-the-dense-plasma-focus.md`
  - `KnowledgeReference/gribkov-2007-pf1000-jphysd-part2.md`
  - `KnowledgeReference/regular-article-deuterium-argon-admixture-for-plasma-focus-neutron-generation-muhammad-luqman.md`
  - `KnowledgeReference/scholz-2007-pf1000-part2-jphysd.md`
  - `KnowledgeReference/characterising-the-plasma-focus-pinch-and-speed-enhancing-the-neutron-yield.md`
- Verification:
  - `python3 -m py_compile src/dpf/validation/kr_corpus.py src/dpf/validation/__init__.py tests/test_kr_corpus.py`
  - `python3 -m pytest tests/test_kr_corpus.py -q` passed (`3 passed in 0.53s`)
- Remaining limit: keyword triage is not extraction. Each candidate still needs
  line-by-line review, typed targets, source lines, units, and either an
  extracted validation target or an explicit non-extractable reason.

### 2026-05-05: PF-1000 Full-Energy Target Bundle From 2007 Papers

- Reviewed and extracted two high-priority PF-1000 full-energy papers from the
  local source of truth:
  - `KnowledgeReference/gribkov-2007-pf1000-jphysd-part2.md`
  - `KnowledgeReference/scholz-2007-pf1000-part2-jphysd.md`
- Added `pf1000_full_energy_phase_context_targets()` for paper I. It records
  phase semantics, the 2-4 Torr / up to 850 kJ / 2.5-3 MA operating regime,
  maximum compression about 100 ns before the current dip, maximum compression
  about 2 us after current maximum, about 150 ns confinement/neutron-pulse
  timing, and the missing digitized endpoints needed for full tier 2.
- Added `pf1000_full_energy_neutron_spatial_targets()` for paper II. It records
  810 kJ operation, shot 3121 at 465 Pa and 35 kV, typical total current
  2.5-2.6 MA, best current near 3 MA, estimated average pinch current about
  2 MA, neutron anisotropy ratios, 5e10-2e11 n/shot yield range with 6e11
  maximum, 7 m TOF correction, 2.45 MeV first-pulse context, density and
  magnetic-field estimates, temperature estimates, and detector/temperature
  limitations.
- The two targets share validation scope
  `pf1000_full_energy_2007_gribkov_scholz`.
- Current corpus status after this ratchet:
  - coded KR target records: 13 at this point in the sequence
  - unique coded KR target source files: 9
  - DPF-named markdown files represented by coded targets: 8 of 54
  - unreviewed DPF-named markdown files: 46
- Current target status remains intentionally blocked:
  - target coverage does not pass: `circuit_waveform`, `phase_timing`, and
    `spatial_temperature` remain partial.
  - same-scope coverage does not pass.
  - PF-1000 full-energy scope now has every required group present, including
    neutron detector response, but current waveform, phase timing, neutron
    timing/spectrum/anisotropy/detector response, spatial magnetic/EM,
    spatial temperature, and uncertainty are still partial.
- Updated triage counts after removing the two newly coded sources:
  - circuit waveform candidates: 28
  - phase timing candidates: 29
  - spatial density candidates: 15
  - spatial magnetic/EM candidates: 31
  - spatial temperature candidates: 40
  - neutron validation candidates: 40
  - uncertainty candidates: 16
- Verification:
  - `python3 -m py_compile src/dpf/validation/kr_targets.py src/dpf/validation/__init__.py tests/test_kr_targets.py`
  - `python3 -m pytest tests/test_kr_targets.py -q` passed (`43 passed in 0.45s`)
- Remaining limit: this is a major target-authority improvement, not a
  validation closure. The PF-1000 full-energy source itself says ion
  temperatures were not directly measured, pinch current was not directly
  measured, neutron pulse tails are affected by scatter, and current/neutron
  traces still need digitization and uncertainty for quantitative validation.

### 2026-05-05: PF-1000 Same-Scope Detector-Response Context

- Extended the PF-1000 full-energy paper II target with activation-counter,
  indium/bubble-detector cross-check, AmBe calibration, scintillator-PM,
  time-of-flight, and room-scatter response requirements.
- The PF-1000 full-energy scope now has every required end-to-end target group
  present in one validation scope.
- The same scope still fails by design because detector response is partial:
  the KR lines identify calibration and TOF context plus scatter limitations,
  but do not provide a complete neutron-field transport, detector-response, or
  room-scatter model.
- Verification:
  - `python3 -m py_compile src/dpf/validation/kr_targets.py tests/test_kr_targets.py`
  - `python3 -m pytest tests/test_kr_targets.py -q` passed (`43 passed in 0.50s`)
- Remaining limit: group presence is not validation closure. PF-1000 still
  needs digitized current and neutron traces, quantitative detector response,
  direct/uncertainty-bearing plasma state diagnostics, and simulation outputs
  compared against those targets.

### 2026-05-05: Deuterium-Argon Admixture Neutron Target

- Reviewed
  `KnowledgeReference/regular-article-deuterium-argon-admixture-for-plasma-focus-neutron-generation-muhammad-luqman.md`.
- Added `deuterium_argon_admixture_neutron_targets()` for the 2.7 kJ
  Mather-type PF gas-mixture experiment.
- The target records 30 uF / 14 kV / 4 mbar operation, 10-70% argon mass
  mixtures, measured current/voltage waveform availability, Rogowski and
  voltage-probe calibration, Lee-model current fitting, focus-time shift from
  2.7 to 3.3 us, voltage-spike FWHM values, indium activation calibration,
  pure-D2 and 50% argon neutron yields with standard deviations, energy into
  pinch, computed pinch current, computed ion-temperature context, and
  shot-to-shot uncertainty.
- Current corpus status after this ratchet:
  - coded KR target records: 14
  - unique coded KR target source files: 10
  - DPF-named markdown files represented by coded targets: 9 of 54
  - unreviewed DPF-named markdown files: 45
- Updated triage counts after removing this source:
  - circuit waveform candidates: 27
  - phase timing candidates: 28
  - spatial density candidates: 14
  - spatial magnetic/EM candidates: 30
  - spatial temperature candidates: 39
  - neutron validation candidates: 39
  - uncertainty candidates: 15
- Verification:
  - `python3 -m py_compile src/dpf/validation/kr_targets.py src/dpf/validation/__init__.py tests/test_kr_targets.py`
  - `python3 -m pytest tests/test_kr_targets.py -q` passed (`44 passed in 0.85s`)
  - `python3 -m pytest tests/test_quality_assessment.py tests/test_kr_targets.py tests/test_kr_corpus.py -q` passed (`98 passed in 0.67s`)
  - `python3 -m pytest tests/test_mhd_physics_integration.py tests/test_mhd_numerical_fidelity.py tests/test_circuit_field_coupling.py tests/test_physics_fidelity.py tests/test_uncertainty_budget.py -q` passed (`87 passed, 3 skipped in 1.84s`)
  - `git diff --check` passed.
- Remaining limit: this source is useful for admixture-yield and activation
  validation, but not for full end-to-end closure. It is time-integrated for
  neutron yield, uses Lee-fitted computed temperature, lacks digitized
  waveform points, and does not provide spatial density/magnetic/temperature
  fields for simulation comparison.

### 2026-05-05: FF-1 Focus Fusion Plasmoid And p-B11 Context Target

- Reviewed
  `KnowledgeReference/focus-fusion-overview-of-progress-towards-p-b11-fusion-with-the-dense-plasma-focus.md`.
- Added `ff1_focus_fusion_plasmoid_targets()` for FF-1 / FF-2B plasmoid,
  neutron, ion-energy, density, impurity, and p-B11 context.
- The target records FF-1 device parameters, diagnostic suite, main and beam
  Rogowski context, ion-beam energy-transfer measurements, confined-ion
  energy by neutron TOF, isotropy support from bubble detectors, best 2016
  neutron yield, wall-plug efficiency, estimated density, n-tau-T product,
  beryllium impurity/deposition measurements, QMF/p-B11 magnetic-field
  constraints, and current oscillation/yield-plateau limitations.
- Current corpus status after this ratchet:
  - coded KR target records: 15
  - unique coded KR target source files: 11
  - DPF-named markdown files represented by coded targets: 10 of 54
  - unreviewed DPF-named markdown files: 44
- Updated triage counts after removing this source:
  - circuit waveform candidates: 26
  - phase timing candidates: 27
  - spatial density candidates: 13
  - spatial magnetic/EM candidates: 29
  - spatial temperature candidates: 38
  - neutron validation candidates: 38
  - uncertainty candidates: 14
- Verification:
  - `python3 -m py_compile src/dpf/validation/kr_targets.py src/dpf/validation/__init__.py tests/test_kr_targets.py`
  - `python3 -m pytest tests/test_kr_targets.py -q` passed (`45 passed in 0.80s`)
- Remaining limit: this target is explicitly not p-B11 net-energy validation.
  The source includes measured deuterium FF-1 values, but the p-B11 and QMF
  parts are constraints, projections, or reduced simulations. Full validation
  still needs digitized waveforms, detector response, shot-series uncertainty,
  and direct advanced-fuel measurements.

### 2026-05-05: Lee Drive-Parameter Speed-Enhancement Target

- Reviewed
  `KnowledgeReference/characterising-the-plasma-focus-pinch-and-speed-enhancing-the-neutron-yield.md`.
- Added `lee_drive_parameter_speed_enhancement_targets()` as a generic
  scaling/regime target, not a same-device validation packet.
- The target records Lee axial snowplow/radial slug phase semantics,
  deuterium and neon pinch radius/length/lifetime scaling with anode radius,
  the neutron-optimized drive parameter `Ip/a/sqrt(p_D2) = 89.0 +/- 7.7`
  kA/cm/sqrt(torr), typical axial and radial speeds, constant-speed `Y ~ I^4`
  scaling, speed-enhanced thermonuclear and beam-target scaling, and
  operational speed limits where focus quality deteriorates.
- Current corpus status after this ratchet:
  - coded KR target records: 16
  - unique coded KR target source files: 12
  - DPF-named markdown files represented by coded targets: 11 of 54
  - unreviewed DPF-named markdown files: 43
- Updated triage counts after removing this source:
  - circuit waveform candidates: 25
  - phase timing candidates: 26
  - spatial density candidates: 13
  - spatial magnetic/EM candidates: 28
  - spatial temperature candidates: 37
  - neutron validation candidates: 37
  - uncertainty candidates: 13
- Verification:
  - `python3 -m py_compile src/dpf/validation/kr_targets.py src/dpf/validation/__init__.py tests/test_kr_targets.py`
  - `python3 -m pytest tests/test_kr_targets.py -q` passed (`46 passed in 0.49s`)
- Remaining limit: this source supports regime checks and scaling law hygiene,
  but it is generic. It cannot close any same-shot validation gate without a
  device-specific waveform, pressure, geometry, phase timing, neutron history,
  and detector-response packet.

### 2026-05-05: PFZ-200 Hybrid X-Pinch Proton/Neutron Target

- Reviewed
  `KnowledgeReference/deuterium-hybrid-x-pinch-driven-by-small-dense-plasma-focus-2.md`.
- Added `pfz200_hybrid_xpinch_proton_neutron_targets()` for the 3 kJ PFZ-200
  DPF-driven deuterium hybrid X-pinch.
- The target records PFZ-200 current/geometry/gas context, Rogowski current
  diagnostics, silver activation and nTOF detector setup, schlieren and CR-39
  diagnostic details, neutron FWHM timing for 3 mm and 5 mm A-K gaps versus
  unmodified DPF operation, neutron-yield ranges, localized proton-source
  dimensions, proton spectrum/yield values, and anisotropy/shot-to-shot
  interpretation limits.
- Current corpus status after this ratchet:
  - coded KR target records: 17
  - unique coded KR target source files: 13
  - DPF-named markdown files represented by coded targets: 12 of 54
  - unreviewed DPF-named markdown files: 42
- Updated triage counts after removing this source:
  - circuit waveform candidates: 24
  - phase timing candidates: 26
  - spatial density candidates: 12
  - spatial magnetic/EM candidates: 27
  - spatial temperature candidates: 36
  - neutron validation candidates: 36
  - uncertainty candidates: 12
- Verification:
  - `python3 -m py_compile src/dpf/validation/kr_targets.py src/dpf/validation/__init__.py tests/test_kr_targets.py`
  - `python3 -m pytest tests/test_kr_targets.py -q` passed (`47 passed in 0.48s`)
- Remaining limit: this is a modified hybrid X-pinch load. It is useful for
  localized DD particle-source and detector-response validation, but it is not
  an ordinary DPF end-to-end target and does not provide density, magnetic, or
  temperature validation for a standard DPF pinch.

### 2026-05-05: LLNL Fully Kinetic DPF Benchmark And Duplicate Review Decisions

- Reviewed the three local copies of the Schmidt/Tang/Welch fully kinetic DPF
  paper:
  `KnowledgeReference/fully-kinetic-simulations-of-dense-plasma-focus-z-pinch-8.md`,
  `KnowledgeReference/fully-kinetic-simulations-of-dense-plasma-focus-z-pinch-9.md`,
  and
  `KnowledgeReference/fully-kinetic-simulations-of-dense-plasma-focus-z-pinch.md`.
- Added explicit corpus review decisions marking the base and `-9` files as
  duplicate local copies represented by the canonical `-8` coded target.
- Added `llnl_fully_kinetic_dpf_targets()` for the LLNL 180 kA, 1 torr
  deuterium fully kinetic benchmark.
- The target records the LSP implicit-PIC setup, 2D cylindrical geometry,
  322-by-151 grid, 5 cm anode, 1.5 cm cathode radius, 10 cm domain length,
  1 mm initial sheath, neutral and sheath density, 4 kV initial voltage drop,
  180 kA current, current dip/impedance context, lower-hybrid-frequency
  fluctuation context, hot-pinch temperatures, MeV-ion spectrum context, and
  fluid/hybrid/fully kinetic neutron-yield comparison.
- Current corpus status after this ratchet:
  - coded KR target records: 18
  - unique coded KR target source files: 14
  - DPF-named markdown files represented by coded targets: 13 of 54
  - DPF-named markdown files closed by explicit duplicate decisions: 2 of 54
  - total DPF-named markdown files review-closed: 15 of 54
  - unreviewed DPF-named markdown files: 39
- Updated triage counts after removing the three fully kinetic local copies:
  - circuit waveform candidates: 21
  - phase timing candidates: 23
  - spatial density candidates: 12
  - spatial magnetic/EM candidates: 24
  - spatial temperature candidates: 33
  - neutron validation candidates: 33
  - uncertainty candidates: 9
- Verification:
  - `python3 -m py_compile src/dpf/validation/kr_targets.py src/dpf/validation/kr_corpus.py src/dpf/validation/__init__.py tests/test_kr_targets.py tests/test_kr_corpus.py`
  - `python3 -m pytest tests/test_kr_targets.py tests/test_kr_corpus.py -q` passed (`52 passed in 0.91s`)
- Remaining limit: this source is a simulation benchmark showing that fully
  kinetic physics is needed for MeV ions and approximate neutron yield in the
  LLNL low-current DPF. It is not a direct experimental data packet, does not
  provide detector response or shot-ensemble uncertainty, and cannot close
  same-scope predictive readiness by itself.

### 2026-05-05: NSTec/Gemini Fully 3D MHD Rundown Benchmark

- Reviewed
  `KnowledgeReference/fully-three-dimensional-simulation-and-modeling-of-a-dense-plasma-focus.md`.
- Added `nstec_3d_mhd_rundown_targets()` as a partial 3D-MHD current/rundown
  benchmark for the NSTec/Gemini DPF.
- The target records device geometry, bank/circuit context, Faraday rotator
  current diagnostic setup, 37-shot waveform repeatability at 37.5 kV and
  7.28 Torr, 2D/3D ALEGRA current comparisons, rundown-time comparisons, 3D
  cathode-bar flow/inductance context, density-floor and artificial hot-start
  limits, and the source's explicit statement that MHD becomes unphysical near
  Z-pinch without kinetic/PIC closure.
- Current corpus status after this ratchet:
  - coded KR target records: 19
  - unique coded KR target source files: 15
  - DPF-named markdown files represented by coded targets: 14 of 54
  - DPF-named markdown files closed by explicit duplicate decisions: 2 of 54
  - total DPF-named markdown files review-closed: 16 of 54
  - unreviewed DPF-named markdown files: 38
- Updated triage counts after removing this source:
  - circuit waveform candidates: 20
  - phase timing candidates: 22
  - spatial density candidates: 12
  - spatial magnetic/EM candidates: 23
  - spatial temperature candidates: 32
  - neutron validation candidates: 32
  - uncertainty candidates: 8
- Verification:
  - `python3 -m py_compile src/dpf/validation/kr_targets.py src/dpf/validation/__init__.py tests/test_kr_targets.py`
  - `python3 -m pytest tests/test_kr_targets.py tests/test_kr_corpus.py -q` passed (`53 passed in 0.56s`)
- Remaining limit: this target supports current/rundown benchmarking and 3D MHD
  scope control, not neutron-yield validation. It lacks digitized Faraday trace
  points, per-shot uncertainty, direct density/temperature/field diagnostics,
  detector response, and any validated late-pinch kinetic closure.

### 2026-05-05: MJOLNIR High/Low-Yield Parasitic-Current Target

- Reviewed `KnowledgeReference/goyon-2022-mjolnir-high-low.md`.
- Added `mjolnir_high_low_parasitic_current_targets()` as a partial
  MJOLNIR same-device mechanism target for variable yield, parasitic current
  paths, current dips, voltage spikes, conditioning, run-down/run-in velocity,
  pressure effects, and PIC/snowplow interpretation.
- The target records the 1-MJ and 2-MJ MJOLNIR pulsed-power configurations,
  highest reported neutron yields, Rogowski/voltage/photodiode/framing-camera
  diagnostics, CHICAGO/BERTHA/PIC setup, snow-plow alternate-current-path model,
  sheath phase sequence, current-dip and voltage-yield correlations, rBtheta
  parasitic-path interpretation, beam-energy mechanism, pressure degradation,
  and the remaining detector/trace/uncertainty gaps.
- Current corpus status after this ratchet:
  - coded KR target records: 20
  - unique coded KR target source files: 16
  - DPF-named markdown files represented by coded targets: 15 of 54
  - DPF-named markdown files closed by explicit duplicate decisions: 2 of 54
  - total DPF-named markdown files review-closed: 17 of 54
  - unreviewed DPF-named markdown files: 37
- Updated triage counts after removing this source:
  - circuit waveform candidates: 19
  - phase timing candidates: 21
  - spatial density candidates: 12
  - spatial magnetic/EM candidates: 22
  - spatial temperature candidates: 31
  - neutron validation candidates: 31
  - uncertainty candidates: 7
- Verification:
  - `python3 -m py_compile src/dpf/validation/kr_targets.py src/dpf/validation/__init__.py tests/test_kr_targets.py`
  - `python3 -m pytest tests/test_kr_targets.py tests/test_kr_corpus.py -q` passed (`54 passed in 0.53s`)
- Remaining limit: this source adds strong MJOLNIR mechanism constraints, but
  it still lacks digitized traces, shot-resolved uncertainty, activation
  detector response details, neutron timing/spectrum/anisotropy, and direct
  spatial density/temperature/field validation.

### 2026-05-05: PF-400J X-Ray Diagnostic Inference Target

- Reviewed
  `KnowledgeReference/inference-of-x-ray-emission-from-a-plasma-focus-discharge-comparison-between-characteristic.md`.
- Added `pf400j_xray_inference_targets()` as a PF-400J hydrogen x-ray
  diagnostic inference target, explicitly not a neutron validation packet.
- The target records PF-400J bank, geometry, fill, and discharge conditions;
  Rogowski, ILS, voltage-divider, Vivaldi, and scintillator-PMT diagnostics;
  x-ray detector response context; 959-shot campaign size; breakdown/pinch
  feature definitions; machine-learning feature-selection results; and the
  limits of using electrical/EM signals to infer x-ray emission.
- Current corpus status after this ratchet:
  - coded KR target records: 21
  - unique coded KR target source files: 17
  - DPF-named markdown files represented by coded targets: 16 of 54
  - DPF-named markdown files closed by explicit duplicate decisions: 2 of 54
  - total DPF-named markdown files review-closed: 18 of 54
  - unreviewed DPF-named markdown files: 36
- Updated triage counts after removing this source:
  - circuit waveform candidates: 18
  - phase timing candidates: 20
  - spatial density candidates: 12
  - spatial magnetic/EM candidates: 21
  - spatial temperature candidates: 30
  - neutron validation candidates: 30
  - uncertainty candidates: 6
- Verification:
  - `python3 -m py_compile src/dpf/validation/kr_targets.py src/dpf/validation/__init__.py tests/test_kr_targets.py`
  - `python3 -m pytest tests/test_kr_targets.py tests/test_kr_corpus.py -q` passed (`55 passed in 0.55s`)
- Remaining limit: this target supports x-ray diagnostic and feature-selection
  hygiene for a hundreds-of-joules hydrogen PF device. It does not provide
  deuterium neutron validation, same-scope spatial validation, absolute x-ray
  spectrum/response, or production solver closure.

### 2026-05-05: Reuben 2024 Thesis Review Decision

- Reviewed
  `KnowledgeReference/modification-and-numerical-modelling-of-dense-plasma-focus.md`.
- Added an explicit `insufficient_extractable_validation_data` corpus review
  decision instead of a coded target.
- Reason: the local markdown contains useful abstract, introduction, table, and
  figure-caption context for a 1 kJ / 1.3 uF / 40 kV modified DPF thesis, but
  the Experimental System, Numerical Modelling, Results and Discussion, and
  Conclusion sections are empty page stubs in this text extraction. Result
  values such as current waveform, radial trajectories, neutron production,
  pinch temperature, and scaling appear only as figure-list captions rather
  than source-line data suitable for validation targets.
- Current corpus status after this decision:
  - coded KR target records: 21
  - unique coded KR target source files: 17
  - DPF-named markdown files represented by coded targets: 16 of 54
  - DPF-named markdown files closed by explicit review decisions: 3 of 54
  - total DPF-named markdown files review-closed: 19 of 54
  - unreviewed DPF-named markdown files: 35
- Updated triage counts after removing this source:
  - circuit waveform candidates: 17
  - phase timing candidates: 19
  - spatial density candidates: 12
  - spatial magnetic/EM candidates: 20
  - spatial temperature candidates: 29
  - neutron validation candidates: 29
  - uncertainty candidates: 5
- Verification:
  - `python3 -m py_compile src/dpf/validation/kr_corpus.py tests/test_kr_corpus.py src/dpf/validation/__init__.py`
  - `python3 -m pytest tests/test_kr_corpus.py -q` passed (`4 passed in 0.59s`)
- Remaining limit: this file should be re-ingested from the original PDF if the
  thesis is needed for validation. The current markdown is not reliable enough
  for line-referenced current waveform, radial trajectory, or neutron-yield
  targets.

### 2026-05-05: Goyon 2025 Neutron-Generation Duplicate Decision

- Reviewed
  `KnowledgeReference/neutron-generation-dynamics-inside-a-ma-class-dense-plasma-focus-z-pinch.md`.
- Added an explicit duplicate review decision pointing to the canonical
  `KnowledgeReference/neutron-generation-dynamics-inside-a-ma-class-dense-plasma-focus-z-pinch-5.md`.
- Reason: the canonical `-5` source already backs the coded MJOLNIR neutron
  timing, stagnation-temperature, and neutron detector-response targets. This
  local copy is the same Phys. Plasmas 2025 Goyon MA-class MJOLNIR
  neutron-generation paper and should not produce duplicate target records.
- Current corpus status after this decision:
  - coded KR target records: 21
  - unique coded KR target source files: 17
  - DPF-named markdown files represented by coded targets: 16 of 54
  - DPF-named markdown files closed by explicit review decisions: 4 of 54
  - total DPF-named markdown files review-closed: 20 of 54
  - unreviewed DPF-named markdown files: 34
- Updated triage counts after removing this source:
  - circuit waveform candidates: 16
  - phase timing candidates: 18
  - spatial density candidates: 12
  - spatial magnetic/EM candidates: 19
  - spatial temperature candidates: 28
  - neutron validation candidates: 28
  - uncertainty candidates: 4
- Verification:
  - `python3 -m py_compile src/dpf/validation/kr_corpus.py tests/test_kr_corpus.py`
  - `python3 -m pytest tests/test_kr_corpus.py -q` passed (`4 passed in 0.55s`)
- Remaining limit: duplicate closure avoids double-counting; it does not add
  new validation coverage beyond the existing MJOLNIR coded targets.

### 2026-05-05: Rawat 2015 Generic DPF Operating-Envelope Target

- Reviewed the duplicate Rawat 2015 review pair:
  `KnowledgeReference/paper-open-access-dense-plasma-focus-from-alternative-fusion-source-to-versatile-high-energy-4.md`
  and
  `KnowledgeReference/paper-open-access-dense-plasma-focus-from-alternative-fusion-source-to-versatile-high-energy.md`.
- Added `rawat_dpf_operating_envelope_targets()` from the canonical `-4`
  source and added an explicit duplicate decision for the header/PDF-name
  variant without the `-4` suffix.
- Encoded the source as a generic DPF operating-envelope target, not as a
  same-device benchmark. Extracted constraints include 100-500 ns current
  sheath formation, 500-3000 ns quarter period, 2-10 cm/us optimized axial
  sheath speed, radial speed 2-2.5 times axial speed, pinch density
  `5e24-1e26 m^-3`, DPF energy density `1.2e10-9.5e10 J/m^3`, pinch
  temperatures `0.2-2 keV`, ion temperatures `0.3-1.5 keV`, 10-30 kV typical
  charge voltage, efficient operation at a few mbar, and explicit shot-to-shot
  repeatability/conditioning limits.
- Current corpus status after this ratchet:
  - coded KR target records: 22
  - unique coded KR target source files: 18
  - DPF-named markdown files represented by coded targets: 17 of 54
  - DPF-named markdown files closed by explicit review decisions: 5 of 54
  - total DPF-named markdown files review-closed: 22 of 54
  - unreviewed DPF-named markdown files: 32
- Updated triage counts after removing this pair:
  - circuit waveform candidates: 16
  - phase timing candidates: 16
  - spatial density candidates: 10
  - spatial magnetic/EM candidates: 17
  - spatial temperature candidates: 26
  - neutron validation candidates: 26
  - uncertainty candidates: 2
- Verification:
  - `python3 -m py_compile src/dpf/validation/kr_targets.py src/dpf/validation/kr_corpus.py src/dpf/validation/__init__.py tests/test_kr_targets.py tests/test_kr_corpus.py`
  - `python3 -m pytest tests/test_kr_targets.py tests/test_kr_corpus.py -q`
    passed (`56 passed in 0.63s`)
  - Broad post-ratchet verification passed:
    `python3 -m pytest tests/test_quality_assessment.py tests/test_kr_targets.py tests/test_kr_corpus.py -q`
    (`107 passed in 0.63s`);
    `python3 -m pytest tests/test_mhd_physics_integration.py tests/test_mhd_numerical_fidelity.py tests/test_circuit_field_coupling.py tests/test_physics_fidelity.py tests/test_uncertainty_budget.py -q`
    (`87 passed, 3 skipped in 1.54s`); `git diff --check` clean.
- Remaining limit: this is a review-derived sanity envelope. It can catch
  simulations outside basic DPF scale, but it cannot close predictive
  validation without same-scope measured current, phase, spatial, neutron, and
  uncertainty data.

### 2026-05-05: Petrov/LLNL 2022 MJOLNIR Duplicate Decision

- Reviewed `KnowledgeReference/petrov-2022-mjolnir-high-low-discharges.md`.
- Added an explicit duplicate review decision pointing to
  `KnowledgeReference/goyon-2022-mjolnir-high-low.md`.
- Reason: the Petrov/LLNL report extraction is the same Schmidt/Goyon 2022
  MJOLNIR high/low-performing discharge paper already represented by the coded
  `mjolnir_high_low_parasitic_current_2022_goyon` target. Differences are
  header, page-stamp, and line-wrap extraction differences, not separate
  validation evidence.
- Current corpus status after this decision:
  - coded KR target records: 22
  - unique coded KR target source files: 18
  - DPF-named markdown files represented by coded targets: 17 of 54
  - DPF-named markdown files closed by explicit review decisions: 6 of 54
  - total DPF-named markdown files review-closed: 23 of 54
  - unreviewed DPF-named markdown files: 31
- Updated triage counts after removing this source:
  - circuit waveform candidates: 15
  - phase timing candidates: 15
  - spatial density candidates: 10
  - spatial magnetic/EM candidates: 16
  - spatial temperature candidates: 25
  - neutron validation candidates: 25
  - uncertainty candidates: 1
- Verification:
  - `python3 -m py_compile src/dpf/validation/kr_corpus.py tests/test_kr_corpus.py`
  - `python3 -m pytest tests/test_kr_corpus.py -q` passed (`4 passed in 0.51s`)
  - Broad post-decision verification passed:
    `python3 -m pytest tests/test_quality_assessment.py tests/test_kr_targets.py tests/test_kr_corpus.py -q`
    (`107 passed in 0.59s`);
    `python3 -m pytest tests/test_mhd_physics_integration.py tests/test_mhd_numerical_fidelity.py tests/test_circuit_field_coupling.py tests/test_physics_fidelity.py tests/test_uncertainty_budget.py -q`
    (`87 passed, 3 skipped in 1.49s`); `git diff --check` clean.
- Remaining limit: duplicate closure avoids double-counting the same MJOLNIR
  parasitic-current evidence. It does not add new same-scope coverage.

### 2026-05-05: Auluck 2023 Generalized Plasma Focus Scaling Target

- Reviewed
  `KnowledgeReference/the-generalized-plasma-focus-problem-and-its-application-to-space-propulsion-s-k-h-auluck.md`.
- Added `auluck_gpf_scaling_theory_targets()` as a KR-backed theory/scaling
  target, not as an experimental validation pass.
- Encoded the paper's key scientific warning: conventional DPF fusion energy
  output involves a complex, not fully understood process; neutron-yield scaling
  failure is experimentally observed; no theoretical understanding or empirical
  workaround exists for conventional DPF propulsion claims in this source.
- Encoded model-scope constraints: Lee and RGV-type reduced models require
  experimental current waveform fitting and compensate for neglected formation,
  propagation-delay, and sheath-geometry physics. The GPF treatment explicitly
  addresses lift-off/propagation delay and treats the moving sheath as a
  power-density-amplifying plasma flow switch.
- Extracted laboratory example values include 20 kV charge voltage, `43 uF`
  capacitance, `160 kA` current scale, `8.6 kJ` stored energy, `8.45 us`
  quarter period, hydrogen density `0.00342 kg/m^3` or about `43 mbar`,
  example power-density amplification about `9000`, magnetic field rising from
  `20 T` to about `200 T` in about `40 ns`, wire current about `80 kA`, current
  density `1.8e12 A/m^2`, radial Alfven transit time about `17 ns`, wire travel
  time about `8.4 ns`, explosion timescale about `3 ps`, jet Alfven velocity
  about `1450 m/s`, and impulse about `0.002 kg m/s`.
- Encoded validation requirements from the source: measure plasma voltage and
  current, compute and compare inductance variation, repeat across profile
  parameters, measure jet momentum and velocity, verify energy deposition in
  dynamic-hohlraum variants, validate gas-distribution/breakdown strategy, and
  separately test deuterium-filled tube neutron emission.
- Current corpus status after this ratchet:
  - coded KR target records: 23
  - unique coded KR target source files: 19
  - DPF-named markdown files represented by coded targets: 18 of 54
  - DPF-named markdown files closed by explicit review decisions: 6 of 54
  - total DPF-named markdown files review-closed: 24 of 54
  - unreviewed DPF-named markdown files: 30
- Updated triage counts after removing this source:
  - circuit waveform candidates: 14
  - phase timing candidates: 14
  - spatial density candidates: 9
  - spatial magnetic/EM candidates: 15
  - spatial temperature candidates: 24
  - neutron validation candidates: 24
  - uncertainty candidates: 1
- Verification:
  - `python3 -m py_compile src/dpf/validation/kr_targets.py src/dpf/validation/__init__.py tests/test_kr_targets.py`
  - `python3 -m pytest tests/test_kr_targets.py tests/test_kr_corpus.py -q`
    passed (`57 passed in 0.60s`)
  - Broad post-ratchet verification passed:
    `python3 -m pytest tests/test_quality_assessment.py tests/test_kr_targets.py tests/test_kr_corpus.py -q`
    (`108 passed in 0.64s`);
    `python3 -m pytest tests/test_mhd_physics_integration.py tests/test_mhd_numerical_fidelity.py tests/test_circuit_field_coupling.py tests/test_physics_fidelity.py tests/test_uncertainty_budget.py -q`
    (`87 passed, 3 skipped in 1.54s`); `git diff --check` clean.
- Remaining limit: this source is important for scope control and scaling
  requirements, but it is not a same-shot DPF benchmark. It does not provide
  measured waveforms, phase endpoints, spatial profiles, neutron data, or
  uncertainty for a completed predictive validation case.

### 2026-05-05: Sandia 2009 ALEGRA-HEDP DPF MHD Target

- Reviewed
  `KnowledgeReference/unlimited-release-printed-september-2009-alegra-hedp-simulations-of-the-dense-plasma-focus.md`.
- Added `alegra_hedp_dpf_mhd_validation_targets()` as a partial MHD/circuit
  benchmark and scope-limit target.
- Encoded the source's central scientific limit: 2D ALEGRA-HEDP can reproduce
  early DPF current, timing, sheath speed, density, and temperature behavior for
  Bernard-class devices, but MHD only predicts the thermonuclear neutron
  component and must stop when charge separation and instabilities make the MHD
  approximation invalid.
- Extracted benchmark values include Bernard Long `135 uF`, `20 kV`, `27 kJ`,
  `27 nH` estimated stray inductance, `3.3 mOhm` estimated resistance, `3 Torr`,
  experiment/simulation peak current `0.6 MA`/`0.5-0.6 MA`, and neutron yield
  `1.5e9` experiment vs `1.2e5` ALEGRA thermonuclear. Bernard Short includes
  `120 uF`, `40 kV`, `96 kJ`, `10 Torr`, peak current `1.5 MA` experiment and
  ALEGRA, and neutron yield `3e10` experiment vs `1.5e6` ALEGRA. Tallboy
  includes `216 uF`, `50 kV`, `270 kJ`, `50 nH`, peak current `2.3 MA`
  experiment vs `1.8 MA` ALEGRA, and neutron yield `3.5e11` experiment vs
  `3.7e7` ALEGRA.
- Encoded spatial/temperature context: generic pinch width about `1 mm`, length
  of a few mm, density `1e19-1e20 cm^-3`, Bernard Long measured pinch density
  `1e18-5e19 cm^-3`, simulated density `1.4e19 cm^-3`, experimental pre-pinch
  ion temperature `300 eV`, simulated pre-pinch ion/electron temperatures
  `250-650 eV` and `200-360 eV`, experimental pinch ion temperature about
  `700 eV`, and unresolved simulated pinch ion temperature `9 keV`.
- Encoded numerical limits: Sesame EOS density floor `0.01 kg/m^3` is
  inconsistent for initial DPF gas, QEOS deuterium was used, approximate cell
  size was `0.5 mm`, the ionized seed layer was arbitrary at `1 eV`, cathode
  bars require 3D modeling, and PIC-to-MHD sheath import is needed.
- Current corpus status after this ratchet:
  - coded KR target records: 24
  - unique coded KR target source files: 20
  - DPF-named markdown files represented by coded targets: 19 of 54
  - DPF-named markdown files closed by explicit review decisions: 6 of 54
  - total DPF-named markdown files review-closed: 25 of 54
  - unreviewed DPF-named markdown files: 29
- Updated triage counts after removing this source:
  - circuit waveform candidates: 13
  - phase timing candidates: 13
  - spatial density candidates: 8
  - spatial magnetic/EM candidates: 14
  - spatial temperature candidates: 23
  - neutron validation candidates: 23
  - uncertainty candidates: 1
- Verification:
  - `python3 -m py_compile src/dpf/validation/kr_targets.py src/dpf/validation/__init__.py tests/test_kr_targets.py`
  - `python3 -m pytest tests/test_kr_targets.py tests/test_kr_corpus.py -q`
    passed (`58 passed in 0.62s`)
  - Broad post-ratchet verification passed:
    `python3 -m pytest tests/test_quality_assessment.py tests/test_kr_targets.py tests/test_kr_corpus.py -q`
    (`109 passed in 0.60s`);
    `python3 -m pytest tests/test_mhd_physics_integration.py tests/test_mhd_numerical_fidelity.py tests/test_circuit_field_coupling.py tests/test_physics_fidelity.py tests/test_uncertainty_budget.py -q`
    (`87 passed, 3 skipped in 1.51s`); `git diff --check` clean.
- Remaining limit: this target supports early MHD/circuit validation only. It
  does not validate total neutron yield, beam-target production, neutron timing,
  neutron spectrum, neutron anisotropy, detector response, or post-MHD kinetic
  pinch evolution.

### 2026-05-05: Auluck 2021 Circuit-Element/Poynting Target

- Reviewed `KnowledgeReference/auluck-2021-dpf-circuit-element.md`.
- Added `auluck_circuit_element_poynting_targets()` as a circuit-field
  coupling target.
- Encoded the paper's core constraint: representing DPF post-stagnation behavior
  as a scalar time-varying inductance is incomplete. The terminal voltage must
  account for the volume-integrated field power, and the difference between the
  Poynting-theorem term and the motional impedance implied by a time-varying
  inductance appears as anomalous impedance.
- Extracted diagnostic context includes standard `dI/dt` and voltage
  diagnostics, current derivative dip and voltage spike as proper-operation
  indicators, their correlation with neutron yield, and the note that voltage
  spike and current derivative minimum are time-correlated but not simultaneous.
- Extracted PF-1000 context includes magnetic probe radii `40`, `13`, and
  `0 mm`, probe height `10 mm` above the anode, interferogram intervals
  `10-15 ns`, current-carrying layer thickness `1.6-2.6 cm`, sheath velocity
  about `2.1e5 m/s` with `25%` shot-to-shot variation, density fall by at least
  two orders of magnitude within less than `1 mm`, illustrative probe times
  `-68`, `-38`, and `22 ns`, and a `10-20 ns` diagnostic propagation delay over
  about `2 m`.
- Encoded field-coupling requirements: 3D magnetic and velocity structures,
  motional dynamo amplification of seed fields, poloidal magnetic fields, all
  three magnetic-field components contributing to plasma inductance, and
  quasi-closed post-breakup current streamlines that still draw energy from the
  external circuit.
- Current corpus status after this ratchet:
  - coded KR target records: 25
  - unique coded KR target source files: 21
  - DPF-named markdown files represented by coded targets: 20 of 54
  - DPF-named markdown files closed by explicit review decisions: 6 of 54
  - total DPF-named markdown files review-closed: 26 of 54
  - unreviewed DPF-named markdown files: 28
- Updated triage counts after removing this source:
  - circuit waveform candidates: 13
  - phase timing candidates: 12
  - spatial density candidates: 7
  - spatial magnetic/EM candidates: 13
  - spatial temperature candidates: 23
  - neutron validation candidates: 22
  - uncertainty candidates: 0
- Verification:
  - `python3 -m py_compile src/dpf/validation/kr_targets.py src/dpf/validation/__init__.py tests/test_kr_targets.py`
  - `python3 -m pytest tests/test_kr_targets.py tests/test_kr_corpus.py -q`
    passed (`59 passed in 0.56s`)
  - Broad post-ratchet verification passed:
    `python3 -m pytest tests/test_quality_assessment.py tests/test_kr_targets.py tests/test_kr_corpus.py -q`
    (`110 passed in 0.64s`);
    `python3 -m pytest tests/test_mhd_physics_integration.py tests/test_mhd_numerical_fidelity.py tests/test_circuit_field_coupling.py tests/test_physics_fidelity.py tests/test_uncertainty_budget.py -q`
    (`87 passed, 3 skipped in 1.53s`); `git diff --check` clean.
- Remaining limit: this target gives theory and diagnostic interpretation, not
  a complete same-shot validation dataset. It requires digitized `dI/dt` and
  voltage traces, 3D field/velocity measurements, volume-integrated `J dot E`,
  and neutron-response coupling before it can close field-circuit predictive
  validation.

### 2026-05-05: Esaulov 2003 2D MHRDR DPF Target

- Reviewed `KnowledgeReference/esaulov_2003_2d_mhd_dpf.md`.
- Added `esaulov_2d_mhrdr_dpf_targets()` as a partial 2D multi-temperature MHD
  and thermal neutron-rate context target for the LANL Begay DPF.
- Encoded device parameters: Mather-type Begay device, inner electrode radius
  `1.18 cm`, outer electrode radius `3.65 cm`, inner electrode length `15.7 cm`,
  deuterium fill `1 Torr`, capacitance `36.4 uF`, charging voltage `14 kV`, and
  series inductance `178 nH`.
- Encoded model physics: MHRDR uses multi-temperature ion/electron/radiation
  MHD, electron and ion thermal conduction, resistive diffusion, radiation
  diffusion, Lorentz force, shock hydrodynamics, self-consistent external
  circuit coupling, and Maxwell-averaged D-D cross sections for neutron-rate
  computation.
- Extracted phase/context values: current-sheath formation examples around
  `0.9` and `2.0 us`, acceleration slices at `1.0` and `2.0 us`, collapse
  pressure contours around `2.6` and `2.65 us`, local neutron-rate peaks at
  `2.74` and `2.92 us`, radial slices at `2.72` and `2.90 us`, focus duration
  `100-150 ns`, current during acceleration about `50-100 kA`, electrode
  voltage drop about `1-2 kV`, abstract density above `1e19 cm^-3`, and
  axis-history temperature scale to `5 keV`.
- Encoded scope limits: the target assumes a high-pressure thermal-MHD regime,
  treats beam-target mechanisms as outside the primary target, uses figure-scale
  quantities as context only, and lacks digitized traces, error bars, detector
  response, and same-shot experimental profiles.
- Current corpus status after this ratchet:
  - coded KR target records: 26
  - unique coded KR target source files: 22
  - DPF-named markdown files represented by coded targets: 21 of 54
  - DPF-named markdown files closed by explicit review decisions: 6 of 54
  - total DPF-named markdown files review-closed: 27 of 54
  - unreviewed DPF-named markdown files: 27
- Updated triage counts after removing this source:
  - circuit waveform candidates: 12
  - phase timing candidates: 12
  - spatial density candidates: 6
  - spatial magnetic/EM candidates: 12
  - spatial temperature candidates: 22
  - neutron validation candidates: 21
  - uncertainty candidates: 0
- Verification:
  - `python3 -m py_compile src/dpf/validation/kr_targets.py src/dpf/validation/__init__.py tests/test_kr_targets.py`
  - `python3 -m pytest tests/test_kr_targets.py tests/test_kr_corpus.py -q`
    passed (`60 passed in 0.58s`)
  - Broad post-ratchet verification passed:
    `python3 -m pytest tests/test_quality_assessment.py tests/test_kr_targets.py tests/test_kr_corpus.py -q`
    (`111 passed in 0.57s`);
    `python3 -m pytest tests/test_mhd_physics_integration.py tests/test_mhd_numerical_fidelity.py tests/test_circuit_field_coupling.py tests/test_physics_fidelity.py tests/test_uncertainty_budget.py -q`
    (`87 passed, 3 skipped in 1.43s`); `git diff --check` clean.
- Remaining limit: this source strengthens the thermal-MHD branch of the
  validation plan but cannot validate end-to-end DPF neutron prediction. It
  lacks absolute neutron yield, neutron timing, spectra, anisotropy, detector
  response, kinetic beam-target physics, and uncertainty.

### 2026-05-05: FAETON-I 2025 High-Voltage DPF Target

- Reviewed
  `KnowledgeReference/faeton-i-investigation-of-plasma-dynamics-and-radiation-output-of-a-100-kv-plasma-focus-device.md`.
- Added `faeton_i_high_voltage_dpf_targets()` as a partial high-voltage DPF
  validation target. The local markdown extraction contains the references,
  conclusion, and Table 3 region, not the full paper body, so the target is
  intentionally marked partial.
- Encoded Table 3 shot values: shot `1062` with `fcr=0.4`, `fcr2=0.35`,
  `Vp=37.3 kV`, code yield `2.77e9`, measured yield `3e9`; shot `1036`
  with `fcr=0.72`, `Vp=101.4 kV`, code yield `2.54e10`, measured yield
  `2.21e10`; shot `1027` with `fcr=0.8`, `Vp=160.5 kV`, code yield
  `5.5e10`, measured yield `5.44e10`; and shot `895` with `fcr=0.9`,
  `Vp=194 kV`, code yield `4.1e10`, measured yield `6e10`.
- Encoded interpretation limits from the source: `fcr=0.7` marks good current
  sheath formation, exceptional shots use `fcr=0.8-0.9`, peak inductive
  voltage `Vmax` is a better high-voltage PF indicator than current-dip
  severity when restrikes truncate the dip, and the voltage spike is reported
  pre-stagnation and dynamics-induced.
- Encoded neutron/radiation diagnostics: consistent D-D yield `2.5e10` over
  five shots without gas refill, exceptional D-D yield up to `8e10`, forward
  anisotropy factor `1.6`, neutron energy peak `2.5 MeV` with `0.3 MeV`
  uncertainty, PMT scintillators at `5`, `10`, `20`, and `40 m`, `40 m` nTOF,
  `30 cm` lead shielding for gamma measurements above `3 MeV`, and Faraday-cup
  deuteron energy about `350 keV`.
- Recorded D-T Faeton-X values only under `projections_not_validation_targets`:
  `2e14` neutrons for `65 kV`, `1 MJ`, `4 MA`, and `2e15` neutrons for
  `150 kV`, `5 MJ`, `7 MA`. These are not treated as validated FAETON-I D-D
  evidence.
- Current corpus status after this ratchet:
  - coded KR target records: 27
  - unique coded KR target source files: 23
  - DPF-named markdown files represented by coded targets: 22 of 54
  - DPF-named markdown files closed by explicit review decisions: 6 of 54
  - total DPF-named markdown files review-closed: 28 of 54
  - unreviewed DPF-named markdown files: 26
- Updated triage counts after removing this source:
  - circuit waveform candidates: 11
  - phase timing candidates: 11
  - spatial density candidates: 6
  - spatial magnetic/EM candidates: 11
  - spatial temperature candidates: 21
  - neutron validation candidates: 20
  - uncertainty candidates: 0
- Verification:
  - `python3 -m py_compile src/dpf/validation/kr_targets.py src/dpf/validation/__init__.py tests/test_kr_targets.py`
  - `python3 -m pytest tests/test_kr_targets.py tests/test_kr_corpus.py -q`
    passed (`61 passed in 0.57s`)
  - Broad post-ratchet verification passed:
    `python3 -m pytest tests/test_quality_assessment.py tests/test_kr_targets.py tests/test_kr_corpus.py -q`
    (`112 passed in 0.62s`);
    `python3 -m pytest tests/test_mhd_physics_integration.py tests/test_mhd_numerical_fidelity.py tests/test_circuit_field_coupling.py tests/test_physics_fidelity.py tests/test_uncertainty_budget.py -q`
    (`87 passed, 3 skipped in 1.51s`); `git diff --check` clean.
- Remaining limit: this source strengthens high-voltage DPF waveform/yield and
  detector-response requirements, but it does not close predictive validation.
  The local extract lacks digitized current/voltage traces, absolute phase
  times, spatial density/temperature/magnetic-field profiles, full detector
  response and calibration uncertainty, full neutron histories/spectra, and
  the complete shot dataset.

### 2026-05-05: Lee/RADPF Theoretical Model-Scope Target

- Reviewed `KnowledgeReference/lee_radpf_theory.md`.
- Added `lee_radpf_theory_model_scope_targets()` as a reduced-model scope
  target, not as experimental validation evidence.
- Encoded model structure: external circuit and sheath motion are coupled; the
  equation of motion is affected by current and the circuit equation is
  affected by sheath motion/position; plasma resistance is ignored for the
  electromagnetic-drive approximation; and axial/radial tube voltage is treated
  as inductive in the reduced model.
- Encoded phase assumptions: axial phase uses a snowplow current sheath for
  trajectory, speed, and current-profile fitting; radial phase replaces the
  singular thin-snowplow limit with a slug model where the magnetic piston
  follows a shock front; reflected shock begins when the radial shock reaches
  the axis; and pinch breakup is modeled as an expanded uniform current column.
- Extracted timing/scale constraints: `alpha` is electrical time over axial
  transit time, `alpha1` is axial transit over radial transit time, axial
  transit time is characteristically about `20` times radial shock transit time,
  the typical axial/radial characteristic time ratio is about `40`, reflected
  shock speed is `0.3` of the on-axis inward radial shock speed, and the
  communication delay expression is `(rp - rs) / SDS`.
- Encoded radiation/temperature constraints: shocked-plasma temperature is
  computed from shock speed, slow-compression temperature from energy balance,
  Spitzer resistivity is used, bremsstrahlung/recombination/line losses are
  represented, self-absorption drives volumetric-to-surface emission transition,
  deuterium radiation collapse critical current is `1.6 MA`, and neon/argon
  line radiation can reduce the critical current below `100 kA`.
- Encoded neutron-model limits: thermonuclear yield uses density, volume,
  thermal `sigma v`, and time; beam-target yield is phenomenological; beam
  deuterons are produced by diode action near the anode; beam voltage is tied
  to `Vmax`; the code uses beam energy `3 * Vmax` for the cross section; the
  source reports code `Vmax` of order `20-50 kV`, experimental beam-energy
  relevance `50-150 keV`, and lower-voltage machine range `30-60 keV`; the
  empirical fit is `Yn = 9e10 * Ipinch^3.8` for `0.1-1 MA`; and the calibration
  point is `0.5 MA`, `7e9` neutrons.
- Current corpus status after this ratchet:
  - coded KR target records: 28
  - unique coded KR target source files: 24
  - DPF-named markdown files represented by coded targets: 23 of 54
  - DPF-named markdown files closed by explicit review decisions: 6 of 54
  - total DPF-named markdown files review-closed: 29 of 54
  - unreviewed DPF-named markdown files: 25
- Updated triage counts after removing this source:
  - circuit waveform candidates: 10
  - phase timing candidates: 10
  - spatial density candidates: 6
  - spatial magnetic/EM candidates: 10
  - spatial temperature candidates: 20
  - neutron validation candidates: 19
  - uncertainty candidates: 0
- Verification:
  - `python3 -m py_compile src/dpf/validation/kr_targets.py src/dpf/validation/__init__.py tests/test_kr_targets.py`
  - `python3 -m pytest tests/test_kr_targets.py tests/test_kr_corpus.py -q`
    passed (`62 passed in 0.59s`)
  - Broad post-ratchet verification passed:
    `python3 -m pytest tests/test_quality_assessment.py tests/test_kr_targets.py tests/test_kr_corpus.py -q`
    (`113 passed in 0.63s`);
    `python3 -m pytest tests/test_mhd_physics_integration.py tests/test_mhd_numerical_fidelity.py tests/test_circuit_field_coupling.py tests/test_physics_fidelity.py tests/test_uncertainty_budget.py -q`
    (`87 passed, 3 skipped in 1.58s`); `git diff --check` clean.
- Remaining limit: this source defines Lee/RADPF reduced-model assumptions and
  calibrations. It does not validate the code against same-shot experimental
  current traces, phase endpoints, spatial profiles, detector response,
  neutron spectra/anisotropy, or independent beam-target calibration.

### 2026-05-05: Blagoev 2025 Electric-Flux Formation Diagnostic Target

- Reviewed
  `KnowledgeReference/measurement-of-electric-flux-emission-a-new-diagnostic-for-the-dense-plasma-focus-a-b-blagoev12aa-v.md`.
- Closed
  `KnowledgeReference/measurement-of-electric-flux-emission-a-new-diagnostic-for-the-dense-plasma-focus-a-b-blagoev12aa-v-4.md`
  as a duplicate header/PDF-name variant of the same paper.
- Added `blagoev_electric_flux_diagnostic_targets()` as a formation-symmetry
  and electric-flux diagnostic target, not a neutron-yield validation target.
- Encoded the University of Sofia plasma focus context: `3 kJ` Mather device,
  `20 uF`, up to `40 kV`, hollow copper tube anode diameter `2 cm`, anode
  length `14.5 cm`, six cathode rods of `0.8 cm` diameter and `16 cm` length,
  cathode radius `3.5 cm`, chamber inner diameter `15.5 cm`, chamber height
  `35 cm`, and operation with air, argon, or deuterium.
- Encoded shot examples: shot `665`, argon `0.95 Torr`, `19.0 kV`; shot `668`,
  argon `0.83 Torr`, `19.1 kV`; and shot `667`, argon `0.77 Torr`, `19.0 kV`,
  with a reference singularity time `3.03 us`.
- Encoded diagnostic requirements: three symmetric identical D-dot probes
  placed through a hexagonal support, SMA central pins as floating conductors,
  `50 ohm` coax termination at both ends, CH2/CH3/CH4 probe channels, `1 ns`
  sampling, `10` point smoothing, and integration after baseline correction.
- Encoded calibration constraints: central-conductor symmetry test, voltage
  divider resistances `1306 ohm` and `13.2 ohm`, applied voltage `5.34 kV`,
  integrated D-dot maxima within `3%` of their mean, and `C1` capacitance
  ballpark `0.006 pF`.
- Encoded phase/symmetry interpretation: current maximum marks end of rundown,
  the interval from current maximum to current-derivative singularity is radial
  phase, lower pressure produces earlier singularity, similar D-dot shape and
  magnitude in formation/rundown indicate adequate symmetry, radial-phase
  divergence indicates changing azimuthal behavior, and Rogowski `dI/dt` can be
  contaminated by electric-flux pickup.
- Current corpus status after this ratchet:
  - coded KR target records: 29
  - unique coded KR target source files: 25
  - DPF-named markdown files represented by coded targets: 24 of 54
  - DPF-named markdown files closed by explicit review decisions: 7 of 54
  - total DPF-named markdown files review-closed: 31 of 54
  - unreviewed DPF-named markdown files: 23
- Updated triage counts after removing these sources:
  - circuit waveform candidates: 8
  - phase timing candidates: 8
  - spatial density candidates: 6
  - spatial magnetic/EM candidates: 8
  - spatial temperature candidates: 18
  - neutron validation candidates: 17
  - uncertainty candidates: 0
- Verification:
  - `python3 -m py_compile src/dpf/validation/kr_targets.py src/dpf/validation/kr_corpus.py src/dpf/validation/__init__.py tests/test_kr_targets.py tests/test_kr_corpus.py`
  - `python3 -m pytest tests/test_kr_targets.py tests/test_kr_corpus.py -q`
    passed (`63 passed in 0.57s`)
  - Broad post-ratchet verification passed:
    `python3 -m pytest tests/test_quality_assessment.py tests/test_kr_targets.py tests/test_kr_corpus.py -q`
    (`114 passed in 0.59s`);
    `python3 -m pytest tests/test_mhd_physics_integration.py tests/test_mhd_numerical_fidelity.py tests/test_circuit_field_coupling.py tests/test_physics_fidelity.py tests/test_uncertainty_budget.py -q`
    (`87 passed, 3 skipped in 1.48s`); `git diff --check` clean.
- Remaining limit: this source improves startup/formation and diagnostic
  symmetry constraints. It still lacks digitized probe/current traces, per-point
  waveform uncertainty, independent phase endpoint diagnostics, calibrated
  electric-field reconstruction, same-shot density/temperature/magnetic-field
  profiles, and same-shot neutron outputs.

### 2026-05-05: Auluck 2024 Poloidal Magnetic-Field Dynamo Target

- Reviewed
  `KnowledgeReference/poloidal-magnetic-field-in-the-dense-plasma-focus.md`.
- Closed `KnowledgeReference/poloidal-magnetic-field-in-the-dense-plasma-focus-5.md`
  as a duplicate header/PDF-name variant of the same Physics of Plasmas letter.
- Added `auluck_poloidal_magnetic_field_targets()` as a poloidal/axial
  magnetic-field scope and proposed-test target.
- Encoded the source's diagnostic warning: point measurement of axial magnetic
  field inside the plasma with a magnetic probe is treated as meaningless
  because the probe has finite `1-2 mm` spatial resolution, perturbs plasma
  flow/current, and forms a Langmuir sheath; Faraday-rotation Abel inversion is
  available for the azimuthal component but not for the axial component.
- Encoded the simple dynamo hypothesis: a curved plasma armature in the
  geomagnetic seed field generates azimuthal electric field through generalized
  Ohm's law; the Hall term is neglected as a model assumption; a zero-resistivity
  limit is used; and the magnetic Reynolds number is assumed much greater than
  one for ballpark plasma-focus values.
- Encoded GPF/GV context: coordinates scale by anode radius, density by fill-gas
  mass density, magnetic field by `B0 = mu0 * I(t) / (2*pi*a*r_tilde)`, velocity
  by `B0` and fill density, Mather-type GV surfaces resemble experimental
  plasma shapes, and the flux function evolves in Hamilton-Jacobi form.
- Encoded circuit implications: MHD codes that neglect the dynamo may
  overestimate observed current, apparent current loss may be azimuthal
  circulating current, Lee radial current fraction should vary under an external
  axial-field sweep, and equivalent loop voltage may include a geomagnetic term
  independent of charging voltage.
- Encoded proposed experiment: use a Helmholtz coil with DC variable polarity,
  a uniform axial field over the whole small DPF, maximum field not more than
  `2` times the local geomagnetic field, monitor current derivative/integrated
  current/poloidal flux emission, and look for variation near the geomagnetic
  null. Nonuniform or excessively high applied fields are explicitly not valid
  tests.
- Encoded supporting Nikulin observation: a cone-shaped copper foil on a
  `2.5 kJ` plasma focus was twisted rather than radially imploded; the source
  argues a purely azimuthal field cannot produce that torque.
- Current corpus status after this ratchet:
  - coded KR target records: 30
  - unique coded KR target source files: 26
  - DPF-named markdown files represented by coded targets: 25 of 54
  - DPF-named markdown files closed by explicit review decisions: 8 of 54
  - total DPF-named markdown files review-closed: 33 of 54
  - unreviewed DPF-named markdown files: 21
- Updated triage counts after removing these sources:
  - circuit waveform candidates: 6
  - phase timing candidates: 8
  - spatial density candidates: 4
  - spatial magnetic/EM candidates: 6
  - spatial temperature candidates: 16
  - neutron validation candidates: 15
  - uncertainty candidates: 0
- Verification:
  - `python3 -m py_compile src/dpf/validation/kr_targets.py src/dpf/validation/kr_corpus.py src/dpf/validation/__init__.py tests/test_kr_targets.py tests/test_kr_corpus.py`
  - `python3 -m pytest tests/test_kr_targets.py tests/test_kr_corpus.py -q`
    passed (`64 passed in 0.57s`)
  - Broad post-ratchet verification passed:
    `python3 -m pytest tests/test_quality_assessment.py tests/test_kr_targets.py tests/test_kr_corpus.py -q`
    (`115 passed in 0.58s`);
    `python3 -m pytest tests/test_mhd_physics_integration.py tests/test_mhd_numerical_fidelity.py tests/test_circuit_field_coupling.py tests/test_physics_fidelity.py tests/test_uncertainty_budget.py -q`
    (`87 passed, 3 skipped in 1.47s`); `git diff --check` clean.
- Remaining limit: this source is a model-scope and proposed-test constraint.
  It lacks the completed external-field sweep dataset, calibrated poloidal flux
  signals, radial-current-fraction response, 3D magnetic reconstruction, and
  same-shot neutron yield/anisotropy response.

### 2026-05-05: Wante 2025 UNU/ICTP Nitrogen-Ion Irradiation Target

- Reviewed
  `KnowledgeReference/regular-article-nitrogen-ion-irradiation-of-carbon-thin-lms-using-a-dense-plasma-focus-enhanced.md`.
- Added `wante_nitrogen_ion_irradiation_targets()` as an ion-beam and
  material-processing target, not a neutron or end-to-end DPF validation target.
- Encoded UNU/ICTP PF configuration: nominal `3.0 kJ` device operated at
  `2.54 kJ`, `30 uF`, `13 kV`, `156 nH`, `21.4 mOhm`, anode radius `0.95 cm`,
  cathode radius `3.2 cm`, anode length `16 cm`, anode diameter `1.9 cm`, six
  copper cathode rods, Pyrex insulator, nitrogen purity `99.999%`, optimal
  pressure `1.5 mbar`, initial vacuum `5e-3 mbar`, four preliminary shots for
  stable pinch, sample distance `38 cm`, and irradiation sequences of `6`, `12`,
  and `24` shots at `5 min` intervals.
- Encoded diagnostics and Lee fit: Yokogawa `DL7480` captures current, voltage,
  and ion signals; Faraday cup biased ion collector uses `-45 V`; ion TOF is
  defined from X-ray peak to ion peak; X-ray peak aligns with voltage peak; Lee
  current-fit parameters are `fm=0.03`, `fc=0.7`, `fmr=0.18`, and `fcr=0.85`.
- Extracted ion-beam outputs: measured nitrogen ion energy `72.40 keV`, Lee
  model ion energy `71.0 keV`, ion flux `7.2e27 ions m^-2 s^-1`, and ion
  fluence `6.4e19 ions m^-2`.
- Encoded contextual plasma scales from the source: pinch temperature order
  `1e6 K` and particle-density range `1e18-1e20 m^-3`, explicitly marked as
  contextual rather than same-shot profile validation.
- Encoded material-response constraints: nitrogen doping `7.06%`, `5.96%`, and
  `7.93%` for `6`, `12`, and `24` shots; deposition rates `1.18%`, `0.50%`,
  and `0.33%` per shot; copper impurity from anode ablation increasing to
  `2.11%` at `24` shots; fluorine falling from `12.06%` to `4.94%`; crystallite
  size increasing from `6.27 nm` to `11.16 nm`; new XRD peaks at `52` and
  `76` degrees; and interlayer spacing decreasing from `0.37 nm` to `0.340 nm`.
- Current corpus status after this ratchet:
  - coded KR target records: 31
  - unique coded KR target source files: 27
  - DPF-named markdown files represented by coded targets: 26 of 54
  - DPF-named markdown files closed by explicit review decisions: 8 of 54
  - total DPF-named markdown files review-closed: 34 of 54
  - unreviewed DPF-named markdown files: 20
- Updated triage counts after removing this source:
  - circuit waveform candidates: 5
  - phase timing candidates: 7
  - spatial density candidates: 3
  - spatial magnetic/EM candidates: 6
  - spatial temperature candidates: 15
  - neutron validation candidates: 14
  - uncertainty candidates: 0
- Verification:
  - `python3 -m py_compile src/dpf/validation/kr_targets.py src/dpf/validation/__init__.py tests/test_kr_targets.py`
  - `python3 -m pytest tests/test_kr_targets.py tests/test_kr_corpus.py -q`
    passed (`65 passed in 0.49s`)
  - Broad post-ratchet verification passed:
    `python3 -m pytest tests/test_quality_assessment.py tests/test_kr_targets.py tests/test_kr_corpus.py -q`
    (`116 passed in 0.62s`);
    `python3 -m pytest tests/test_mhd_physics_integration.py tests/test_mhd_numerical_fidelity.py tests/test_circuit_field_coupling.py tests/test_physics_fidelity.py tests/test_uncertainty_budget.py -q`
    (`87 passed, 3 skipped in 1.54s`); `git diff --check` clean.
- Remaining limit: this target validates only a bounded ion-beam/material
  processing use case. It lacks digitized current/voltage/ion waveforms,
  Faraday-cup response uncertainty, absolute X-ray/ion peak times, same-shot
  density/temperature/magnetic profiles, and any neutron output.

### 2026-05-05: Kiai 2025 Double 3 MJ DPF/ICF Concept Target

- Reviewed
  `KnowledgeReference/2025-double-3mj-dense-plasma-focus-thermonuclear-icf.md`.
- Closed both local duplicates as reviewed:
  `KnowledgeReference/double-3-mj-dense-plasma-focus-for-thermonuclear-drive-inertial-confinement-fusion-5.md`
  and
  `KnowledgeReference/double-3-mj-dense-plasma-focus-for-thermonuclear-drive-inertial-confinement-fusion.md`.
- Added `kiai_double_dpf_icf_concept_targets()` as a concept and experimental
  roadmap target, not as experimental validation evidence.
- Encoded the full-scale design table: deuterium at `10 torr`, impedance
  `12.5 mOhm`, peak circuit current `20 MA`, charging voltage `200 kV`,
  capacitance `150 uF`, stored bank energy `6 MJ` total with `3 MJ` per DPF,
  inductance `35 nH`, circuit period `17.5 us`, anode radius `15 cm`, anode
  length `80 cm`, cathode radius `22.5 cm`, axial speed `29.5 cm/us`, radial
  speed `42.4 cm/us`, pinch radius `1.8 cm`, pinch lifetime `300 ns` for each
  DPF, pinch length `12 cm`, current loss factor `0.7`, mass sweep factor
  `0.13`, and induced voltage `20 MV`.
- Encoded the proposed `30 kJ` prototype table: operating voltage `50-60 kV`,
  capacitance `500 uF`, plasma/deuteron density `6e25 ions/m^3`, projected
  fusion neutron yield `1e10 neutrons/shot`, pinch efficiency `20-30%`, peak
  current `3.54-4.24 MA`, maximum pinch current `0.71-1.06 MA`, pinch radius
  `3.0 mm`, pinch length `2.0 cm`, and pinch lifetime `50 ns`.
- Encoded the HTS and pellet projections as model outputs only: HTS field
  `10-15 T`, pellet ignition temperature `10-20 keV`, simplified with-HTS
  comparison `75 MW` fusion and `30 MW` electric power, without-HTS comparison
  `25 MW` fusion and `10 MW` electric power, and an explicitly flagged extreme
  pellet power projection of `3.61 PW` fusion and `613 TW` electric.
- Encoded the proposed validation roadmap: single `30 kJ` DPF prototype,
  synchronized double `30 kJ` DPF, and full-scale fusion testing with plasma
  diagnostics, neutron-yield measurements, and high-speed imaging.
- Current corpus status after this ratchet:
  - coded KR target records: 32
  - unique coded KR target source files: 28
  - DPF-named markdown files represented by coded targets: 27 of 54
  - DPF-named markdown files closed by explicit review decisions: 10 of 54
  - total DPF-named markdown files review-closed: 37 of 54
  - unreviewed DPF-named markdown files: 17
- Updated triage counts after removing these sources:
  - circuit waveform candidates: 2
  - phase timing candidates: 7
  - spatial density candidates: 3
  - spatial magnetic/EM candidates: 3
  - spatial temperature candidates: 12
  - neutron validation candidates: 11
  - uncertainty candidates: 0
- Verification:
  - `python3 -m py_compile src/dpf/validation/kr_targets.py src/dpf/validation/kr_corpus.py src/dpf/validation/__init__.py tests/test_kr_targets.py tests/test_kr_corpus.py`
  - `python3 -m pytest tests/test_kr_targets.py tests/test_kr_corpus.py -q`
    passed (`66 passed in 0.57s`)
  - Broad post-ratchet verification passed:
    `python3 -m pytest tests/test_quality_assessment.py tests/test_kr_targets.py tests/test_kr_corpus.py -q`
    (`117 passed in 0.61s`);
    `python3 -m pytest tests/test_mhd_physics_integration.py tests/test_mhd_numerical_fidelity.py tests/test_circuit_field_coupling.py tests/test_physics_fidelity.py tests/test_uncertainty_budget.py -q`
    (`87 passed, 3 skipped in 1.55s`); `git diff --check` clean.
- Remaining limit: this source is theoretical and explicitly points to future
  laboratory validation. It does not provide measured current/voltage traces,
  synchronized double-DPF timing, same-shot density/temperature/HTS-field
  profiles, DT pellet coupling diagnostics, measured neutron yield/timing/
  spectrum/anisotropy, detector response, full energy accounting, or validated
  scale-up from `30 kJ` to `6 MJ`.

### 2026-05-05: Beresnyak 2018 HAWK 3D MHD Model-Scope Target

- Reviewed `KnowledgeReference/beresnyak_2018_dpf_hawk_simulations.md`.
- Added `beresnyak_hawk_3d_mhd_targets()` as a HAWK-specific 3D MHD
  model-scope target, not as an experimental validation packet.
- Encoded HAWK setup: `665 kA` generator, `1.2 us` rise time, `720 nH`
  high-impedance generator inductance, local plasma injection by plasma guns,
  evacuated interelectrode space, and fully ionized deuterium assumption.
- Encoded circuit coupling: `720 nH`, `0.15 ohm`, `1.07 uF`, initial
  capacitor voltage `640 kV`, zero initial current, current and `dI/dt` as
  simulation inputs, azimuthal magnetic boundary from current, velocity-gradient
  boundary from `dI/dt`, and device voltage from integrated electric field.
- Encoded HAWK geometry and injected-plasma setup: anode radius `6.33 cm`,
  anode length `4 cm`, cathode radius `8.57 cm`, high-to-low injected-density
  ratio `2`, background density `1/4 rho0`, azimuthal modes `m=0`, `m=3`, and
  `m=6`, and characteristic density `1e-7 g/cc` or `3e16 cm^-3`.
- Encoded phase/current behavior: Lee-estimated density gives pinch time
  `0.95 us`, near the current peak; device voltage is typically below `10 kV`
  at the target density; short-circuit sine period is `5.2 us`; grid resolution
  examples are `480 x 480 x 288`.
- Encoded model outputs and limits: total thermal-yield metric peaks at
  `9e15 cm^-3`, thermal fusion is explicitly subdominant and not a projected
  HAWK yield, Hall-MHD positive-polarity runs give faster/tighter pinch near
  the anode, Spitzer resistivity does not qualitatively change dynamics, and
  stochastic ion acceleration gives a mostly isotropic power-law tail to about
  `200 keV`.
- Current corpus status after this ratchet:
  - coded KR target records: 33
  - unique coded KR target source files: 29
  - DPF-named markdown files represented by coded targets: 28 of 54
  - DPF-named markdown files closed by explicit review decisions: 10 of 54
  - total DPF-named markdown files review-closed: 38 of 54
  - unreviewed DPF-named markdown files: 16
- Updated triage counts after removing this source:
  - circuit waveform candidates: 2
  - phase timing candidates: 6
  - spatial density candidates: 2
  - spatial magnetic/EM candidates: 3
  - spatial temperature candidates: 11
  - neutron validation candidates: 10
  - uncertainty candidates: 0
- Verification:
  - `python3 -m py_compile src/dpf/validation/kr_targets.py src/dpf/validation/kr_corpus.py src/dpf/validation/__init__.py tests/test_kr_targets.py tests/test_kr_corpus.py`
  - `python3 -m pytest tests/test_kr_targets.py tests/test_kr_corpus.py -q`
    passed (`67 passed in 0.61s`)
  - Broad post-ratchet verification passed:
    `python3 -m pytest tests/test_quality_assessment.py tests/test_kr_targets.py tests/test_kr_corpus.py -q`
    (`118 passed in 0.64s`);
    `python3 -m pytest tests/test_mhd_physics_integration.py tests/test_mhd_numerical_fidelity.py tests/test_circuit_field_coupling.py tests/test_physics_fidelity.py tests/test_uncertainty_budget.py -q`
    (`87 passed, 3 skipped in 1.62s`); `git diff --check` clean.
- Remaining limit: HAWK experiments were planned in this paper, current
  disruption was not modeled, and the local extract lacks measured HAWK
  current/voltage traces, measured phase endpoints, spatial profile
  diagnostics, measured neutron yield/timing/spectrum/anisotropy, detector
  response, and uncertainty.

### 2026-05-05: Wang/Yang 1999 DPF-16 Metallic-Vapor Interferometry Target

- Reviewed
  `KnowledgeReference/observation-of-the-metallic-vapor-from-a-plasma-focus-wang-xinxin-3-yang-jinji-department-of.md`.
- Added `wang_metallic_vapor_interferometry_targets()` as an interferometry
  and anode-material-vapor target, not as a neutron or complete DPF validation
  target.
- Encoded DPF-16 setup: `16 kJ`, `20 kV`, `380 kA`, Mather type, hydrogen
  fill pressure `70-650 Pa`, typical interferograms at `200 Pa`, and
  higher-pressure vapor-development images at `330 Pa`.
- Encoded geometry: oxygen-free copper anode, anode diameter `66 mm`, anode
  and cathode length `265 mm`, tungsten target `10 mm` diameter and `6 mm`
  high, and interferometer field of view about `60 mm`.
- Encoded phase timing: `t=0` is the pinch spike in the `dI/dt` waveform and
  maximum compression above the anode; compression frames at `-200`, `-140`,
  and `-60 ns`; expansion beginning at `40 ns`; post-focus expansion at
  `200 ns`; metallic vapor visible at `280 ns`; and higher-pressure vapor
  frames at `220` and `300 ns`.
- Encoded evidence interpretation: laser differential interferometry records
  plasma-sheath evolution; a high-density volume emerges from the anode target
  after the focus is over; target erosion after many shots supports material
  evaporation; the high-density volume disappears when a hollow anode replaces
  the target; and the source links the delayed metallic plasma to hard X-ray
  emission several hundred nanoseconds after focus.
- Current corpus status after this ratchet:
  - coded KR target records: 34
  - unique coded KR target source files: 30
  - DPF-named markdown files represented by coded targets: 29 of 54
  - DPF-named markdown files closed by explicit review decisions: 10 of 54
  - total DPF-named markdown files review-closed: 39 of 54
  - unreviewed DPF-named markdown files: 15
- Updated triage counts after removing this source:
  - circuit waveform candidates: 2
  - phase timing candidates: 5
  - spatial density candidates: 1
  - spatial magnetic/EM candidates: 3
  - spatial temperature candidates: 10
  - neutron validation candidates: 9
  - uncertainty candidates: 0
- Verification:
  - `python3 -m py_compile src/dpf/validation/kr_targets.py src/dpf/validation/kr_corpus.py src/dpf/validation/__init__.py tests/test_kr_targets.py tests/test_kr_corpus.py`
  - `python3 -m pytest tests/test_kr_targets.py tests/test_kr_corpus.py -q`
    passed (`68 passed in 0.56s`)
  - Broad post-ratchet verification passed:
    `python3 -m pytest tests/test_quality_assessment.py tests/test_kr_targets.py tests/test_kr_corpus.py -q`
    (`119 passed in 0.60s`);
    `python3 -m pytest tests/test_mhd_physics_integration.py tests/test_mhd_numerical_fidelity.py tests/test_circuit_field_coupling.py tests/test_physics_fidelity.py tests/test_uncertainty_budget.py -q`
    (`87 passed, 3 skipped in 1.55s`); `git diff --check` clean.
- Remaining limit: this source is qualitative visual evidence. It lacks
  digitized `dI/dt`, current, voltage, interferogram phase shift, density
  inversion, vapor-species spectroscopy, X-ray time history/spectrum, electron
  beam energy/current, neutron diagnostics, detector response, and uncertainty.

### 2026-05-05: Altarabulsi 2024 Deuteron-Beam Fluence Target

- Reviewed
  `KnowledgeReference/original-deuteron-beam-fluence-emitted-from-dense-plasma-focus.md`.
- Added `altarabulsi_deuteron_beam_fluence_targets()` as a Lee-code
  deuteron-beam fluence target, not as neutron validation.
- Encoded three fitted devices: PF-1000 (`863.1 kJ`), MPEF-12 kJ (`9.7 kJ`),
  and PF-2.7 kJ (`2.7 kJ`) operated in deuterium using `RADPFV6.16FIB`.
- Encoded Table 1 device parameters, including PF-1000 `L0=33.5 nH`,
  `C0=1332 uF`, `r0=6.3 mOhm`, `a=11.5 cm`, `b=16 cm`, `V0=36 kV`,
  `p0=3.5 Torr`; MPEF-12 kJ `L0=65 nH`, `C0=40 uF`, `r0=1 mOhm`,
  `a=3 cm`, `b=5.5 cm`, `V0=22 kV`, `p0=3 Torr`; and PF-2.7 kJ
  `L0=110 nH`, `C0=30 uF`, `r0=22 mOhm`, `a=0.95 cm`, `b=3.2 cm`,
  `V0=13.5 kV`, `p0=0.15 Torr`.
- Encoded current-waveform fitting requirements: computed current is fitted to
  measured discharge current by adjusting Lee mass/current factors and
  sometimes `L0`/`r0`; the example MPEF-12 fit is to the end of pinch at about
  `2.08 us`; after that point divergence is not considered important for ion
  acceleration in this model.
- Encoded Table 3 fluence comparisons: PF-1000 at `14 cm`, `0.5 Torr`,
  simulated `7.3e19 ions/m^2` versus measured about `7.5e19`; MPEF-12 kJ at
  `14 cm`, pressures `0.76-7.5 Torr`, simulated `5.5e18-7.5e18` versus
  measured values with errors; and PF-2.7 kJ at `40 cm`, pressures
  `0.075-0.6 Torr`, simulated `1.77e15-4.94e15` versus measured values with
  errors.
- Encoded distance/application scaling: pinch-exit fluence order `1e20
  ions/m^2`, `14 cm` fluence order `1e19 ions/m^2`, PF-24 at `11 Torr` with
  pinch-exit fluence `3.87e20 ions/m^2`, flux dropping from `8.7e27` at the
  pinch exit to `2.61e26 ions/m^2/s` at `26 cm`, and energy flux dropping from
  `1.37e14` to `4.09e12 W/m^2`.
- Current corpus status after this ratchet:
  - coded KR target records: 35
  - unique coded KR target source files: 31
  - DPF-named markdown files represented by coded targets: 30 of 54
  - DPF-named markdown files closed by explicit review decisions: 10 of 54
  - total DPF-named markdown files review-closed: 40 of 54
  - unreviewed DPF-named markdown files: 14
- Updated triage counts after removing this source:
  - circuit waveform candidates: 1
  - phase timing candidates: 4
  - spatial density candidates: 1
  - spatial magnetic/EM candidates: 3
  - spatial temperature candidates: 9
  - neutron validation candidates: 8
  - uncertainty candidates: 0
- Verification:
  - `python3 -m py_compile src/dpf/validation/kr_targets.py src/dpf/validation/kr_corpus.py src/dpf/validation/__init__.py tests/test_kr_targets.py tests/test_kr_corpus.py`
  - `python3 -m pytest tests/test_kr_targets.py tests/test_kr_corpus.py -q`
    passed (`69 passed in 0.57s`)
  - Broad post-ratchet verification passed:
    `python3 -m pytest tests/test_quality_assessment.py tests/test_kr_targets.py tests/test_kr_corpus.py -q`
    (`120 passed in 0.61s`);
    `python3 -m pytest tests/test_mhd_physics_integration.py tests/test_mhd_numerical_fidelity.py tests/test_circuit_field_coupling.py tests/test_physics_fidelity.py tests/test_uncertainty_budget.py -q`
    (`87 passed, 3 skipped in 1.58s`); `git diff --check` clean.
- Remaining limit: this source validates a bounded ion-beam fluence workflow
  after current-waveform fitting. It lacks raw digitized current/voltage
  waveforms, raw fluence detector response, raw detector calibration,
  same-shot density/temperature/beam divergence diagnostics, complete
  uncertainty propagation, and neutron timing/spectrum/anisotropy validation.

### 2026-05-05: Narkis/Hahn 2021 Kr-Doped Gemini-Like DPF MHD Target

- Reviewed `KnowledgeReference/seyler-2021-kr-doped-dpf-mhd.md`.
- Added `narkis_kr_doped_dpf_mhd_targets()` as a 2D radiation-MHD
  model-scope target for Kr-doped, Gemini-like DPF simulations.
- Encoded the core warning from the source: fully kinetic simulations are
  required for pinch stagnation and total neutron yield; MHD cannot capture
  kinetic effects or beam-target neutron production.
- Encoded setup: HYDRA quasi-2D `R-Z` geometry with one azimuthal cell, current
  levels `2-3 MA`, Kr volume fractions `0`, `0.1%`, and `1%`, charging
  voltages `35`, `40`, `45`, and `50 kV`, experimental current data only for
  `35` and `40 kV`, anode radius `7.62 cm`, cathode radius `10.16 cm`, anode
  length `43.18 cm`, cathode length `59.18 cm`, and near-cap mesh resolution
  `200 x 200 um`.
- Encoded circuit and initial-condition limits: RLC circuit with `R=1.4 mOhm`,
  `L=40 nH`, `C=432 uF`; resistance treated as a free parameter; fill pressure
  scaled by `0.75`; matching implosion times and peak currents is described as
  a sanity check rather than strict quantitative comparison; breakdown physics
  is neglected.
- Encoded Table I: sheath-radius `5 mm` timing, ion/electron temperatures, and
  ion densities for all dopant/voltage cases. Example high-density case: `1%`
  Kr, `50 kV`, `ni=15.87e18 cm^-3`, `Ti=156 eV`, `Te=98.5 eV`, `t=6.525 us`.
- Encoded radiation and temperature results: Kr increases radiative losses,
  narrows the sheath, gives approximate peak temperatures `6.7`, `8.3`, and
  `12.6 keV` for `0%`, `0.1%`, and `1%` Kr, and leaves two-temperature
  behavior throughout radial implosion for `0.1%` and `1%` Kr.
- Encoded neutron outputs and caveats: thermonuclear yield order `1e9-1e10`,
  yield increases with Kr dopant in 2D MHD, all-point scaling exponents
  `5.726`, `4.643`, and `4.859`, and `35 kV` maximum `dN/dt` values
  `1.1e9`, `2.4e9`, and `1.8e9 neutrons/ns` for `0%`, `0.1%`, and `1%` Kr.
- Current corpus status after this ratchet:
  - coded KR target records: 36
  - unique coded KR target source files: 32
  - DPF-named markdown files represented by coded targets: 31 of 54
  - DPF-named markdown files closed by explicit review decisions: 10 of 54
  - total DPF-named markdown files review-closed: 41 of 54
  - unreviewed DPF-named markdown files: 13
- Updated triage counts after removing this source:
  - circuit waveform candidates: 0
  - phase timing candidates: 3
  - spatial density candidates: 1
  - spatial magnetic/EM candidates: 3
  - spatial temperature candidates: 8
  - neutron validation candidates: 7
  - uncertainty candidates: 0
- Verification:
  - `python3 -m py_compile src/dpf/validation/kr_targets.py src/dpf/validation/kr_corpus.py src/dpf/validation/__init__.py tests/test_kr_targets.py tests/test_kr_corpus.py`
  - `python3 -m pytest tests/test_kr_targets.py tests/test_kr_corpus.py -q`
    passed (`70 passed in 0.83s`)
  - Broad post-ratchet verification passed:
    `python3 -m pytest tests/test_quality_assessment.py tests/test_kr_targets.py tests/test_kr_corpus.py -q`
    (`121 passed in 0.63s`);
    `python3 -m pytest tests/test_mhd_physics_integration.py tests/test_mhd_numerical_fidelity.py tests/test_circuit_field_coupling.py tests/test_physics_fidelity.py tests/test_uncertainty_budget.py -q`
    (`87 passed, 3 skipped in 1.76s`); `git diff --check` clean.
- Remaining limit: this is not a predictive total-yield target. It lacks strict
  digitized current/voltage trace fitting, measured phase endpoints for every
  voltage/dopant case, breakdown physics, 3D instability growth, species
  separation, fully kinetic stagnation, beam-target neutron production,
  detector response, and neutron spectrum/anisotropy validation.

### 2026-05-05: Auluck 2022 DPF Theory Part 1 Extraction Decision

- Reviewed `KnowledgeReference/auluck-2022-dpf-theory-part1.md`.
- The local markdown is not usable as a line-referenced scientific target:
  despite metadata indicating a 74-page PDF with tables and figures, the
  extracted markdown contains only the final references page.
- Added an explicit `insufficient_extractable_validation_data` corpus decision
  instead of inferring theory content from the title or references.
- Current corpus status after this ratchet:
  - coded KR target records: 36
  - unique coded KR target source files: 32
  - DPF-named markdown files represented by coded targets: 31 of 54
  - DPF-named markdown files closed by explicit review decisions: 11 of 54
  - total DPF-named markdown files review-closed: 42 of 54
  - unreviewed DPF-named markdown files: 12
- Updated triage counts after removing this source:
  - circuit waveform candidates: 0
  - phase timing candidates: 3
  - spatial density candidates: 0
  - spatial magnetic/EM candidates: 2
  - spatial temperature candidates: 8
  - neutron validation candidates: 6
  - uncertainty candidates: 0
- Verification:
  - `python3 -m py_compile src/dpf/validation/kr_corpus.py tests/test_kr_corpus.py`
  - `python3 -m pytest tests/test_kr_targets.py tests/test_kr_corpus.py -q`
    passed (`70 passed in 0.49s`)
  - Broad post-ratchet verification passed:
    `python3 -m pytest tests/test_quality_assessment.py tests/test_kr_targets.py tests/test_kr_corpus.py -q`
    (`121 passed in 0.57s`);
    `python3 -m pytest tests/test_mhd_physics_integration.py tests/test_mhd_numerical_fidelity.py tests/test_circuit_field_coupling.py tests/test_physics_fidelity.py tests/test_uncertainty_budget.py -q`
    (`87 passed, 3 skipped in 1.55s`); `git diff --check` clean.
- Remaining limit: the original PDF would need to be re-ingested before any
  KR-only theory target can be extracted from this source.

### 2026-05-05: Auluck 2023 Neutron-Yield Scaling Failure Target

- Reviewed
  `KnowledgeReference/on-the-failure-of-neutron-yield-scaling-in-the-dense-plasma-focus-s-k-h-auluck-international.md`.
- Added `auluck_neutron_yield_scaling_failure_targets()` as a narrow
  theory/test target. Only the exposed conclusion and references were used.
- Encoded the source's core claim: large plasma-focus devices can abruptly stop
  following expected neutron-yield scaling above some voltage because the device
  must satisfy drive-parameter limits and generalized optimization criteria.
- Encoded the insulator-radius scaling claim: reaction yield should vary as the
  inverse fifth power of the outer-insulator-radius to anode-radius ratio; the
  source proposes reducing the ratio from typical `~1` to `~0.4` by placing the
  insulator in the shadow of the anode, with a possible two-order yield increase
  only if all optimization conditions are satisfied simultaneously.
- Encoded the proposed inexpensive tests: measure lift-off time and correlate it
  with drive parameter and insulator radius; change the operating pressure range
  by increasing insulator radius with an add-on insulator; and test insulators
  with outer radius less than the anode radius.
- Encoded the source's warning that small devices should study this scaling
  failure through lift-off-time measurements, not by using neutron measurements
  as the primary test.
- Current corpus status after this ratchet:
  - coded KR target records: 37
  - unique coded KR target source files: 33
  - DPF-named markdown files represented by coded targets: 32 of 54
  - DPF-named markdown files closed by explicit review decisions: 11 of 54
  - total DPF-named markdown files review-closed: 43 of 54
  - unreviewed DPF-named markdown files: 11
- Updated triage counts after removing this source:
  - circuit waveform candidates: 0
  - phase timing candidates: 3
  - spatial density candidates: 0
  - spatial magnetic/EM candidates: 1
  - spatial temperature candidates: 7
  - neutron validation candidates: 5
  - uncertainty candidates: 0
- Verification:
  - `python3 -m py_compile src/dpf/validation/kr_targets.py src/dpf/validation/__init__.py tests/test_kr_targets.py`
  - `python3 -m pytest tests/test_kr_targets.py tests/test_kr_corpus.py -q`
    passed (`71 passed in 0.58s`)
  - Broad post-ratchet verification passed:
    `python3 -m pytest tests/test_quality_assessment.py tests/test_kr_targets.py tests/test_kr_corpus.py -q`
    (`122 passed in 0.59s`);
    `python3 -m pytest tests/test_mhd_physics_integration.py tests/test_mhd_numerical_fidelity.py tests/test_circuit_field_coupling.py tests/test_physics_fidelity.py tests/test_uncertainty_budget.py -q`
    (`87 passed, 3 skipped in 1.57s`); `git diff --check` clean.
- Remaining limit: equations `12` and `17`, the derivation, and the actual
  validation data are not in the markdown. Full use requires PDF re-ingestion or
  a new KR source exposing liftoff-time, pressure-range, drive-parameter, and
  neutron-yield sweeps.

### 2026-05-05: Ou/FOI 2D Dense Plasma Focus Simulation Target

- Reviewed `KnowledgeReference/two-dimensional-simulation-of-dense-plasma-focus.md`.
- Closed `KnowledgeReference/two-dimensional-simulation-of-dense-plasma-focus-5.md`
  as a duplicate header/PDF-name variant of the same source.
- Added `ou_foi_2d_dpf_simulation_targets()` as a 2D MHD parameter-sweep target.
- Encoded FOI model scope: electron inertia ignored, simplified Ohm law closes
  Maxwell equations, electromagnetic solver `TVD-CP`, fluid solver `RTVD`,
  adiabatic single-phase ideal gas, high-resistivity swept/vacuum region,
  low-resistivity plasma region, fixed electrodes, Courant number `0.5`, and
  sine-current boundary `Imax * sin(2*pi*f*t)`.
- Encoded LLNL reference case: anode diameter `15.2 cm`, cathode-anode gap
  `4.3 cm`, peak current `2.5 MA`, fill pressure `2926 Pa`, sheath images at
  `3.9 us`, `6.2 us`, `7.4 us`, and breakup at `7.4 us`. The source says
  simulated morphology agrees with LLNL optical framing images but timing
  differs greatly.
- Encoded current sweep: amplitudes `1.5`, `2.0`, `2.5`, `3.0`, `3.5 MA`;
  pinch times `188.99`, `155.08`, `135.65`, `123.40`, `114.29 ns`; quarter
  period `135 ns`; and corresponding pinch currents `1.213`, `1.946`, `2.500`,
  `2.973`, `3.399 MA`.
- Encoded pressure/anode/gap trends: pressure sweep `133-2660 Pa`; sheath speed
  above `1e5 m/s`; sheath speed decreases with square root of pressure,
  increases with current, and decreases with anode radius; anode radii
  `30-50 mm`; gaps `15-35 mm`; gap has little effect on near-anode axial motion.
- Current corpus status after this ratchet:
  - coded KR target records: 38
  - unique coded KR target source files: 34
  - DPF-named markdown files represented by coded targets: 33 of 54
  - DPF-named markdown files closed by explicit review decisions: 12 of 54
  - total DPF-named markdown files review-closed: 45 of 54
  - unreviewed DPF-named markdown files: 9
- Updated triage counts after removing these sources:
  - circuit waveform candidates: 0
  - phase timing candidates: 1
  - spatial density candidates: 0
  - spatial magnetic/EM candidates: 1
  - spatial temperature candidates: 5
  - neutron validation candidates: 3
  - uncertainty candidates: 0
- Verification:
  - `python3 -m py_compile src/dpf/validation/kr_targets.py src/dpf/validation/kr_corpus.py src/dpf/validation/__init__.py tests/test_kr_targets.py tests/test_kr_corpus.py`
  - `python3 -m pytest tests/test_kr_targets.py tests/test_kr_corpus.py -q`
    passed (`72 passed in 0.49s`)
  - Broad post-ratchet verification passed:
    `python3 -m pytest tests/test_quality_assessment.py tests/test_kr_targets.py tests/test_kr_corpus.py -q`
    (`123 passed in 0.64s`);
    `python3 -m pytest tests/test_mhd_physics_integration.py tests/test_mhd_numerical_fidelity.py tests/test_circuit_field_coupling.py tests/test_physics_fidelity.py tests/test_uncertainty_budget.py -q`
    (`87 passed, 3 skipped in 1.65s`); `git diff --check` clean.
- Remaining limit: this is design guidance, not a complete validation packet. It
  lacks measured current/voltage traces, timing uncertainty, quantitative LLNL
  frame alignment, density/temperature/magnetic-field diagnostics, and neutron
  outputs.

### 2026-05-05: Sun 2025 Two-Temperature MHD Motion Target

- Reviewed
  `KnowledgeReference/2025-theoretical-and-numerical-studies-on-motion-process-of-dense-plasma-focus.md`.
- Added `sun_two_temperature_mhd_motion_targets()` as a two-temperature MHD
  motion and design-scaling target for UNU / UDMPF1 / PF-1000 studies.
- Encoded model scope: nonideal two-temperature MHD coupled to an external RLC
  circuit, electron-ion thermal nonequilibrium, Braginskii transport
  coefficients, resistive effects, and qualitative/plot-based benchmark
  comparisons against UNU current/voltage and UDMPF1 radial trajectory.
- Encoded UNU circuit and geometry: charging voltage `15 kV`, capacitance
  `30 uF`, inductance `110 nH`, resistance `12 mOhm`, anode radius `0.95 cm`,
  cathode radius `3.2 cm`, cathode-anode gap `2.25 cm`, anode length `16 cm`,
  and cathode length `25 cm`.
- Encoded motion targets: axial phase `0-2.5 us`, radial implosion
  `2.78-2.90 us`, pinch around `2.8 us`, background density `2.4e23 m^-3`,
  background pressure about `3.5 Torr`, axial sheath speed up to `90 km/s`,
  axial ion-temperature rise from `1` to `100 eV`, radial density about
  `1e24 m^-3`, and radial ion temperature about `1 keV`.
- Encoded parameter-law guidance: for large DPF devices, current saturates when
  increasing capacitance or decreasing inductance; increasing circuit voltage is
  more effective; and the anode-to-cathode radius ratio should be small. The
  PF-1000 `c` cases in the source are `1.4`, `1.8`, `2.2`, and `2.6`.
- Current corpus status after this ratchet:
  - coded KR target records: 39
  - unique coded KR target source files: 35
  - DPF-named markdown files represented by coded targets: 34 of 54
  - DPF-named markdown files closed by explicit review decisions: 12 of 54
  - total DPF-named markdown files review-closed: 46 of 54
  - unreviewed DPF-named markdown files: 8
- Updated triage counts after removing this source:
  - circuit waveform candidates: 0
  - phase timing candidates: 1
  - spatial density candidates: 0
  - spatial magnetic/EM candidates: 1
  - spatial temperature candidates: 4
  - neutron validation candidates: 2
  - uncertainty candidates: 0
- Verification:
  - `python3 -m py_compile src/dpf/validation/kr_targets.py src/dpf/validation/kr_corpus.py src/dpf/validation/__init__.py tests/test_kr_targets.py tests/test_kr_corpus.py`
  - `python3 -m pytest tests/test_kr_targets.py tests/test_kr_corpus.py -q`
    passed (`73 passed in 0.55s`)
  - Semantic/source audits passed:
    `source True`, `semantic True []`
  - Broad post-ratchet verification passed:
    `python3 -m pytest tests/test_quality_assessment.py tests/test_kr_targets.py tests/test_kr_corpus.py -q`
    (`124 passed in 0.66s`);
    `python3 -m pytest tests/test_mhd_physics_integration.py tests/test_mhd_numerical_fidelity.py tests/test_circuit_field_coupling.py tests/test_physics_fidelity.py tests/test_uncertainty_budget.py -q`
    (`87 passed, 3 skipped in 1.83s`); `git diff --check` clean.
- Remaining limit: the source strengthens macroscopic MHD motion, phase,
  temperature, and design-scaling targets, but it explicitly states that MHD
  cannot self-consistently resolve high-energy particle beams or neutron
  production. It also lacks digitized current/voltage traces, quantified error
  bars, density/temperature profile uncertainty, and neutron validation outputs.

### 2026-05-05: Demina/Gribkov DPF Material-Damage Irradiation Target

- Reviewed
  `KnowledgeReference/application-of-a-plasma-accelerator-of-the-dense-plasma-focus-type-in-simulation-of-radiation.md`.
- Added `demina_dpf_material_damage_targets()` as an application-response target.
  It is not core DPF machine validation.
- Encoded device and irradiation context: PF-5M bank energy `5 kJ`, PF-6 bank
  energy `7 kJ`, PF-1000 bank energy `1.2 MJ`, PF-1000 experimental stored
  energy about `600 kJ`, deuterium working gas at `470 Pa`, sample exposure
  power flux `1e7-1e10 W/cm2`, pulse duration `0.2-1 us`, `10` W/W-CFC pulses,
  and `5` CFC/SiC pulses.
- Encoded tungsten response: melting, evaporation, wavelike relief, nanoscale
  cellular structure at `1e10 W/cm2`, intergranular/transgranular microcracks
  above `1e8 W/cm2`, bubble size around `1 um`, microcrack penetration around
  `10 um`, and table-derived erosion depths including about `2.05 um` per pulse
  for the highest ion/plasma-stream condition.
- Encoded CFC/CFC-SiC response: W droplets/ridges on CFC, stronger evaporation
  when fibers are normal to the irradiated surface, lower erosion when fibers
  are parallel to the surface, CFC-8SiC evaporated layer `2.6 um` per shot at
  `1e9 W/cm2`, and CFC-40SiC `1.9 um` per shot.
- Encoded redeposition observations: Cu/O/Fe/Cr on W, Fe/Cr/Si/Cu on CFC-SiC,
  steel-holder sources for Fe/Cr, copper-anode source for Cu, and possible
  compounds `Fe2C`, `Fe5C2`, `Cu4Si`, and `(Cr,Fe)7C3`.
- Current corpus status after this ratchet:
  - coded KR target records: 40
  - unique coded KR target source files: 36
  - DPF-named markdown files represented by coded targets: 35 of 54
  - DPF-named markdown files closed by explicit review decisions: 12 of 54
  - total DPF-named markdown files review-closed: 47 of 54
  - unreviewed DPF-named markdown files: 7
- Updated triage counts after removing this source:
  - circuit waveform candidates: 0
  - phase timing candidates: 1
  - spatial density candidates: 0
  - spatial magnetic/EM candidates: 1
  - spatial temperature candidates: 3
  - neutron validation candidates: 1
  - uncertainty candidates: 0
- Verification:
  - `python3 -m py_compile src/dpf/validation/kr_targets.py src/dpf/validation/kr_corpus.py src/dpf/validation/__init__.py tests/test_kr_targets.py tests/test_kr_corpus.py`
  - `python3 -m pytest tests/test_kr_targets.py tests/test_kr_corpus.py -q`
    passed (`74 passed in 0.69s`)
  - Semantic/source audits passed:
    `source True`, `semantic True []`
  - Broad post-ratchet verification passed:
    `python3 -m pytest tests/test_quality_assessment.py tests/test_kr_targets.py tests/test_kr_corpus.py -q`
    (`125 passed in 0.61s`);
    `python3 -m pytest tests/test_mhd_physics_integration.py tests/test_mhd_numerical_fidelity.py tests/test_circuit_field_coupling.py tests/test_physics_fidelity.py tests/test_uncertainty_budget.py -q`
    (`87 passed, 3 skipped in 1.63s`); `git diff --check` clean.
- Remaining limit: the source can bound DPF-driven material erosion and
  redeposition, but it does not provide current/voltage waveforms, incident
  particle spectra, sample-distance tables by condition, same-shot plasma
  profiles, neutron observables, or uncertainty budgets.

### 2026-05-05: Unity Front-End Guide Review Decision

- Reviewed
  `KnowledgeReference/building-a-sci-fi-themed-dense-plasma-focus-simulation-front-end-in-unity.md`.
- Classified the source as `non_scientific_frontend_guide`.
- No validation target was added. The document is a Unity/URP/VFX Graph/UI,
  raymarching, data-ingestion, and WebSocket display tutorial. It is not a
  verified DPF physics source and does not provide KR-backed equations,
  experimental targets, diagnostics, or model-validation data.
- Current corpus status after this ratchet:
  - coded KR target records: 40
  - unique coded KR target source files: 36
  - DPF-named markdown files represented by coded targets: 35 of 54
  - DPF-named markdown files closed by explicit review decisions: 13 of 54
  - total DPF-named markdown files review-closed: 48 of 54
  - unreviewed DPF-named markdown files: 6
- Updated triage counts after removing this source:
  - circuit waveform candidates: 0
  - phase timing candidates: 1
  - spatial density candidates: 0
  - spatial magnetic/EM candidates: 0
  - spatial temperature candidates: 2
  - neutron validation candidates: 1
  - uncertainty candidates: 0
- Verification:
  - `python3 -m py_compile src/dpf/validation/kr_targets.py src/dpf/validation/kr_corpus.py src/dpf/validation/__init__.py tests/test_kr_targets.py tests/test_kr_corpus.py`
  - `python3 -m pytest tests/test_kr_targets.py tests/test_kr_corpus.py -q`
    passed (`74 passed in 0.48s`)
  - Semantic/source audits passed:
    `source True`, `semantic True []`
  - Broad post-ratchet verification passed:
    `python3 -m pytest tests/test_quality_assessment.py tests/test_kr_targets.py tests/test_kr_corpus.py -q`
    (`125 passed in 0.61s`);
    `python3 -m pytest tests/test_mhd_physics_integration.py tests/test_mhd_numerical_fidelity.py tests/test_circuit_field_coupling.py tests/test_physics_fidelity.py tests/test_uncertainty_budget.py -q`
    (`87 passed, 3 skipped in 1.57s`); `git diff --check` clean.
- Remaining limit: this source is review-closed only to keep the KR scientific
  queue accurate. It is not evidence for DPF scientific accuracy.

### 2026-05-05: Lee 2014 Radiative Lee-Model Review Target

- Reviewed `KnowledgeReference/lee-2014-plasma-focus-radiative-model.md`.
- Added `lee_2014_radiative_model_review_targets()` as a peer-reviewed
  equation/scope target for the radiative Lee model.
- Encoded 5-phase scope: axial snowplow, radial inward shock slug model,
  radial reflected shock, slow compression/pinch, expanded column, plus optional
  Type-2 `Phase 4a` anomalous-resistance extension.
- Encoded timing/model constraints: radial inward phase equation set
  `14,15,17,19`; reflected-shock equation set `34,35,36,37`; reflected-shock
  speed fraction `0.3`; axial phase ends when the current sheath reaches the
  anode end; radial inward phase ends when the shock reaches axis; pinch phase
  ends after one small-disturbance transit time.
- Encoded radiative-pinch terms: Joule heating, Spitzer resistance, Bennett
  temperature, Bremsstrahlung, line radiation, total `dQ/dt`, self-absorption,
  surface-emission transition, radiation collapse, deuterium critical current
  `1.6 MA`, and Ne/Ar critical current below `100 kA`.
- Current corpus status after this ratchet:
  - coded KR target records: 41
  - unique coded KR target source files: 37
  - DPF-named markdown files represented by coded targets: 36 of 54
  - DPF-named markdown files closed by explicit review decisions: 13 of 54
  - total DPF-named markdown files review-closed: 49 of 54
  - unreviewed DPF-named markdown files: 5
- Updated triage counts after removing this source:
  - circuit waveform candidates: 0
  - phase timing candidates: 0
  - spatial density candidates: 0
  - spatial magnetic/EM candidates: 0
  - spatial temperature candidates: 1
  - neutron validation candidates: 1
  - uncertainty candidates: 0
- Verification:
  - `python3 -m py_compile src/dpf/validation/kr_targets.py src/dpf/validation/kr_corpus.py src/dpf/validation/__init__.py tests/test_kr_targets.py tests/test_kr_corpus.py`
  - `python3 -m pytest tests/test_kr_targets.py tests/test_kr_corpus.py -q`
    passed (`75 passed in 0.74s`)
  - Semantic/source audits passed:
    `source True`, `semantic True []`
  - Broad post-ratchet verification passed:
    `python3 -m pytest tests/test_quality_assessment.py tests/test_kr_targets.py tests/test_kr_corpus.py -q`
    (`126 passed in 0.61s`);
    `python3 -m pytest tests/test_mhd_physics_integration.py tests/test_mhd_numerical_fidelity.py tests/test_circuit_field_coupling.py tests/test_physics_fidelity.py tests/test_uncertainty_budget.py -q`
    (`87 passed, 3 skipped in 1.66s`); `git diff --check` clean.
- Remaining limit: this is an equation/scope source, not an experimental
  validation packet. It lacks measured waveforms, shock/piston trajectories,
  radiated-power traces, profile diagnostics, neutron observables, and
  uncertainty budgets; the local extract also omits equations `51`, `52`, and
  `53`.

### 2026-05-05: Focus Fusion p-B11 Correction-Only Decision

- Reviewed
  `KnowledgeReference/2023-correction-to-focus-fusion-overview-of-progress-towards-p-b11-fusion-with-the.md`.
- Classified the source as `correction_only`.
- The one-page correction fixes the original Focus Fusion abstract's highest
  `n tau T` product to `3.4e20 keV-s/m3`.
- No new target was added. The corrected value is already encoded in
  `ff1_focus_fusion_plasmoid_targets()` from the canonical original Focus
  Fusion source.
- Current corpus status after this ratchet:
  - coded KR target records: 41
  - unique coded KR target source files: 37
  - DPF-named markdown files represented by coded targets: 36 of 54
  - DPF-named markdown files closed by explicit review decisions: 14 of 54
  - total DPF-named markdown files review-closed: 50 of 54
  - unreviewed DPF-named markdown files: 4
- Updated triage counts after removing this source:
  - circuit waveform candidates: 0
  - phase timing candidates: 0
  - spatial density candidates: 0
  - spatial magnetic/EM candidates: 0
  - spatial temperature candidates: 0
  - neutron validation candidates: 1
  - uncertainty candidates: 0
- Verification:
  - `python3 -m py_compile src/dpf/validation/kr_targets.py src/dpf/validation/kr_corpus.py src/dpf/validation/__init__.py tests/test_kr_targets.py tests/test_kr_corpus.py`
  - `python3 -m pytest tests/test_kr_targets.py tests/test_kr_corpus.py -q`
    passed (`75 passed in 0.49s`)
  - Semantic/source audits passed:
    `source True`, `semantic True []`
  - Broad post-ratchet verification passed:
    `python3 -m pytest tests/test_quality_assessment.py tests/test_kr_targets.py tests/test_kr_corpus.py -q`
    (`126 passed in 0.61s`);
    `python3 -m pytest tests/test_mhd_physics_integration.py tests/test_mhd_numerical_fidelity.py tests/test_circuit_field_coupling.py tests/test_physics_fidelity.py tests/test_uncertainty_budget.py -q`
    (`87 passed, 3 skipped in 1.60s`); `git diff --check` clean.
- Remaining limit: the correction notice adds no independent DPF validation
  data. It only fixes a scalar abstract value in a target already represented by
  the canonical source.

### 2026-05-05: McAlpine 2014 DPF/NRTA MCNP Application Target

- Reviewed
  `KnowledgeReference/monte-carlo-simulations-of-neutron-resonance-transmission-analysis-with-the-dense-plasma-focus.md`.
- Added `mcalpine_dpf_nrta_mcnp_targets()` as a downstream neutron-resonance
  transmission analysis application target, not a DPF plasma-validation target.
- Encoded DPF source context: LLNL DPF D-D `2.45 MeV` neutrons, yield about
  `1e7`, simulated pulse duration `20-60 ns`, generic DPF yield `1e4-1e13`
  neutrons in `10-100 ns`, deuterium working gas, optional DT context, and
  kinetic simulations used to inform desired yield/pinch length.
- Encoded MCNP/NRTA setup: monoenergetic isotropic point source, `3 cm`
  polyethylene moderator, detector volume `2 m` away, assumed `3He` detector
  with `1/v` absorption postprocessing, inspection object about `180 cm3`,
  Gaussian DPF pulse FWHM `20 ns`, conventional ENG trapezoid `4 us`, and
  `1e10` source particles per simulation.
- Encoded application results: TOF slightly broadens resonances but preserves
  locations; DPF resolves resonances not detectable with ENG; an ENG would take
  about a day for comparable resolvable measurement while DPF can do it in one
  pulse; depleted uranium, highly enriched uranium, plutonium, and lead were
  compared and distinguished.
- Updated the corpus triage test because the remaining unreviewed DPF-named
  files have no scientific category-marker hits.
- Current corpus status after this ratchet:
  - coded KR target records: 42
  - unique coded KR target source files: 38
  - DPF-named markdown files represented by coded targets: 37 of 54
  - DPF-named markdown files closed by explicit review decisions: 14 of 54
  - total DPF-named markdown files review-closed: 51 of 54
  - unreviewed DPF-named markdown files: 3
- Updated triage counts after removing this source:
  - circuit waveform candidates: 0
  - phase timing candidates: 0
  - spatial density candidates: 0
  - spatial magnetic/EM candidates: 0
  - spatial temperature candidates: 0
  - neutron validation candidates: 0
  - uncertainty candidates: 0
- Verification:
  - `python3 -m py_compile src/dpf/validation/kr_targets.py src/dpf/validation/kr_corpus.py src/dpf/validation/__init__.py tests/test_kr_targets.py tests/test_kr_corpus.py`
  - `python3 -m pytest tests/test_kr_targets.py tests/test_kr_corpus.py -q`
    passed (`76 passed in 0.49s`)
  - Semantic/source audits passed:
    `source True`, `semantic True []`
  - Broad post-ratchet verification passed:
    `python3 -m pytest tests/test_quality_assessment.py tests/test_kr_targets.py tests/test_kr_corpus.py -q`
    (`127 passed in 0.62s`);
    `python3 -m pytest tests/test_mhd_physics_integration.py tests/test_mhd_numerical_fidelity.py tests/test_circuit_field_coupling.py tests/test_physics_fidelity.py tests/test_uncertainty_budget.py -q`
    (`87 passed, 3 skipped in 1.65s`); `git diff --check` clean.
- Remaining limit: this report models DPF-enabled NRTA, not the DPF plasma. It
  assumes a monoenergetic isotropic point source, postprocesses detector
  response, ignores room scatter/passive background, and explicitly calls for
  experiments, minimum-yield analysis, room geometry, and direct detector-
  response modeling.

### 2026-05-05: DimLifePF96 Empty Extraction Decision

- Reviewed
  `KnowledgeReference/dimensions-and-lifetime-of-the-plasma-focus-pinch-plasma-science-ieee-transactions-on-2.md`.
- Classified the source as `insufficient_extractable_validation_data`.
- No validation target was added. The local markdown contains only a
  title/source header and page stub, so pinch dimensions and lifetime cannot be
  extracted under the KR-only rule.
- Current corpus status after this ratchet:
  - coded KR target records: 42
  - unique coded KR target source files: 38
  - DPF-named markdown files represented by coded targets: 37 of 54
  - DPF-named markdown files closed by explicit review decisions: 15 of 54
  - total DPF-named markdown files review-closed: 52 of 54
  - unreviewed DPF-named markdown files: 2
- Updated triage counts after removing this source:
  - circuit waveform candidates: 0
  - phase timing candidates: 0
  - spatial density candidates: 0
  - spatial magnetic/EM candidates: 0
  - spatial temperature candidates: 0
  - neutron validation candidates: 0
  - uncertainty candidates: 0
- Verification:
  - `python3 -m py_compile src/dpf/validation/kr_targets.py src/dpf/validation/kr_corpus.py src/dpf/validation/__init__.py tests/test_kr_targets.py tests/test_kr_corpus.py`
  - `python3 -m pytest tests/test_kr_targets.py tests/test_kr_corpus.py -q`
    passed (`76 passed in 0.59s`)
  - Broad post-ratchet verification passed:
    `python3 -m pytest tests/test_quality_assessment.py tests/test_kr_targets.py tests/test_kr_corpus.py -q`
    (`127 passed in 0.73s`);
    `python3 -m pytest tests/test_mhd_physics_integration.py tests/test_mhd_numerical_fidelity.py tests/test_circuit_field_coupling.py tests/test_physics_fidelity.py tests/test_uncertainty_budget.py -q`
    (`87 passed, 3 skipped in 1.96s`); `git diff --check` clean.
- Remaining limit: this source needs re-ingestion from the original PDF before
  any KR-only pinch dimension, lifetime, or diagnostic target can be extracted.

### 2026-05-05: DPF-Bi-RRT Acronym-Collision Decision

- Reviewed
  `KnowledgeReference/dpf-bi-rrt-an-improved-path-planning-algorithm-for-complex-3d-environments-with-adaptive-sampling.md`.
- Classified the source as `non_dpf_acronym_collision`.
- In this IEEE Access path-planning paper, DPF means Dual Potential Field in
  the `DPF-Bi-RRT*` algorithm for autonomous aerial vehicle navigation. It is
  unrelated to Dense Plasma Focus physics.
- No validation target was added.
- Current corpus status after this ratchet:
  - coded KR target records: 42
  - unique coded KR target source files: 38
  - DPF-named markdown files represented by coded targets: 37 of 54
  - DPF-named markdown files closed by explicit review decisions: 16 of 54
  - total DPF-named markdown files review-closed: 53 of 54
  - unreviewed DPF-named markdown files: 1
- Updated triage counts after removing this source:
  - circuit waveform candidates: 0
  - phase timing candidates: 0
  - spatial density candidates: 0
  - spatial magnetic/EM candidates: 0
  - spatial temperature candidates: 0
  - neutron validation candidates: 0
  - uncertainty candidates: 0
- Verification:
  - `python3 -m py_compile src/dpf/validation/kr_targets.py src/dpf/validation/kr_corpus.py src/dpf/validation/__init__.py tests/test_kr_targets.py tests/test_kr_corpus.py`
  - `python3 -m pytest tests/test_kr_targets.py tests/test_kr_corpus.py -q`
    passed (`76 passed in 0.54s`)
  - Broad post-ratchet verification passed:
    `python3 -m pytest tests/test_quality_assessment.py tests/test_kr_targets.py tests/test_kr_corpus.py -q`
    (`127 passed in 0.66s`);
    `python3 -m pytest tests/test_mhd_physics_integration.py tests/test_mhd_numerical_fidelity.py tests/test_circuit_field_coupling.py tests/test_physics_fidelity.py tests/test_uncertainty_budget.py -q`
    (`87 passed, 3 skipped in 1.73s`); `git diff --check` clean.
- Remaining limit: none for Dense Plasma Focus; this file is outside the
  scientific domain.

### 2026-05-05: DPF Simulator Software-Performance Summary Decision

- Reviewed
  `KnowledgeReference/optimization-and-development-of-a-dense-plasma-focus-simulator.md`.
- Classified the source as `non_scientific_software_performance_summary`.
- No validation target was added. The two-page local source summarizes DPF
  simulator software architecture and performance claims: GUI, solvers,
  ML-control, visualization, Metal GPU acceleration, CPU utilization, memory,
  and FPS. It provides no verified DPF physics equations, diagnostics,
  calibration data, validation targets, or uncertainty data.
- Updated corpus tests for the completed review state: all DPF-named markdown
  files are now reviewed, but `kr_corpus_review_status()["passed"]` remains
  false because validation coverage and same-scope predictive evidence remain
  incomplete.
- Current corpus status after this ratchet:
  - coded KR target records: 42
  - unique coded KR target source files: 38
  - DPF-named markdown files represented by coded targets: 37 of 54
  - DPF-named markdown files closed by explicit review decisions: 17 of 54
  - total DPF-named markdown files review-closed: 54 of 54
  - unreviewed DPF-named markdown files: 0
- Updated triage status:
  - `kr_unreviewed_dpf_source_triage()` passes.
  - unreviewed DPF-named markdown files: 0
  - all tracked scientific category counts: 0
- Remaining KR target coverage blockers after full DPF-named corpus review:
  `circuit_waveform`, `phase_timing`, and `spatial_temperature`; same-scope
  predictive readiness remains false.
- Verification:
  - `python3 -m py_compile src/dpf/validation/kr_targets.py src/dpf/validation/kr_corpus.py src/dpf/validation/__init__.py tests/test_kr_targets.py tests/test_kr_corpus.py`
  - `python3 -m pytest tests/test_kr_targets.py tests/test_kr_corpus.py -q`
    passed (`76 passed in 0.54s`)
  - Semantic/source audits passed:
    `source True`, `semantic True []`
  - Broad post-ratchet verification passed:
    `python3 -m pytest tests/test_quality_assessment.py tests/test_kr_targets.py tests/test_kr_corpus.py -q`
    (`127 passed in 0.65s`);
    `python3 -m pytest tests/test_mhd_physics_integration.py tests/test_mhd_numerical_fidelity.py tests/test_circuit_field_coupling.py tests/test_physics_fidelity.py tests/test_uncertainty_budget.py -q`
    (`87 passed, 3 skipped in 1.80s`); `git diff --check` clean.
- Remaining limit: all DPF-named local markdown files have now been reviewed or
  target-extracted, but the repository is still not a validated end-to-end
  predictive DPF simulation tool. The remaining blockers are validation depth
  and implementation fidelity, not unreviewed DPF-named KR files.

### 2026-05-05: Corpus-Review Completion Plan Update

- Reviewed the post-corpus status after all DPF-named markdown files reached
  closure.
- Updated `kr_corpus_review_status()["next_ratcheting_steps"]` so it no longer
  asks for unreviewed-source extraction after the queue is empty.
- The code now reports this local plan:
  - DPF-named KnowledgeReference markdown review is complete.
  - Close remaining target coverage blockers: `circuit_waveform`,
    `phase_timing`, and `spatial_temperature`.
  - Promote one same-scope validation packet by adding KR-backed circuit,
    phase, spatial, neutron, and uncertainty evidence for a single
    device/shot/scope, or keep readiness blocked when KR lacks those
    observables.
- Verification:
  - `python3 -m py_compile src/dpf/validation/kr_corpus.py tests/test_kr_corpus.py`
  - `python3 -m pytest tests/test_kr_targets.py tests/test_kr_corpus.py -q`
    passed (`76 passed in 0.50s`)
  - `python3 -m pytest tests/test_quality_assessment.py tests/test_kr_targets.py tests/test_kr_corpus.py -q`
    passed (`127 passed in 0.61s`)
  - `git diff --check` clean.
- Remaining limit: source review is no longer the ratchet. The next ratchet
  must improve validation evidence or explicitly preserve readiness blockers
  where KR data is absent.

### 2026-05-05: Same-Scope Closure-Path Report

- Reviewed same-scope target status after full DPF-named corpus closure.
- Added `widest_available_scope` and `next_same_scope_steps` to
  `kr_validation_same_scope_target_report()`.
- The report now distinguishes:
  - `best_available_scope`: MJOLNIR currently has fewer total blockers but is
    missing several required groups.
  - `widest_available_scope`: PF-1000 full-energy
    `pf1000_full_energy_2007_gribkov_scholz` has all required groups present
    but remains incomplete because most groups are partial.
- PF-1000 full-energy partial blockers: `circuit_waveform`,
  `neutron_anisotropy`, `neutron_detector_response`, `neutron_spectrum`,
  `neutron_timing`, `phase_timing`, `spatial_magnetic_or_em`,
  `spatial_temperature`, and `uncertainty`.
- The code-level next step now says to use the widest same-scope packet as the
  closure path and keep predictive readiness blocked until those partial groups
  have digitized traces, uncertainty, and same-shot diagnostic support.
- Verification:
  - `python3 -m py_compile src/dpf/validation/kr_targets.py tests/test_kr_targets.py`
  - `python3 -m pytest tests/test_kr_targets.py tests/test_kr_corpus.py -q`
    passed (`76 passed in 0.62s`)
  - `python3 -m pytest tests/test_quality_assessment.py tests/test_kr_targets.py tests/test_kr_corpus.py -q`
    passed (`127 passed in 0.61s`)
  - `git diff --check` clean.
- Remaining limit: the KR corpus, as currently extracted, gives a broad
  PF-1000 packet but not a complete predictive validation packet. Closing it
  requires digitized waveform, phase, spatial, neutron, and uncertainty evidence
  for the same PF-1000 scope, or explicit permanent blockers where KR lacks
  those observables.

### 2026-05-05: PF-1000 Closure-Blocker Checklist

- Added `closure_blockers` and `closure_blocker_groups` to
  `kr_validation_same_scope_target_report()`.
- The PF-1000 full-energy scope now reports the exact blocker checklist for
  each partial group rather than only listing group names.
- Encoded blocker examples:
  - `circuit_waveform`: `digitized_current_trace_points`
  - `phase_timing`: `radial_transit_start_and_end_times`
  - `spatial_temperature`: `direct_experimental_temperature_diagnostic`
  - `neutron_detector_response`:
    `neutron_field_transport_or_room_scatter_response_model`
  - `uncertainty`: `fast_ion_distribution_uncertainty`
- Review outcome: the current closure path is not more validated than before;
  it is more auditable. The plan now has code-level blockers that can be used as
  the extraction checklist or as explicit KR-absence reasons.
- Verification:
  - `python3 -m py_compile src/dpf/validation/kr_targets.py tests/test_kr_targets.py`
  - `python3 -m pytest tests/test_kr_targets.py::test_kr_validation_same_scope_target_report_requires_one_scope -q`
    passed (`1 passed in 0.53s`)
  - `python3 -m pytest tests/test_kr_targets.py tests/test_kr_corpus.py -q`
    passed (`76 passed in 0.49s`)
  - `python3 -m pytest tests/test_quality_assessment.py tests/test_kr_targets.py tests/test_kr_corpus.py -q`
    passed (`127 passed in 0.64s`)
  - `python3 -m pytest tests/test_mhd_physics_integration.py tests/test_mhd_numerical_fidelity.py tests/test_circuit_field_coupling.py tests/test_physics_fidelity.py tests/test_uncertainty_budget.py -q`
    passed (`87 passed, 3 skipped in 1.64s`)
  - `git diff --check` clean.
- Remaining limit: the missing observables are still missing from the coded
  target packet. The next ratchet must either extract more same-scope PF-1000
  evidence from already reviewed KR line windows or mark each blocker as a
  KR-absence gate that predictive readiness cannot pass.

### 2026-05-05: Broad DPF-Content Corpus Queue

- Updated corpus review accounting from filename-only DPF relevance to
  filename-or-strong-content DPF relevance.
- Strong content markers: `dense plasma focus`, `plasma focus`, `PF-1000`,
  `PF1000`, `PF 1000`, `MJOLNIR`, `Mather-type`, and `Filippov`.
- Current inventory:
  - 827 total local source files
  - 398 markdown files
  - 396 JSON files
  - 54 DPF-named markdown files
  - 94 DPF-content markdown files
  - 96 DPF-relevant markdown files by filename or content
- Current review status:
  - 55 of 96 DPF-relevant markdown files are review-closed by coded target or
    explicit decision.
  - 41 DPF-relevant markdown files remain open.
- Review outcome: the earlier DPF-named queue is complete, but that is not the
  same as complete DPF-relevant corpus review. The plan is now corrected to
  process the remaining content-hit files before claiming source closure.
- Verification:
  - `python3 -m py_compile src/dpf/validation/kr_corpus.py tests/test_kr_corpus.py`
  - `python3 -m pytest tests/test_kr_corpus.py -q` passed (`4 passed in 1.37s`)
  - `python3 -m pytest tests/test_kr_targets.py tests/test_kr_corpus.py -q`
    passed (`76 passed in 1.24s`)
- Remaining limit: the 41 broad content-hit files still need review. Each must
  become either a coded target or an explicit decision before source closure is
  honest.

### 2026-05-05: Broad DPF-Content Review Wave 1

- Added explicit review decisions for 20 of the 41 newly exposed broad
  DPF-content markdown files.
- Closed categories:
  - duplicate FAETON-I and hybrid X-pinch extractions
  - non-DPF/reference-only Z-pinch papers
  - general Z-pinch snowplow/scaling papers without DPF device targets
  - educational, software-manual, image-index, and application/materials files
    that do not contain DPF machine-validation observables
- Current status:
  - 75 of 96 DPF-relevant markdown files are review-closed.
  - 21 DPF-relevant markdown files remain open.
- Verification:
  - `python3 -m py_compile src/dpf/validation/kr_corpus.py`
  - `python3 -m pytest tests/test_kr_corpus.py -q` passed (`4 passed in 1.09s`)
- Remaining limit: the remaining 21 files are more likely to contain useful DPF
  science or diagnostics, so the next wave must review them individually rather
  than bulk-closing them as reference-only.

### 2026-05-05: Broad DPF-Content Review Closure

- Added two coded targets from the broad content queue:
  - `mjolnir_first_experiments_2021_offermann`
  - `uofsi_argon_temperature_thesis_2020`
- Added explicit review decisions for the remaining broad content-hit files.
- Current source-review status:
  - 96 of 96 DPF-relevant markdown files are review-closed.
  - The unreviewed DPF-relevant queue is empty.
- Remaining code-reported validation blockers:
  - `circuit_waveform`
  - `phase_timing`
  - `spatial_temperature`
  - `uncertainty`
- Review outcome: the source-review question is now answered more honestly than
  the prior filename-only pass. The DPF-relevant markdown corpus has been
  reviewed into either coded target records or explicit non-target/duplicate/
  context decisions. The project remains blocked by validation evidence quality,
  not by unreviewed DPF-relevant markdown sources.
- Verification:
  - `python3 -m py_compile src/dpf/validation/kr_corpus.py src/dpf/validation/kr_targets.py src/dpf/validation/__init__.py tests/test_kr_corpus.py tests/test_kr_targets.py`
  - `python3 -m pytest tests/test_kr_corpus.py -q` passed (`4 passed in 1.13s`)
  - `python3 -m pytest tests/test_kr_targets.py -q` passed (`74 passed in 0.46s`)
  - `python3 -m pytest tests/test_quality_assessment.py tests/test_kr_targets.py tests/test_kr_corpus.py -q`
    passed (`129 passed in 1.29s`)
  - `python3 -m pytest tests/test_mhd_physics_integration.py tests/test_mhd_numerical_fidelity.py tests/test_circuit_field_coupling.py tests/test_physics_fidelity.py tests/test_uncertainty_budget.py -q`
    passed (`87 passed, 3 skipped in 5.88s`)
  - `git diff --check` clean.
- Remaining limit: no same-scope validation packet passes. The next phase must
  either improve same-scope evidence or make the final product explicitly block
  high-fidelity claims when KR lacks digitized traces, phase endpoints, spatial
  temperature/density/B validation, detector response, and propagated
  uncertainty.

### 2026-05-05: Source-Review Gap Closure In Readiness Reports

- Added a `kr_source_review` entry to `scientific_accuracy_gap_report()`.
- Current status: `kr_source_review` is `supported` because the
  DPF-relevant markdown queue is empty.
- Updated the `kr_target_coverage` blocker so it points at the widest
  same-scope closure path when target evidence remains partial.
- Review outcome: the app/readiness layer now distinguishes the closed source
  review from the still-open validation-evidence problem. This prevents future
  status summaries from saying "source review remains" when the real blockers
  are digitized traces, same-scope phase/spatial evidence, detector response,
  and UQ.
- Verification:
  - `python3 -m py_compile src/dpf/validation/quality_assessment.py tests/test_quality_assessment.py tests/test_mhd_physics_integration.py`
  - `python3 -m pytest tests/test_quality_assessment.py::TestQualityAssessment::test_scientific_accuracy_gap_report_lists_remaining_work tests/test_mhd_physics_integration.py::test_predictive_readiness_exported_and_blocks_unvalidated_claims -q`
    passed (`2 passed in 1.16s`)
  - `python3 -m pytest tests/test_quality_assessment.py -q` passed
    (`51 passed in 2.38s`)
  - `python3 -m pytest tests/test_quality_assessment.py tests/test_kr_targets.py tests/test_kr_corpus.py -q`
    passed (`129 passed in 3.13s`)
  - `python3 -m pytest tests/test_mhd_physics_integration.py tests/test_mhd_numerical_fidelity.py tests/test_circuit_field_coupling.py tests/test_physics_fidelity.py tests/test_uncertainty_budget.py -q`
    passed (`87 passed, 3 skipped in 7.50s`)
  - `git diff --check` clean.
- Remaining limit: high-fidelity readiness is still false. This is now clearly
  because target evidence is incomplete, not because the KR source queue is
  unreviewed.

### 2026-05-05: Same-Scope Uncertainty Packet Gate

- Tightened `uncertainty_evidence_from_result()` so a complete UQ component set
  must share one `validation_scope`.
- Cross-scope uncertainty component packets now fail with
  `same_scope_uncertainty_packet`.
- Review outcome: UQ can no longer be assembled from unrelated scopes to satisfy
  high-fidelity readiness.
- Verification:
  - `python3 -m py_compile src/dpf/validation/uncertainty_budget.py tests/test_uncertainty_budget.py`
  - `python3 -m pytest tests/test_uncertainty_budget.py -q` passed
    (`10 passed in 0.91s`)
  - `python3 -m pytest tests/test_quality_assessment.py::TestQualityAssessment::test_high_fidelity_readiness_can_pass_with_complete_evidence_packet tests/test_uncertainty_budget.py::test_complete_uncertainty_components_must_share_validation_scope -q`
    passed (`2 passed in 0.65s`)
- Remaining limit: real runs still need a same-scope uncertainty packet with
  experimental, input, numerical, model-form, shot-to-shot, propagated,
  acceptance-rule, and KR-target components.

### 2026-05-05: Same-Scope Physics-Fidelity Packet Gate

- Tightened `physics_fidelity_evidence_from_result()` so a complete
  high-fidelity physics-effect packet must share one `validation_scope`.
- Cross-scope physics-effect packets now fail with
  `same_scope_physics_packet`.
- Review outcome: high-fidelity physics readiness can no longer be assembled
  from unrelated validation scopes for EOS/conductivity, ionization,
  two-temperature partition, radiation transport, ablation/impurity mixing,
  Hall/FLR/kinetic effects, 3D instabilities, flashover, restrike, and
  beam-target coupling.
- Verification:
  - `python3 -m py_compile src/dpf/validation/physics_fidelity.py tests/test_physics_fidelity.py`
  - `python3 -m pytest tests/test_physics_fidelity.py -q` passed
    (`7 passed in 1.32s`)
  - `python3 -m pytest tests/test_quality_assessment.py::TestQualityAssessment::test_high_fidelity_readiness_can_pass_with_complete_evidence_packet tests/test_physics_fidelity.py::test_complete_physics_effects_must_share_validation_scope tests/test_uncertainty_budget.py::test_complete_uncertainty_components_must_share_validation_scope -q`
    passed (`3 passed in 0.85s`)
  - `python3 -m pytest tests/test_mhd_physics_integration.py tests/test_mhd_numerical_fidelity.py tests/test_circuit_field_coupling.py tests/test_physics_fidelity.py tests/test_uncertainty_budget.py -q`
    passed (`89 passed, 3 skipped in 7.71s`)
- Remaining limit: real runs still need one KR-backed claim scope whose
  required high-fidelity physics effects are implemented and validated, or
  explicitly bounded out for that same scope.

### 2026-05-05: Same-Scope Circuit/Field-Coupling Packet Gate

- Tightened `field_coupling_evidence_from_result()` so a complete
  field-coupling component packet must share one `validation_scope`.
- Cross-scope field-coupling packets now fail with
  `same_scope_field_coupling_packet`.
- Updated `scientific_accuracy_gap_report()` so a complete-but-cross-scope
  field-coupling packet is `blocked`, not merely `partial`.
- Review outcome: MHD-mode current prediction cannot be promoted by combining
  inductance, dL/dt/back-EMF, Poynting power, circuit energy, transition
  metadata, and KR experimental comparison evidence from unrelated scopes.
- Verification:
  - `python3 -m py_compile src/dpf/validation/circuit_field_coupling.py src/dpf/validation/quality_assessment.py tests/test_circuit_field_coupling.py`
  - `python3 -m pytest tests/test_circuit_field_coupling.py -q` passed
    (`12 passed in 0.85s`)
  - `python3 -m pytest tests/test_quality_assessment.py::TestQualityAssessment::test_high_fidelity_readiness_can_pass_with_complete_evidence_packet tests/test_circuit_field_coupling.py::test_complete_field_coupling_components_must_share_validation_scope tests/test_physics_fidelity.py::test_complete_physics_effects_must_share_validation_scope tests/test_uncertainty_budget.py::test_complete_uncertainty_components_must_share_validation_scope -q`
    passed (`4 passed in 1.05s`)
  - `python3 -m pytest tests/test_quality_assessment.py tests/test_mhd_physics_integration.py tests/test_mhd_numerical_fidelity.py tests/test_circuit_field_coupling.py tests/test_physics_fidelity.py tests/test_uncertainty_budget.py -q`
    passed (`141 passed, 3 skipped in 9.54s`)
- Remaining limit: real runs still need same-scope validated circuit/field
  coupling, not only exported coupling signals or code-verification energy
  identities.

### 2026-05-05: Global High-Fidelity Scope-Alignment Gate

- Added `same_scope_high_fidelity_claim` to
  `scientific_accuracy_gap_report()`.
- The new gap requires KR target coverage, field-coupling,
  physics-fidelity, and uncertainty packets to share at least one
  `validation_scope`.
- Complete same-scope synthetic evidence remains high-fidelity ready.
- Complete but cross-scope support packets are now blocked at the
  high-fidelity claim level even when each packet passes internally.
- Verification:
  - `python3 -m py_compile src/dpf/validation/quality_assessment.py tests/test_quality_assessment.py`
  - `python3 -m pytest tests/test_quality_assessment.py::TestQualityAssessment::test_scientific_accuracy_gap_report_lists_remaining_work tests/test_quality_assessment.py::TestQualityAssessment::test_high_fidelity_readiness_can_pass_with_complete_evidence_packet tests/test_quality_assessment.py::TestQualityAssessment::test_high_fidelity_scope_alignment_blocks_cross_scope_packets -q`
    passed (`3 passed in 1.30s`)
  - `python3 -m pytest tests/test_quality_assessment.py tests/test_mhd_physics_integration.py tests/test_mhd_numerical_fidelity.py tests/test_circuit_field_coupling.py tests/test_physics_fidelity.py tests/test_uncertainty_budget.py -q`
    passed (`142 passed, 3 skipped in 9.86s`)
- Remaining limit: this enforces claim consistency. It does not create the
  missing same-scope experimental data needed for a real DPF validation packet.

### 2026-05-05: Global Scope Gate Extended To Tier Evidence

- Extended `same_scope_high_fidelity_claim` so it also requires source
  authority, circuit validation, snowplow validation, spatial validation,
  neutron validation, and neutron detector-response evidence to share the same
  `validation_scope`.
- The global gate no longer checks only the support packets. It now aligns the
  actual tier evidence used for the predictive claim with the KR target packet,
  field-coupling packet, physics-fidelity packet, and uncertainty packet.
- Verification:
  - `python3 -m py_compile src/dpf/validation/quality_assessment.py tests/test_quality_assessment.py`
  - `python3 -m pytest tests/test_quality_assessment.py::TestQualityAssessment::test_scientific_accuracy_gap_report_lists_remaining_work tests/test_quality_assessment.py::TestQualityAssessment::test_high_fidelity_readiness_can_pass_with_complete_evidence_packet tests/test_quality_assessment.py::TestQualityAssessment::test_high_fidelity_scope_alignment_blocks_cross_scope_packets -q`
    passed (`3 passed in 1.31s`)
  - `python3 -m pytest tests/test_quality_assessment.py tests/test_mhd_physics_integration.py tests/test_mhd_numerical_fidelity.py tests/test_circuit_field_coupling.py tests/test_physics_fidelity.py tests/test_uncertainty_budget.py -q`
    passed (`142 passed, 3 skipped in 9.78s`)
- Remaining limit: this is still a gate, not data. The live gap report remains
  blocked because no real KR-backed DPF run provides the full same-scope packet.

### 2026-05-05: Same-Scope MHD Numerical-Fidelity Packet Gate

- Added `verification_scope` metadata to the MHD numerical evidence builders
  for cylindrical convergence, resistive diffusion, backend parity, MHD phase
  scope limits, and circuit-coupled energy verification.
- Tightened `mhd_numerical_fidelity_evidence_from_result()` so a complete
  Tier-3 numerical-fidelity packet must share one verification scope.
- Cross-scope numerical verification bundles now fail with
  `same_scope_mhd_numerical_packet`.
- Updated `scientific_accuracy_gap_report()` so a complete-but-cross-scope MHD
  numerical packet is `blocked`, not `partial`.
- Verification:
  - `python3 -m py_compile src/dpf/validation/mhd_numerical_fidelity.py src/dpf/validation/circuit_field_coupling.py src/dpf/validation/quality_assessment.py tests/test_mhd_numerical_fidelity.py`
  - `python3 -m pytest tests/test_mhd_numerical_fidelity.py -q` passed
    (`21 passed in 1.09s`)
  - `python3 -m pytest tests/test_quality_assessment.py tests/test_mhd_physics_integration.py tests/test_mhd_numerical_fidelity.py tests/test_circuit_field_coupling.py tests/test_physics_fidelity.py tests/test_uncertainty_budget.py -q`
    passed (`143 passed, 3 skipped in 10.25s`)
- Remaining limit: this verifies numerical-packet consistency only. It does not
  validate DPF late-pinch physics or provide same-scope experimental closure.

### 2026-05-05: Same-Scope Predictive-Readiness Tier Gate

- Tightened `predictive_readiness_report()` so tiers 1, 2, 4, and 5 must share
  one `validation_scope` before the lower `predictive_ready` label can pass.
- The guard covers circuit waveform validation, snowplow phase/timing
  validation, spatial DPF validation, and same-scope neutron validation.
- Cross-scope tier evidence now fails with `Predictive validation scope
  alignment` in `missing_evidence`.
- Verification:
  - `python3 -m py_compile src/dpf/validation/quality_assessment.py tests/test_quality_assessment.py`
  - `python3 -m pytest tests/test_quality_assessment.py::TestQualityAssessment::test_predictive_readiness_passes_only_with_all_required_tiers tests/test_quality_assessment.py::TestQualityAssessment::test_predictive_readiness_requires_one_validation_scope tests/test_quality_assessment.py::TestQualityAssessment::test_high_fidelity_readiness_requires_gap_closure tests/test_quality_assessment.py::TestQualityAssessment::test_high_fidelity_readiness_can_pass_with_complete_evidence_packet -q`
    passed (`4 passed in 1.11s`)
  - `python3 -m pytest tests/test_quality_assessment.py tests/test_mhd_physics_integration.py tests/test_mhd_numerical_fidelity.py tests/test_circuit_field_coupling.py tests/test_physics_fidelity.py tests/test_uncertainty_budget.py -q`
    passed (`144 passed, 3 skipped in 10.45s`)
- Remaining limit: no real run currently has same-scope tier evidence across
  circuit, snowplow, spatial, and neutron validation.

### 2026-05-05: Machine-Readable KR Data-Availability Blockers

- Added `data_availability` and `required_data_to_complete` to each
  `closure_blockers` record emitted by
  `kr_validation_same_scope_target_report()`.
- Missing same-scope groups are now marked
  `absent_from_same_scope_targets`.
- Partial PF-1000 closure groups are now marked
  `partial_only_in_same_scope_targets` with the exact required data list copied
  into `required_data_to_complete`.
- Verification:
  - `python3 -m py_compile src/dpf/validation/kr_targets.py tests/test_kr_targets.py`
  - `python3 -m pytest tests/test_kr_targets.py::test_kr_validation_same_scope_target_report_requires_one_scope -q`
    passed (`1 passed in 0.55s`)
  - `python3 -m pytest tests/test_kr_targets.py tests/test_kr_corpus.py -q`
    passed (`78 passed in 1.13s`)
- Remaining limit: the report now says more clearly what is absent or partial
  in the reviewed KR targets. It does not manufacture the absent digitized
  traces, uncertainties, detector response, or same-shot diagnostics.

### 2026-05-05: Verification Sweep Checkpoint

- Validation/KR/readiness regression sweep passed:
  - `python3 -m pytest tests/test_quality_assessment.py tests/test_mhd_physics_integration.py tests/test_mhd_numerical_fidelity.py tests/test_circuit_field_coupling.py tests/test_physics_fidelity.py tests/test_uncertainty_budget.py tests/test_kr_targets.py tests/test_kr_corpus.py -q`
  - Result: `222 passed, 3 skipped in 11.85s`
- `git diff --check` is clean.
- Full `python3 -m pytest -q` was attempted and aborted during collection while
  importing `dpf.metal.mlx_device` from `tests/test_amr_mlx.py`. The traceback
  indicates a fatal Python abort in the MLX import path before assertions ran.
- Claim-surface scan found current blocker language in README/SCOPE/JOSS-style
  files and tests; remaining positive hits are historical planning/debate docs
  or non-DPF backend production notes.
- Current live gap report:
  - KR source review: supported, 96/96 DPF-relevant markdown files closed.
  - KR target coverage: partial; PF-1000 full-energy remains the widest closure
    path.
  - Predictive and high-fidelity readiness: blocked by missing same-scope
    validation evidence, physics-fidelity evidence, field coupling, UQ, and
    Tier-3/Tier-4/Tier-5 validation evidence.

### 2026-05-06: User Decisions And Next Scientific-Closure Plan

- User decisions captured:
  - New source-of-truth material is allowed only after an AI researches and
    provides a link/source document and the user acquires the correct document.
  - Manual digitization from existing or newly acquired source documents is
    allowed, but the project needs a reproducible one-for-one verification
    method before using digitized data for validation claims.
  - Device choice is secondary; the physics closure matters.
  - Product target is a full high-fidelity neutron-predictive DPF simulator.
  - Scientific closure is priority 1; product hardening is priority 2.
- Next plan:
  1. Build a digitization provenance and verification workflow: source file
     hash, figure/page/axis metadata, calibration points, extracted point
     arrays, reviewer check, and residual/error report against the source
     image or table.
  2. Turn the current closure blockers into a source-acquisition queue grouped
     by physics need: circuit waveform, phase timing, spatial density/B/T,
     neutron timing/spectrum/anisotropy, detector response, and uncertainty.
  3. Research candidate source documents for each queue item and provide links
     for user acquisition before adding anything to `KnowledgeReference`.
  4. After user acquisition, ingest the document locally, review it under the
     KR-only rule, extract typed targets, and update the same-scope closure
     report.
  5. Only after same-scope data exists, implement or validate the required
     physics closures: EOS/conductivity, ionization, two-temperature energy
     partition, radiation transport/opacities, impurity/ablation, Hall/FLR/PIC
     or bounded kinetic treatment, 3D instability scope, flashover/startup,
     restrike/anomalous resistance, and beam-target neutron coupling.
- Working assumption: until new acquired documents or verified digitized data
  close the evidence gaps, the code must keep both predictive and high-fidelity
  readiness blocked.

### 2026-05-06: Digitization Gate And Source-Acquisition Queue

- Added a reusable one-for-one digitization audit in
  `src/dpf/validation/digitization.py`.
- The audit requires:
  - local `KnowledgeReference/` source path
  - matching source SHA-256 hash
  - figure image path and hash for figure extractions
  - figure/table ID and page
  - x/y axis calibration with units and residual limits
  - extracted series arrays with units
  - overlay residual evidence for figure data
  - at least one accepted independent review
- The audit fails closed on `KnowledgeReference` path traversal and malformed
  review-count metadata.
- Added `scientific_closure_source_acquisition_queue()` in
  `src/dpf/validation/source_acquisition.py`.
- The queue is built from live blockers in
  `kr_validation_same_scope_target_report()` and keeps candidate sources
  separate from source-of-truth evidence.
- Added `docs/SCIENTIFIC_CLOSURE_SOURCE_QUEUE.md` as the actionable acquisition
  list for the user. It records candidate DOI links for:
  - PF-1000 circuit waveform and phase timing
  - direct or bounded spatial temperature
  - same-scope uncertainty
  - neutron anisotropy
  - neutron detector response
  - neutron spectrum
  - neutron timing
  - spatial magnetic/EM validation
- Corrected the Zr/Be activation-detector candidate DOI to
  `10.1016/j.nima.2020.164830`.
- Verification:
  - `python3 -m py_compile src/dpf/validation/digitization.py src/dpf/validation/source_acquisition.py src/dpf/validation/__init__.py tests/test_digitization.py`
  - `python3 -m pytest tests/test_digitization.py -q` passed
    (`5 passed in 0.75s`)
  - `python3 -m pytest tests/test_digitization.py tests/test_kr_targets.py tests/test_kr_corpus.py tests/test_quality_assessment.py::TestQualityAssessment::test_scientific_accuracy_gap_report_lists_remaining_work -q`
    passed (`84 passed in 1.62s`)
  - `python3 -m pytest tests/test_quality_assessment.py tests/test_mhd_physics_integration.py tests/test_mhd_numerical_fidelity.py tests/test_circuit_field_coupling.py tests/test_physics_fidelity.py tests/test_uncertainty_budget.py tests/test_kr_targets.py tests/test_kr_corpus.py tests/test_digitization.py -q`
    passed (`227 passed, 3 skipped in 12.22s`)
  - `git diff --check` is clean.
- Remaining limit: this closes the process and provenance gap, not the
  scientific evidence gap. The project still needs user-acquired local sources
  and/or verified digitization for same-scope current traces, phase timing,
  temperature, magnetic/EM fields, neutron timing/spectra/anisotropy,
  detector response, fast-ion distribution uncertainty, density uncertainty,
  and propagated UQ before predictive or high-fidelity readiness can pass.

### 2026-05-06: Local PDF Source Audit

- Checked local PDFs under DPF-Unified by filename, PDF metadata, DOI/title
  text extraction, and SHA-256 duplicate checks.
- Added the audit to `docs/SCIENTIFIC_CLOSURE_SOURCE_QUEUE.md`.
- Added `docs/LOCAL_PDF_SOURCE_AUDIT_2026_05_06.md`.
- Exact local matches found:
  - Akel et al. 2021, DOI `10.1016/j.radphyschem.2021.109633`, outside KR
    at `archive_reference_OLD/references/papers/core-dpf/akel-2021-pf1000-neutron-yield.pdf`
    with an identical duplicate under `archive_reference_OLD/references/papers/archive/`.
  - Gribkov et al. 2007 Part I, DOI `10.1088/0022-3727/40/7/021`.
  - Gribkov et al. 2007 Part II, DOI `10.1088/0022-3727/40/12/008`.
  - Schmidt et al. 2022 MJOLNIR high-low, DOI `10.1063/5.0089121`.
  - Malir et al. 2024 interferometry, DOI `10.1063/5.0193268`.
  - Goyon et al. 2025 neutron-generation dynamics, DOI `10.1063/5.0253547`.
- Filename problems found:
  - `gribkov-2007-pf1000-jphysd-part2` is actually Part I.
  - `scholz-2007-pf1000-part2-jphysd` is actually Gribkov et al. Part II.
  - `goyon-2022-mjolnir-high-low` is the Schmidt et al. article named by a
    non-first author.
  - `petrov-2022-mjolnir-high-low-discharges` appears to be an LLNL
    accepted-manuscript/preprint copy of the Schmidt/Goyon article.
- No exact local PDF match found for:
  - Cikhardtova et al. 2015 linear densities
  - Sadowski/Scholz/PF-1000 team 2004 fast ions/neutrons
  - Catenacci et al. 2020 neutron time-energy tomography
  - Springham et al. 2021 Zr/Be activation detectors
  - Klir et al. 2011 TOF detector calibration
  - Jednorog et al. 2017 PF-1000 activation monitor
- Correction from the subsequent parity pass: Akel et al. 2021 was already in
  `KnowledgeReference/radiation-physics-and-chemistry-188-2021-109633.md`.
  The next action is verified digitization of waveform/yield figures and
  tables, not paper ingestion.
- Verification: `git diff --check` is clean after the audit documentation
  update.

### 2026-05-06: KR PDF Parity Verification

- Added `scripts/verify_kr_pdf_parity.py`.
- The verifier checks:
  - PDF page count equals KR JSON `page_count`
  - every PDF page's extracted text matches KR JSON `pages[].text`
  - every PDF page's extracted text is present in the KR markdown after
    normalization
  - source PDF SHA-256 is reported for provenance
- Corrected the prior audit: Akel et al. 2021 was already represented in
  `KnowledgeReference/radiation-physics-and-chemistry-188-2021-109633.md`.
  The earlier check missed it because the KR filename is generic.
- No new `KnowledgeReference` markdown file was created. All exact local PDF
  matches already had KR markdown/JSON pairs and passed text parity:
  - Akel et al. 2021: 6/6 pages
  - Gribkov et al. 2007 Part I: 13/13 pages
  - Gribkov et al. 2007 Part II: 16/16 pages
  - Schmidt et al. 2022 MJOLNIR article: 29/29 pages
  - Schmidt/Goyon accepted-manuscript copy: 16/16 pages
  - Malir et al. 2024: 14/14 pages
  - Goyon et al. 2025 canonical KR record: 10/10 pages
  - Goyon et al. 2025 short-name KR duplicate: 10/10 pages
- Boundary: this verifies text parity only. Figure pixels and plotted curves
  are not numeric validation evidence until they pass the digitization
  provenance gate.
- Immediate next action: verified digitization of Akel 2021 waveform/yield
  figures and tables, not paper ingestion.
- Verification:
  - `python3 -m py_compile scripts/verify_kr_pdf_parity.py`
  - `python3 -m pytest tests/test_digitization.py -q` passed
    (`5 passed in 0.85s`)
  - `git diff --check` is clean.

### 2026-05-06: Akel 2021 Typed Table Target

- Added `pf1000_16kv_shot_table_2021_akel` from
  `KnowledgeReference/radiation-physics-and-chemistry-188-2021-109633.md`.
- The new target encodes all 24 merged Akel Table 1/Table 2 shot rows:
  pressure, L0, r0, peak current, pinch current, fitted Lee factors, axial
  speed, shock speed, piston speed, pinch density, pinch radius/length,
  computed neutron yield, measured neutron yield, and measured-yield
  uncertainty.
- Added table provenance metadata:
  - Table 1 rows: `330-583`
  - Table 2 rows: `584-837`
  - merged rows: `24`
  - table shot IDs match
  - KR markdown/PDF text parity verified
  - Akel PDF SHA-256:
    `9a762bc36bc1f5c175a0ec8dc07b69c48ad956d0c6a382882daf4e24677dcb3b`
- Corrected the existing shot-12581 phase target `fmr` from `0.25` to `0.26`
  for table-row consistency. The prose gives `0.25`; Table 1 gives `0.26`.
  The table-backed row now preserves the table value explicitly.
- Grouped the Akel PF-1000 16 kV waveform, phase, and scalar/yield table
  targets under validation scope `pf1000_16kv_2021_akel`.
- Scientific status:
  - Closed: row-level scalar current, fitted-parameter, pinch-geometry, and
    neutron-yield targets are now available from KR without manual paper
    rereading.
  - Still open: waveform curves, phase timing curves, neutron timing,
    neutron spectrum, neutron anisotropy, detector response, and blind
    predictive acceptance criteria. This is not high-fidelity neutron
    predictive closure.
- Verification:
  - `python3 -m py_compile src/dpf/validation/kr_targets.py src/dpf/validation/__init__.py tests/test_kr_targets.py`
  - focused Akel/KR target tests passed (`6 passed in 0.92s`)
  - `python3 -m pytest tests/test_kr_targets.py tests/test_kr_corpus.py -q`
    passed (`79 passed in 1.29s`)
  - `python3 -m pytest tests/test_quality_assessment.py::TestQualityAssessment::test_scientific_accuracy_gap_report_lists_remaining_work tests/test_kr_targets.py::test_kr_validation_same_scope_target_report_requires_one_scope -q`
    passed (`2 passed in 0.68s`)
  - `git diff --check` is clean.

### 2026-05-06: Akel 2021 Scalar-Table Evidence Comparator

- Added `pf1000_16kv_akel_table_candidate_evidence()`.
- The comparator accepts either:
  - a mapping keyed by shot number
  - a list of row mappings with `shot`
  - a mapping containing `shot_rows`
- Default required fields:
  - `peak_current_kA`
  - `pinch_current_kA`
  - `axial_speed_cm_per_us`
  - `shock_speed_cm_per_us`
  - `piston_speed_cm_per_us`
  - `pinch_density_1e23_per_m3`
  - `pinch_radius_cm`
  - `pinch_length_cm`
  - `neutron_yield_n`
- The neutron-yield target is the measured yield from Akel Table 2. The
  article's computed Lee yield remains available in the table rows as source
  context, but the validation comparison defaults to measured yield.
- Evidence output includes:
  - required/provided row counts
  - missing shots
  - extra shots
  - missing fields
  - field pass/fail flags
  - maximum relative errors
  - per-shot/per-field errors
- Boundary: this can pass only scalar table agreement. It does not close
  waveform, phase timing, neutron timing, spectrum, anisotropy, detector
  response, or blind predictive acceptance.
- Updated the source-queue/audit docs so the remaining Akel work is figure
  digitization; table rows are no longer listed as uningested.
- Verification:
  - `python3 -m py_compile src/dpf/validation/kr_targets.py src/dpf/validation/__init__.py tests/test_kr_targets.py`
  - focused comparator tests passed (`3 passed in 0.70s`)
  - `python3 -m pytest tests/test_kr_targets.py tests/test_kr_corpus.py -q`
    passed (`81 passed in 1.24s`)
  - `python3 -m pytest tests/test_kr_targets.py tests/test_kr_corpus.py tests/test_quality_assessment.py::TestQualityAssessment::test_scientific_accuracy_gap_report_lists_remaining_work -q`
    passed (`82 passed in 1.47s`)
  - broader validation slice passed (`230 passed, 3 skipped in 12.36s`)
  - `git diff --check` is clean.

### 2026-05-06: Tier-5 Scalar-Yield Closure Gate

- Tightened `neutron_validation_scope_closure_report()`.
- Tier 5 now requires same-scope evidence for:
  - scalar neutron yield
  - neutron mechanism/timing
  - neutron spectrum
  - neutron anisotropy
- Added `neutron_yield_validation` to source-authority auditing.
- Updated `validation_tier_report()` and predictive-readiness wording to say
  `Neutron yield/mechanism/timing/spectrum/anisotropy validation`.
- Updated `pf1000_16kv_akel_table_candidate_evidence()` so passing scalar yield
  table comparison exposes `validated_features={"yield": True}`.
- Updated docs that still described neutron validation as
  timing/spectrum/anisotropy-only.
- Resulting scientific status:
  - A timing/spectrum/anisotropy packet alone is no longer Tier-5 supported.
  - App-level MJOLNIR helper evidence now remains `decomposed_estimate` until
    a same-scope scalar-yield validation packet is attached.
  - The Akel table comparator can supply scalar-yield evidence for the
    PF-1000 16 kV Akel scope, but that scope still lacks waveform, neutron
    timing, spectrum, anisotropy, and detector-response closure.
- Verification:
  - `python3 -m py_compile src/dpf/validation/quality_assessment.py src/dpf/validation/kr_targets.py tests/test_quality_assessment.py tests/test_kr_targets.py tests/test_mhd_physics_integration.py`
  - focused Tier-5/yield tests passed (`6 passed in 0.90s`)
  - `python3 -m pytest tests/test_quality_assessment.py tests/test_kr_targets.py tests/test_mhd_physics_integration.py -q`
    passed (`171 passed, 3 skipped in 8.84s`)
  - broader validation slice passed (`230 passed, 3 skipped in 12.49s`)
  - `git diff --check` is clean.

### 2026-05-06: Neutron-Yield KR Target Group

- Added `neutron_yield` to the end-to-end KR target groups.
- `_typed_observable_groups()` now treats `neutron_yield_targets` as a first-
  class observable group.
- Same-scope closure blockers now report neutron-yield missing/partial status.
- `scientific_closure_source_acquisition_queue()` now has priority-1
  `neutron_yield` items.
- PF-1000 full-energy target now explicitly records scalar-yield context:
  - yield range `5.0e10` to `2.0e11` neutrons/shot
  - maximum yield `6.0e11` neutrons/shot
  - shot-3121 activation anisotropy availability
  - 90 degree bubble-detector cross-check angle
  - same-scope detector response required for predictive yield
- Current widest scope remains `pf1000_full_energy_2007_gribkov_scholz`, but
  neutron yield is now a partial group with blockers:
  - `yield_calibration_uncertainty`
  - `neutron_field_transport_or_room_scatter_response_model`
  - `fast_ion_distribution_uncertainty`
- Updated `docs/SCIENTIFIC_CLOSURE_SOURCE_QUEUE.md` with a priority-1
  `neutron_yield` section.
- Verification:
  - `python3 -m py_compile src/dpf/validation/kr_targets.py src/dpf/validation/source_acquisition.py tests/test_kr_targets.py tests/test_digitization.py`
  - `python3 -m pytest tests/test_kr_targets.py tests/test_kr_corpus.py tests/test_digitization.py tests/test_quality_assessment.py::TestQualityAssessment::test_scientific_accuracy_gap_report_lists_remaining_work -q`
    passed (`87 passed in 1.90s`)
  - broader validation slice passed (`230 passed, 3 skipped in 12.30s`)
  - `git diff --check` is clean.

### 2026-05-06: App-Level Akel Scalar-Yield Validation Hook

- Added a PF-1000 16 kV Akel table hook in `_apply_post_processing()`.
- Accepted input keys:
  - `pf1000_16kv_akel_table_predictions`
  - `akel_2021_table_predictions`
  - `neutron_yield_validation_rows`
- Guardrails:
  - device/preset must identify PF-1000
  - circuit voltage must be within 5 percent of 16 kV
  - row comparison must pass `pf1000_16kv_akel_table_candidate_evidence()`
- Passing rows are promoted to `neutron_yield_validation`.
- Failing/incomplete rows remain `neutron_yield_validation_candidate`.
- App neutron scope closure now runs when only scalar-yield validation is
  present, so timing/spectrum/anisotropy blockers are visible.
- Scientific status:
  - Closed: production result dictionaries can now carry KR-backed scalar-
    yield validation for the Akel PF-1000 16 kV scope.
  - Still open: the hook needs the full 24-shot table. A single run yield does
    not validate predictive neutron performance.
- Verification:
  - `python3 -m py_compile app_mhd.py tests/test_mhd_physics_integration.py`
  - focused app Akel/Tier-5 tests passed (`3 passed in 1.72s`)
  - broader validation slice passed (`232 passed, 3 skipped in 13.01s`)
  - `git diff --check` is clean.

### 2026-05-06: Akel 2021 Figure Digitization Queue

- Added `scientific_closure_digitization_queue()`.
- Exported it from `dpf.validation`.
- The queue now tracks the six remaining Akel 2021 figure tasks:
  - Fig. 1 current waveform, shot 12581, 1.2 Torr, source lines 294-295.
  - Fig. 2 current waveform, shot 12584, 1.2 Torr, source lines 296-297.
  - Fig. 3 current waveform, shot 12592, 1.05 Torr, source lines 298-299.
  - Fig. 4 current waveform, shot 12604, 1.05 Torr, source lines 300-301.
  - Fig. 5 neutron-yield plot, 1.2 Torr, source line 916.
  - Fig. 6 neutron-yield plot, 1.05 Torr, source line 917.
- Each task records:
  - KR markdown path and SHA-256
  - parity-verified Akel PDF SHA-256
  - local PDF candidates
  - required series
  - page hint
  - required digitization packet fields
  - `digitization_verification_evidence()` as the gate
  - `figure_image_status="not_extracted"`
- Updated `docs/SCIENTIFIC_CLOSURE_SOURCE_QUEUE.md` so the local Akel figure
  work is explicit and tested.
- Scientific status:
  - Closed: figure digitization is now represented as machine-readable local
    closure work instead of an unstructured note.
  - Still open: no extracted figure image, axis calibration, digitized series,
    overlay residual, or independent review exists yet.
- Verification:
  - `python3 -m py_compile src/dpf/validation/digitization.py src/dpf/validation/__init__.py tests/test_digitization.py`
  - focused digitization/KR/quality tests passed (`89 passed in 1.88s`)
  - broader validation slice passed (`234 passed, 3 skipped in 13.47s`)
  - `git diff --check` is clean

### 2026-05-06: Digitization Queue Acceptance Status

- Added `scientific_closure_digitization_status()`.
- Exported it from `dpf.validation`.
- The status function evaluates future digitization packets against both:
  - `digitization_verification_evidence()`
  - the exact local queue task metadata
- Additional task-level checks require matching:
  - task ID
  - KR source path
  - KR source SHA-256
  - local PDF SHA-256
  - source line window
  - figure ID
  - page
  - required series names
- The report separates accepted, failed, open, invalid, and extra packets.
- Scientific status:
  - Closed: the Akel digitization workflow now has a tested one-for-one
    acceptance method.
  - Still open: no real Akel figure packet has been accepted; all six figure
    tasks remain open unless packets are supplied.
- Verification:
  - `python3 -m py_compile src/dpf/validation/digitization.py src/dpf/validation/__init__.py tests/test_digitization.py`
  - `tests/test_digitization.py` passed (`10 passed in 0.44s`)
  - focused digitization/KR/quality tests passed (`92 passed in 1.48s`)
  - broader validation slice passed (`237 passed, 3 skipped in 12.80s`)
  - `git diff --check` is clean

### 2026-05-06: App-Level Digitization Closure Export

- App post-processing now exports:
  - `scientific_closure_digitization_queue`
  - `scientific_closure_digitization_status`
- If a caller supplies `scientific_closure_digitization_packets` or
  `digitization_packets`, the app evaluates them through
  `scientific_closure_digitization_status()`.
- Default production runs now show the Akel figure queue as open instead of
  leaving figure digitization outside the result metadata.
- Scientific status:
  - Closed: app results now carry local figure-digitization blockers alongside
    KR target coverage, KR corpus review status, predictive readiness, and
    high-fidelity readiness.
  - Still open: no Akel figure data has been rendered, cropped, calibrated,
    digitized, reviewed, or accepted.
- Verification:
  - `python3 -m py_compile app_mhd.py tests/test_mhd_physics_integration.py`
  - focused app/digitization tests passed (`11 passed in 1.02s`)
  - broader validation slice passed (`237 passed, 3 skipped in 12.67s`)
  - `git diff --check` is clean

### 2026-05-06: Figure-Digitization Scientific-Accuracy Gap

- Added `figure_digitization` to `scientific_accuracy_gap_report()`.
- The gap reads `scientific_closure_digitization_status` from the result when
  present, or computes the default open queue status when absent.
- Status rules:
  - `supported`: the local digitization queue is complete.
  - `partial`: at least one task is accepted, with the rest failed or open.
  - `blocked`: no task is accepted, or status is unavailable.
- App results now show `figure_digitization` blocked by `0/6` accepted local
  scientific-closure figure tasks.
- Scientific status:
  - Closed: open Akel figure digitization is now a first-class high-fidelity
    blocker.
  - Still open: no digitized figure data has been created or accepted.
- Verification:
  - `python3 -m py_compile src/dpf/validation/quality_assessment.py tests/test_quality_assessment.py tests/test_mhd_physics_integration.py`
  - targeted readiness/gap tests passed (`3 passed in 1.52s`)
  - broader validation slice passed (`237 passed, 3 skipped in 13.60s`)
  - `git diff --check` is clean

### 2026-05-06: App-Level Source-Acquisition Queue Export

- App post-processing now exports `scientific_closure_source_acquisition_queue`.
- Result payloads now carry candidate DOI links and required local-ingestion
  steps beside KR target coverage, corpus review status, digitization status,
  predictive readiness, and high-fidelity readiness.
- Scientific status:
  - Closed: app results expose the user-requested source acquisition workflow.
  - Still open: acquisition candidates are not evidence until the correct
    document is acquired, added locally under `KnowledgeReference`, reviewed,
    and digitized if required.
- Verification:
  - `python3 -m py_compile app_mhd.py tests/test_mhd_physics_integration.py`
  - focused app/source-queue tests passed (`3 passed in 1.44s`)
  - broader validation slice passed (`237 passed, 3 skipped in 13.10s`)
  - `git diff --check` is clean

### 2026-05-06: Local-vs-Acquisition Source Queue Split

- Source-acquisition queue entries now annotate DOI leads with local status
  from the local PDF parity audit.
- Queue items now separate:
  - `local_sources_available`
  - `candidate_sources_for_acquisition`
  - `candidate_sources`, retained for compatibility and annotated per source
- Tagged as `parity_verified_knowledge_reference`:
  - Akel 2021
  - Gribkov 2007 Parts I/II
  - Schmidt 2022
  - Malir 2024
  - Goyon 2025
- Tagged as `not_found_as_exact_local_pdf`:
  - Cikhardtova 2015
  - Sadowski/Scholz 2004
  - Catenacci 2020
  - Springham 2021
  - Klir 2011
  - Jednorog 2017
- Updated `docs/SCIENTIFIC_CLOSURE_SOURCE_QUEUE.md` so local sources are not
  framed as user-acquisition requests.
- Scientific status:
  - Closed: the queue no longer tells the user to acquire sources already
    verified locally.
  - Still open: local sources still need typed targets or verified digitized
    data for any missing observable before they close validation groups.
- Verification:
  - `python3 -m py_compile src/dpf/validation/source_acquisition.py tests/test_digitization.py`
  - focused source-queue/app tests passed (`3 passed in 1.38s`)
  - broader validation slice passed (`237 passed, 3 skipped in 12.87s`)
  - `git diff --check` is clean

### 2026-05-06: Akel Figure Render Page Correction

- Rendered the parity-verified Akel PDF pages into the temporary workbench
  `/private/tmp/dpf_akel_digitization` using `pdftoppm`.
- Corrected digitization queue page hints:
  - Figs. 1-4 render on PDF page 3, not page 4.
  - Figs. 5-6 render on PDF page 5, not page 6.
- Page 4 is the typed table page, and page 6 is references, so the previous
  queue hints would have sent digitization to the wrong page renders.
- Scientific status:
  - Closed: the queue now points to the rendered pages that actually contain
    the cited Akel plots.
  - Still open: the temporary renders are not KR evidence, accepted packets, or
    digitized arrays.
- Verification:
  - `python3 -m py_compile src/dpf/validation/digitization.py tests/test_digitization.py`
  - `tests/test_digitization.py` passed (`10 passed in 0.76s`)
  - broader validation slice passed (`237 passed, 3 skipped in 13.36s`)
  - `git diff --check` is clean

### 2026-05-06: Akel Scalar-Yield Uncertainty Diagnostics

- `pf1000_16kv_akel_table_candidate_evidence()` now reports:
  - neutron-yield absolute error
  - source-reported measured-yield uncertainty per row
  - measurement-uncertainty-normalized error per row
  - `max_measurement_uncertainty_normalized_error`
- Scientific status:
  - Closed: scalar-yield comparison now exposes the uncertainty scale printed
    in Akel Table 2.
  - Still open: this remains scalar table comparison. It does not provide a
    blind-prediction acceptance criterion, detector response, neutron timing,
    spectrum, or anisotropy closure.
- Verification:
  - `python3 -m py_compile src/dpf/validation/kr_targets.py tests/test_kr_targets.py`
  - focused comparator/app tests passed (`3 passed in 0.94s`)
  - broader validation slice passed (`237 passed, 3 skipped in 13.30s`)
  - `git diff --check` is clean

### 2026-05-06: PF-1000 16 kV Candidate Scope Consistency

- Fixed Akel candidate evidence scope reporting in:
  - `pf1000_16kv_phase_candidate_evidence_from_history()`
  - `pf1000_16kv_derived_output_candidate_evidence()`
- Both now report `validation_scope="pf1000_16kv_2021_akel"` instead of the
  individual target ID.
- App-level PF-1000 16 kV phase and derived-output candidates now share the
  same Akel validation scope as the waveform and scalar-yield targets.
- Scientific status:
  - Closed: Akel candidate evidence has consistent scope identity for
    same-scope accounting.
  - Still open: phase and derived-output packets remain partial candidates
    because the KR record lacks complete measured axial, radial, and pinch
    phase endpoints.
- Verification:
  - `python3 -m py_compile src/dpf/validation/kr_targets.py tests/test_kr_targets.py tests/test_mhd_physics_integration.py`
  - focused scope tests passed (`3 passed in 0.84s`)
  - broader validation slice passed (`237 passed, 3 skipped in 12.92s`)
  - `git diff --check` is clean

### 2026-05-06: Akel Phase-Semantics Target

- Added `phase_semantics` to `pf1000_16kv_shot12581_phase_2021_akel`.
- The target now records that Akel's fitted Lee factors map to:
  - axial phase mass/current semantics: `fm`, `fc`
  - radial phase mass/current semantics: `fmr`, `fcr`
- Same-scope target reporting now marks `phase_semantics` present for
  `pf1000_16kv_2021_akel`.
- Scientific status:
  - Closed: the Akel 16 kV scope no longer has a false missing
    phase-semantics blocker.
  - Still open: phase timing remains partial because complete measured axial,
    radial, and pinch endpoint timings with uncertainty are not available.
- Verification:
  - `python3 -m py_compile src/dpf/validation/kr_targets.py tests/test_kr_targets.py`
  - focused target/same-scope tests passed (`2 passed in 0.60s`)
  - broader validation slice passed (`237 passed, 3 skipped in 12.65s`)
  - `git diff --check` is clean

### 2026-05-06: Akel Table Uncertainty Target

- Added an explicit `uncertainty` block to
  `pf1000_16kv_shot_table_2021_akel`.
- The target now records:
  - measured neutron-yield uncertainty is available per row
  - row uncertainty range is `2.0e7` to `2.0e8` neutrons/shot
  - missing uncertainty components for waveform uncertainty, detector-response
    uncertainty, model-form uncertainty, input-parameter covariance, and a
    blind-prediction acceptance rule
- Same-scope reporting now marks `uncertainty` present but partial for
  `pf1000_16kv_2021_akel`.
- Scientific status:
  - Closed: Akel scalar yield uncertainty is now typed KR target data.
  - Still open: this is not a full uncertainty budget.
- Verification:
  - `python3 -m py_compile src/dpf/validation/kr_targets.py tests/test_kr_targets.py`
  - focused table/same-scope tests passed (`2 passed in 0.43s`)
  - broader validation slice passed (`237 passed, 3 skipped in 12.67s`)
  - `git diff --check` is clean

### 2026-05-06: MLX Collection Abort Hardening

- Issue:
  - The local MLX install/path is not a normal missing optional dependency.
  - `import mlx.core` aborts the Python interpreter, so pytest cannot catch it
    as `ImportError` and cannot safely apply `pytest.importorskip`.
  - Local metadata: `mlx==0.31.0`, Python `3.11.9`, and
    `macOS-26.3.1-arm64-arm-64bit`; the project safe detector reports
    `HAS_MLX=False` because the child-process import probe fails.
  - Initial collection abort path was `tests/test_amr_mlx.py` ->
    `dpf.metal.__init__` -> `dpf.metal.mlx_device` -> eager `mlx.core`
    import. Full collection then exposed the same eager-import hazard in
    `dpf.metal.mlx_kernels`.
- Fix:
  - `src/dpf/metal/mlx_device.py` now probes MLX in a child process and sets
    `HAS_MLX` from the child exit code.
  - `require_mlx()` imports MLX in-process only after the safe probe succeeds.
  - `src/dpf/metal/device.py`, `src/dpf/metal/mlx_amr.py`, and
    `src/dpf/metal/mlx_kernels.py` now use the safe detector path.
  - MLX tests and mixed CPU/MLX tests now use safe `HAS_MLX` gates, and
    `tests/conftest.py` guards legacy `pytest.importorskip("mlx.core")` calls.
- Scientific status:
  - Closed: broken MLX can no longer abort collection, so KR validation and
    CPU fallback tests remain runnable.
  - Still open: this does not validate MLX/Metal physics. In this environment
    MLX-specific tests skip until `mlx.core` imports cleanly.
- Verification:
  - targeted MLX/mixed tests passed or skipped cleanly
    (`15 passed, 28 skipped in 0.30s`)
  - full pytest collection completed without abort
    (`3657/3775 tests collected, 118 deselected in 3.11s`)
  - broader validation slice passed (`237 passed, 3 skipped in 12.97s`)
  - `git diff --check` is clean

### 2026-05-06: MLX Runtime Triage And Scientific Gate Correction

- Touched:
  - `tests/test_mlx_circuit_coupling.py`
  - `tests/test_mlx_pf1000.py`
  - `CodexFindings.md`
  - `CortexFindings.md`
- MLX collection issue:
  - `mlx==0.31.0` and `mlx-metal==0.31.0` are installed.
  - Outside the sandbox, `mlx.core` imports and reports `Device(gpu, 0)`.
  - Inside the sandbox, Metal enumeration returns no device, and `mlx.core`
    aborts natively during Metal device construction.
  - CPU/disable-Metal environment attempts did not prevent the import abort:
    `MLX_DEFAULT_DEVICE=cpu`, `MLX_DEVICE=cpu`, `MLX_DISABLE_METAL=1`, and
    `MLX_DISABLE_COMPILE=1`.
- Fix status:
  - Sandbox collection remains protected by the safe MLX child-process probe.
  - Real MLX validation is now explicitly a Metal-visible/outside-sandbox
    execution path.
- Scientific correction:
  - `test_btheta_increases_inward` now expects stronger `B_theta` at smaller
    radius.
  - Source-of-truth basis:
    `KnowledgeReference/plasma-formulary.md:2470-2473` gives
    `B_theta = mu I / (2*pi*r)`, and
    `KnowledgeReference/two-dimensional-simulation-of-dense-plasma-focus-5.md:78-84`
    gives the DPF boundary relation `Bphi = mu I / (2*pi*r)`.
- PF-1000 gate correction:
  - `TestMLXPF1000MustHave` and `TestMLXPF1000ShouldHave` are
    `xfail(run=False)`.
  - Reason: M6/full-discharge stability is still blocked by the documented
    CFL/full-duration issue. A run that stops before the required post-peak
    interval is not five-phase validation.
  - Source/project gate basis:
    `docs/SPRINT4_VALIDATION_REVIEW.md:105-113` and
    `docs/METAL_V2_DOD.md:330-337`.
- Remaining plan:
  - Close PF-1000 M6/CFL duration stability.
  - Re-enable the PF-1000 full-discharge MLX classes only after the fixture
    reaches the required post-peak/full-discharge duration without hitting the
    fixed step cap.
  - Keep fast config-level MLX checks runnable.
  - Continue same-scope KR-backed closure for spatial state, neutron timing,
    spectrum, anisotropy, detector response, uncertainty, and numerical
    convergence before any high-fidelity neutron-predictive claim.
- Verification:
  - Python syntax compilation passed for touched MLX hardening/test files.
  - `git diff --check` is clean.
  - sandbox collection passed:
    `4228/4346 tests collected, 118 deselected in 9.91s`
  - standing KR validation slice passed:
    `237 passed, 3 skipped in 13.40s`
  - outside-sandbox targeted MLX tests passed:
    `139 passed in 0.97s`
  - outside-sandbox full MLX glob passed with blocked PF-1000 gates as xfail:
    `553 passed, 19 xfailed in 50.43s`
  - `tests/test_mlx_pf1000.py`:
    `4 passed, 14 xfailed in 0.97s`

### 2026-05-06: PF-1000 MLX Probe Stability Through 10000 Steps

- Touched:
  - `src/dpf/metal/mlx_primitives.py`
  - `src/dpf/metal/mlx_state.py`
  - `src/dpf/metal/mlx_solver.py`
  - `tests/test_mlx_primitives.py`
  - `tests/test_mlx_state.py`
- Issue:
  - The interrupted 3000-step PF-1000 probe was rerun outside the sandbox.
  - The dense 1900-step probe passed, but the 2200-step probe exposed a
    deterministic `pressure` NaN at engine step `1985`, `t=0.690156 us`.
  - After pressure unpack hardening, the next run exposed the underlying
    conservative-state overflow as non-finite `B` at step `1986`.
- Fix:
  - Dual-energy pressure recovery now sanitizes the total-energy and entropy
    pressure candidates before blending, so an unused `inf` candidate cannot
    produce `NaN` through `0*inf`.
  - `MLXState.to_state_dict()` uses the same finite dual-energy blend for
    cylindrical and Cartesian unpacking.
  - The MLX solver's CPU-side post-hyperbolic floor now rebuilds momentum from
    bounded velocity after density flooring instead of multiplying momentum by
    `_rho_floor/rho` in vacuum cells.
  - CPU-side energy and vacuum `B_theta` prescription bookkeeping now uses
    finite float64 intermediates before returning to MLX float32.
- Operational MLX float32 rule:
  - For future MLX float32 nonfinite/overflow issues, first test the same
    pattern before adding narrower clamps: move CPU-side repair bookkeeping to
    float64, recover finite primitive-like quantities, rebuild conserved
    fields from bounded finite values, clip only for representability, and only
    then return to MLX float32.
  - Do not multiply conserved components by huge floor ratios in vacuum cells;
    rebuild from finite velocity, pressure/energy, and magnetic components
    instead.
- Verification:
  - Python syntax compilation passed for touched MLX files and tests.
  - Focused MLX regressions passed:
    `2 passed in 0.73s`.
  - Short PF-1000 MLX probe passed:
    `DPF_MLX_PROBE_STEPS=600`, `1 passed in 16.79s`.
  - Former failure-window PF-1000 MLX probe passed:
    `DPF_MLX_PROBE_STEPS=2200`, `1 passed in 48.60s`.
  - Original long PF-1000 MLX probe passed:
    `DPF_MLX_PROBE_STEPS=3000`, reaching `t=0.868989 us`,
    `I=0.671730 MA`, `max_B=1.164430`, `1 passed in 65.62s`.
  - Extended PF-1000 MLX probe passed:
    `PYTHONFAULTHANDLER=1 DPF_MLX_PROBE_STEPS=5000`, reaching
    `t=1.141062 us`, `I=0.872277 MA`, `max_B=1.512073`,
    `1 passed in 110.73s`.
  - Longer PF-1000 MLX probe passed:
    `PYTHONFAULTHANDLER=1 DPF_MLX_PROBE_STEPS=10000`, reaching
    `t=1.584247 us`, `I=1.187937 MA`, `max_B=2.059263`,
    `1 passed in 218.22s`.
- Late-window native-abort triage:
  - A first 20000-step probe attempt exited at native/process level with code
    `-1` before the first 2000-step checkpoint, with no Python faulthandler
    traceback and no probe assertion.
  - A dense 2000-step rerun passed with `DPF_MLX_PROBE_PRINT_INTERVAL=100`,
    reaching `t=0.747684 us`, `I=0.580770 MA`, and `max_B=1.006752`.
  - An exact 20000-step rerun then advanced past the historical stall and
    reached step 18000 (`t=2.075847 us`, `I=1.521106 MA`,
    `max_B=2.636804`) before another native/process-level `-1` exit before
    step 20000.
  - The Python nonfinite state checks did not fire in either native exit, so
    the current working hypothesis is late-window MLX/Metal runtime stability
    or cache/resource pressure rather than a caught Python-level NaN.
  - The probe now has optional MLX memory/cache controls:
    `DPF_MLX_PROBE_MEMORY=1` prints `mlx_active_MB`, `mlx_cache_MB`, and
    `mlx_peak_MB`; `DPF_MLX_PROBE_CLEAR_CACHE_INTERVAL=N` calls
    `mlx.clear_cache()` every `N` completed steps.
  - A cache-clearing 20000-step run with
    `DPF_MLX_PROBE_CLEAR_CACHE_INTERVAL=1000` and
    `DPF_MLX_PROBE_MEMORY=1` reached step 12000 (`t=1.709192 us`,
    `I=1.274353 MA`, `max_B=2.209062`, `mlx_active_MB=0.288`,
    `mlx_cache_MB=10.770`, `mlx_peak_MB=9.801`) and then exited with native
    code `-1` before step 14000. Periodic `mlx.clear_cache()` did not remove
    the late-window native abort.
  - A bounded dense-window run with `DPF_MLX_PROBE_PRINT_START=12000` and
    `DPF_MLX_PROBE_PRINT_START_INTERVAL=25` exited natively after the first
    printed step, reinforcing that the current abort behavior is intermittent
    process/runtime stability and not a deterministic field value at a fixed
    step.
  - Fresh macOS crash reports were found under
    `~/Library/Logs/DiagnosticReports/Python-2026-05-06-*.ips`. The latest
    MLX-related report shows `SIGABRT`, `NSRangeException`,
    `-[__NSArray0 objectAtIndex:]: index 0 beyond bounds for empty array`, with
    the backtrace in `mlx::core::metal::Device::Device()`. A separate report
    shows `crashed on child side of fork pre-exec`. These reports support a
    native MLX/Metal/subprocess-environment issue around device discovery or
    process spawning; they do not identify a Python-level state NaN.
  - A fresh direct MLX initialization check after those reports still passed:
    `python3 -X faulthandler -c "import mlx.core as mx; print(mx.default_device())"`
    reported `Device(gpu, 0)`.
  - Added `scripts/run_mlx_pf1000_probe.py`, a standalone probe runner that
    bypasses pytest/conftest and sets `DPF_MLX_ASSUME_AVAILABLE=1` only after
    importing `mlx.core` directly in the Metal-visible process. This separates
    solver/runtime failures from pytest plugin, safe import-probe, and
    subprocess behavior.
  - Added `DPF_MLX_ASSUME_AVAILABLE=1` support in `mlx_device.py` as an
    explicit opt-in for already-validated Metal-visible processes. Do not use
    this in sandboxed collection because it bypasses the protective child
    import probe.
  - The standalone probe passed 2000 steps with memory telemetry, reaching
    `t=0.747684 us`, `I=0.580770 MA`, `max_B=1.006752`,
    `mlx_active_MB=0.288`, `mlx_cache_MB=10.525`, and
    `mlx_peak_MB=9.801`.
  - The standalone 20000-step cap run passed, reaching `t=2.200558 us`,
    `I=1.602652 MA`, `max_B=2.778162`, `mlx_active_MB=0.288`,
    `mlx_cache_MB=10.238`, and `mlx_peak_MB=9.801`. This indicates the prior
    `-1` exits were tied to the pytest/conftest/subprocess path or local
    process-spawn/device-discovery instability rather than a deterministic
    solver-state failure in the MLX MHD step.
  - Focused verification after standalone isolation:
    `python3 -m py_compile src/dpf/metal/mlx_device.py
    tests/test_mlx_pf1000_probe.py scripts/run_mlx_pf1000_probe.py` passed;
    `python3 -m pytest tests/test_mlx_device.py -q` passed (`21 passed`);
    `python3 -m pytest tests/test_mlx_primitives.py tests/test_mlx_state.py -q`
    passed (`61 passed`); `git diff --check` passed.
  - Target-time gate update:
    `tests/test_mlx_pf1000.py` now uses named PF-1000 cap/target controls
    instead of a hidden `range(20000)`: `DPF_MLX_PF1000_STEP_CAP` and
    `DPF_MLX_PF1000_TARGET_US`. The target is increase-only and cannot be set
    below the M6 `6 us` requirement. The fixture records target, cap, and
    cap-exhaustion metadata on the engine so M6 reports `step cap reached
    before target` explicitly.
  - Probe target-time update:
    both `tests/test_mlx_pf1000_probe.py` and
    `scripts/run_mlx_pf1000_probe.py` now accept `DPF_MLX_PROBE_TARGET_US`.
    The pytest probe asserts if the target is not reached within
    `DPF_MLX_PROBE_STEPS`; the standalone runner prints `CAP_EXHAUSTED` and
    returns exit code `2`.
  - Verification after target-time update:
    `python3 -m pytest tests/test_mlx_pf1000.py -q` passed with blocked gates
    preserved (`4 passed, 14 xfailed`); standalone target success smoke passed
    with `DPF_MLX_PROBE_TARGET_US=0.00005`; standalone cap-exhaustion smoke
    returned code `2` with `CAP_EXHAUSTED steps=5 target_us=1.000000
    final_t_us=0.243416`; focused MLX detector/pressure/state regressions
    passed (`82 passed`); `git diff --check` passed.
  - M6 target-time probe:
    standalone `DPF_MLX_PROBE_TARGET_US=6` with
    `DPF_MLX_PROBE_STEPS=80000` reached the M6 target and exited `PASSED`.
    Checkpoints: step 10000 `t=1.584247 us`, step 20000
    `t=2.200558 us`, step 30000 `t=2.812534 us`, step 40000
    `t=3.427694 us`, step 50000 `t=4.066377 us`, step 60000
    `t=4.711821 us`, and step 70000 `t=5.354566 us`. The pre-fix runner did
    not print the final target-hit step/time; the probe now prints final
    `PASSED steps=... final_t_us=...` and includes target-hit in the telemetry
    print condition.
- Remaining limit:
  - This closes the observed early PF-1000 MLX probe instability through the
    M6 `6 us` target on the standalone path, provided the cap is raised above
    the old 20000-step fixture limit. The blocked PF-1000 full-discharge
    classes should remain disabled because the current waveform is not
    accepted: by step 70000 the current was still rising at `3.215728 MA`,
    far above the M2 nominal upper band, and S2 current-dip behavior is not
    demonstrated.

### 2026-05-06: DoD Source-Of-Truth Audit

- Reviewed the Metal v2 DoD surface against the local KR source rule:
  `docs/METAL_V2_DOD.md`, `docs/METAL_V2_SPEC.md`, and
  `docs/SPRINT4_VALIDATION_REVIEW.md`, with matching comment cleanup in
  `tests/test_mlx_pf1000.py`.
- Added source-audit/superseded-status addenda instead of rewriting the
  historical March 2026 documents.
- Corrected PF-1000 scope discipline:
  - Akel 2021 16 kV (`pf1000_akel`) is the current same-scope MLX acceptance
    target.
  - Scholz/Gribkov full-energy PF-1000 remains a separate 27 kV/full-energy
    target packet and must not be mixed into Akel M2/S1/S2 gates.
- Corrected M2 target:
  - Akel shot 12581 uses `Ipeak = 1.165 MA +/- 10%`, i.e.
    `1.0485-1.2815 MA`.
  - The previous unspecified `1.2 MA` target and `1.87 MA` spec gate were
    marked/replaced as mixed-scope or stale for the Akel 16 kV gate.
- Corrected waveform/dip status:
  - Akel establishes measured current waveform figures and derivative/dip
    timing context.
  - NRMSE and dip-depth gates remain blocked until same-scope digitized current
    trace points and per-point uncertainty are accepted.
- Corrected mass and duration language:
  - M3 distinguishes closed-domain conservation from open-discharge
    outflow/density-floor accounting.
  - M6 `12 us` is documented as a conservative engineering endurance gate, not
    a direct measured Akel source value.
- Recorded the latest Akel MLX probe result:
  - Standalone `pf1000_akel` 40000-step run passed to `t = 3.238777 us`.
  - Peak current was `1.685154 MA`, above the Akel shot-12581 M2 upper bound
    `1.2815 MA`, and was still rising.
- Remaining limit:
  - The DoD is now more truthful about source scope, but full-discharge MLX
    scientific acceptance is still blocked by current-waveform mismatch,
    missing digitized Akel waveform evidence, and incomplete same-scope
    duration/dip closure.

### 2026-05-06: Akel Preset And Axial Pressure Coupling

- Corrected `pf1000_akel` from an average/nominal Akel 24-shot preset to the
  current same-scope shot-12581 preset.
- The preset now follows the typed KR shot target:
  `p0=1.2 Torr`, `rho0=2.583e-4 kg/m^3`, `C0=1332 uF`, `V0=16 kV`,
  `L0=25 nH`, `r0=6.1 mOhm`, `fm=0.17`, `fc=0.70`, `fmr=0.26`,
  and `fcr=0.75`.
- Added `tests/test_pf1000_akel_preset.py` to ratchet the preset against
  `pf1000_16kv_shot12581_phase_targets()`.
- Added `radial_current_fraction` support to the reduced MLX snowplow and
  forwarded it from `run_mlx_discharge()`, so reduced MLX Lee/RADPF runs can
  use the same Akel `fcr` scalar as the CPU snowplow.
- Fixed the remaining overshoot mechanism in the full `SimulationEngine`
  path: during axial rundown, `_dynamic_sheath_pressure()` now returns the
  configured cold fill pressure instead of feeding MHD total plasma pressure
  into the Lee/RADPF snowplow. The old path gave Akel shot 12581 about
  `640 Pa` axial back-pressure at step 1 instead of the source `160 Pa` cold
  fill, delaying rundown and under-loading the circuit.
- Probe telemetry now prints phase, voltage, sheath position, shock radius,
  `Lp`, `dL/dt`, plasma resistance, and sheath pressure.
- Evidence before the pressure fix but after the preset fix:
  standalone 40000-step `pf1000_akel` probe passed to `t=3.316852 us`, but
  current was still rising at `peak_I=1.367902 MA`, above the shot-12581 M2
  upper bound `1.2815 MA`.
- Evidence after the pressure fix:
  standalone 32000-step `pf1000_akel` probe passed to `t=2.971234 us` with
  `peak_I=0.977154 MA`.
  - step 10000: `t=1.389409 us`, `I=0.678638 MA`, `phase=rundown`,
    `Lp=2.762352 nH`, `sheath_p=160 Pa`
  - step 20000: `t=2.028429 us`, `I=0.844696 MA`, `Lp=5.431736 nH`
  - step 30000: `t=2.806466 us`, `I=0.961079 MA`, `Lp=9.382222 nH`
- Reduced reference check:
  `run_mlx_discharge(preset_name="pf1000_akel", mode="lee", max_steps=80000)`
  now peaks at `1.150685 MA` at `5.250577 us`, inside the Akel M2 band.
- Verification:
  `py_compile` passed for touched preset/coupling/probe/test files;
  `tests/test_mlx_snowplow.py tests/test_pf1000_akel_preset.py` passed
  (`6 passed`);
  `tests/test_snowplow_consolidated.py::TestDynamicPressureFallback` passed
  (`9 passed`);
  focused Akel KR target checks passed (`2 passed`);
  `tests/test_mlx_pf1000.py -q` remains `4 passed, 14 xfailed`.
- Remaining limit:
  the 32k/2.97 us run shows the full path is back on the reference trajectory.
- Post-fix M6 probe:
  standalone `pf1000_akel` with `DPF_MLX_PROBE_TARGET_US=6` and
  `DPF_MLX_PROBE_STEPS=90000` exited `PASSED`, reaching `t=6.000007 us`
  in `76948` steps.
  - Final reported `peak_I=1.047183 MA` at `t=4.990339 us`.
  - step 40000: `t=3.576539 us`, `I=1.018345 MA`
  - step 50000: `t=4.283569 us`, `I=1.041204 MA`
  - step 60000: `t=4.927035 us`, `I=1.047142 MA`
  - step 70000: `t=5.567840 us`, `I=1.044211 MA`
- Updated remaining limit:
  standalone M6 `6 us` is now closed post-fix, but strict M2 is a low-side
  near miss (`1.047183 MA` vs lower bound `1.0485 MA`). Keep the PF-1000
  full-discharge gates blocked until M2 is confirmed inside the strict band,
  S1/S2 have accepted same-scope digitized waveform evidence, and the
  conservative 12 us engineering endurance gate is proven post-fix.
- Post-fix 8 us radial/pinch probe:
  standalone `pf1000_akel` with `DPF_MLX_PROBE_TARGET_US=8` and
  `DPF_MLX_PROBE_STEPS=130000` exited `PASSED`, reaching `t=8.000045 us`
  in `107566` steps.
  - Final reported `peak_I=1.047183 MA` at `t=4.990339 us`.
  - step 90000: `t=6.809701 us`, `phase=radial`, `I=1.015650 MA`,
    `r=12.387635 cm`, `Lp=34.725876 nH`
  - step 100000: `t=7.435049 us`, `phase=radial`, `I=0.923846 MA`,
    `r=6.068274 cm`, `Lp=44.316989 nH`, `dLdt=26.719455 nH/us`
  - final step 107566: `phase=pinch`, `I=0.739814 MA`,
    `r=2.863039 cm`, `Lp=54.412990 nH`, `dLdt=-15.836659 nH/us`
  - MLX memory telemetry stayed flat (`active=0.288 MB`, cache about
    `10.47 MB`, peak about `9.80 MB`).
- Next checks:
  the 8 us result closes the immediate radial-to-pinch stability question on
  the standalone path.
- Post-fix 12 us endurance probe:
  standalone `pf1000_akel` with `DPF_MLX_PROBE_TARGET_US=12` and
  `DPF_MLX_PROBE_STEPS=220000` exited `PASSED`, reaching `t=12.000000 us`
  in `160418` steps.
  - Final reported `peak_I=1.047183 MA` at `t=4.990339 us`.
  - step 110000: `t=8.171744 us`, `phase=pinch`, `I=0.716376 MA`,
    `r=3.442282 cm`, `Lp=51.936660 nH`, `dLdt=-13.171781 nH/us`
  - step 120000: `t=9.000231 us`, `I=0.643373 MA`, `r=6.237254 cm`,
    `Lp=43.947850 nH`
  - step 140000: `t=10.458063 us`, `I=0.641996 MA`, `r=11.155373 cm`,
    `Lp=36.134086 nH`
  - step 160000: `t=11.965820 us`, `I=0.520224 MA`, `r=15.200000 cm`,
    `Lp=31.976097 nH`
  - final step 160418: `I=0.517539 MA`, `r=15.200000 cm`,
    `Lp=31.976097 nH`
  - MLX memory telemetry stayed flat (`active=0.288 MB`, cache about
    `10.489 MB`, peak about `9.801 MB`).
- Updated remaining limit before source-scope cleanup:
  standalone `6 us`, `8 us`, and conservative `12 us` engineering endurance
  targets were closed post-fix through post-pinch expansion. Full-discharge
  acceptance remained blocked because strict M2 was still a low-side near miss
  (`1.047183 MA` vs lower bound `1.0485 MA`), S1/S2 still needed accepted
  same-scope digitized waveform evidence, and the fixed-time crowbar was not
  Akel shot-scope sourced.
- Late-voltage telemetry explanation:
  `V_kV=0.000000` after about `11.19 us` was explained by the inherited
  fixed-time crowbar previously present in `pf1000_akel` (`crowbar_enabled=True`,
  `crowbar_mode="fixed_time"`, `crowbar_time=10.5e-6`). The local Akel source
  search did not find shot-scope crowbar timing support, so post-10.5 us
  voltage/current behavior was engineering crowbar behavior,
  not same-scope Akel waveform evidence.
- Probe telemetry update:
  `scripts/run_mlx_pf1000_probe.py` and `tests/test_mlx_pf1000_probe.py` now
  print `crowbar` and `crowbar_t_us` fields. Verification: `py_compile` passed;
  5-step standalone Akel smoke passed and printed
  `crowbar=0 crowbar_t_us=-1.000000`.
- Next checks before source-scope cleanup:
  inspect the strict M2 low-side near miss without arbitrary tuning, review
  whether `pf1000_akel` should keep the unsourced fixed-time crowbar or move it
  behind an engineering preset/override, and continue same-scope Akel current
  trace digitization for S1/S2 waveform and dip acceptance.
- Final verification snapshot after 12 us/crowbar telemetry update:
  `git diff --check` clean; trailing-whitespace scan clean for touched
  notes/docs/probe files; targeted preset/PF-1000 gate slice passed
  (`5 passed, 14 xfailed in 1.72s`).

### 2026-05-06: Akel Source-Scoped Crowbar Cleanup

- Source audit result:
  the typed Akel shot-12581 target records circuit, geometry, Lee factors,
  waveform availability, and phase/dip context, but no crowbar enablement,
  crowbar time, crowbar resistance, or crowbar inductance. Local search in the
  Akel KR source found no shot-scope support for the inherited `10.5 us`
  fixed-time crowbar.
- Fix implemented:
  removed the unsourced inherited fixed-time crowbar from the source-scoped
  `pf1000_akel` preset. The preset now keeps `crowbar_enabled=False` and no
  longer carries `crowbar_time`, `crowbar_resistance`, or `crowbar_inductance`.
  `tests/test_pf1000_akel_preset.py` ratchets this source boundary.
- Probe telemetry retained:
  `scripts/run_mlx_pf1000_probe.py` and `tests/test_mlx_pf1000_probe.py` still
  print `crowbar` and `crowbar_t_us`, so engineering crowbar overrides remain
  visible in future logs.
- Verification after cleanup:
  `py_compile` passed for the touched preset/probe/test files; focused
  Akel/PF-1000 gate slice passed (`5 passed, 14 xfailed in 1.83s`); 5-step
  standalone Akel smoke passed with `crowbar=0 crowbar_t_us=-1.000000`.
- Source-scoped no-crowbar 12 us probe:
  standalone `pf1000_akel` with `DPF_MLX_PROBE_TARGET_US=12`,
  `DPF_MLX_PROBE_STEPS=220000`, and `DPF_MLX_PROBE_PRINT_INTERVAL=20000`
  exited `PASSED`, reaching `t=12.000000 us` in `161659` steps.
  - Final reported `peak_I=1.047183 MA` at `t=4.990339 us`.
  - step 120000: `t=9.000231 us`, `phase=pinch`, `I=0.643373 MA`,
    `V=10.293450 kV`, `r=6.237254 cm`, `Lp=43.947850 nH`
  - step 140000: `t=10.458063 us`, `I=0.641996 MA`, `V=9.600860 kV`,
    `r=11.155373 cm`
  - step 160000: `t=11.879750 us`, `I=0.704460 MA`, `V=8.881655 kV`,
    `r=15.200000 cm`
  - final step 161659: `I=0.707858 MA`, `V=8.817907 kV`, `crowbar=0`,
    `r=15.200000 cm`, `Lp=31.976097 nH`
  - MLX memory telemetry stayed flat (`active=0.288 MB`, cache about
    `10.529 MB`, peak about `9.793 MB`).
- Updated remaining limit:
  standalone `6 us`, `8 us`, and conservative `12 us` source-scoped endurance
  are now closed without a crowbar. Scientific acceptance remains blocked
  because strict M2 is still a low-side near miss (`1.047183 MA` vs lower bound
  `1.0485 MA`) and S1/S2 still require accepted same-scope digitized Akel
  waveform evidence and uncertainty.
- Next checks:
  troubleshoot the M2 low-side near miss by comparing full-engine
  `Lp/dLdt/phase/current` against the reduced Lee path that peaks inside band;
  continue Akel current trace digitization to turn S1/S2 into source-backed
  gates.

### 2026-05-07: CPU Snowplow Lee Current-Factor Circuit Loading

- Source-of-truth basis:
  the Lee course describes the axial `fm`/`fc` equation of motion as coupled to
  a circuit equation, defines `fc` as the current fraction effectively flowing
  in/driving the axial moving structure, and states that radial `fmr`/`fcr`
  factors are incorporated in all three radial phases. The same course gives an
  axial dynamic-resistance example where `0.5*dL/dt` drops from about `5 mOhm`
  to `3.5 mOhm` when the current factor is considered. That supports
  current-factor scaling of circuit-facing `Lp`/`dLdt`.
- Issue isolated:
  the full `SimulationEngine` path used CPU `SnowplowModel`, whose magnetic
  force already used `(fc*I)^2`, but whose axial `plasma_inductance`, axial
  `dL_dt`, and frozen axial inductance used unscaled `L_coeff`. The reduced
  `MLXSnowplow` path already scaled axial circuit inductance by `fc` and radial
  circuit inductance/back-EMF by `fcr`, and it peaked inside the Akel M2 band.
  This explained why the full-engine path stayed just below strict M2.
- Fix implemented:
  `SnowplowModel` now keeps `L_coeff` as the unscaled coaxial geometry
  coefficient, but uses explicit circuit-facing helpers:
  - axial `fc * L_coeff * z`
  - radial `fcr_eff * (mu0/2pi) * z_f * ln(b/r)`
  - matching current-factor-scaled axial/radial/reflected/post-pinch `dL/dt`
  Tests now assert this convention while preserving the unscaled geometry
  coefficient checks.
- Verification:
  `py_compile` passed for `src/dpf/fluid/snowplow.py` and
  `tests/test_snowplow_consolidated.py`; focused snowplow formula slice passed
  (`35 passed in 1.30s`); full consolidated snowplow suite passed
  (`417 passed, 1 xfailed, 5 xpassed in 11.93s`); focused Akel/PF-1000 gate
  slice passed (`5 passed, 14 xfailed in 1.40s`).
- Source-scoped no-crowbar 6 us probe after the fix:
  standalone `pf1000_akel` with `DPF_MLX_PROBE_TARGET_US=6` exited `PASSED`,
  reaching `t=6.000050 us` in `75181` steps.
  - Final reported `peak_I=1.150507 MA` at `t=5.250198 us`, inside the Akel
    shot-12581 M2 band `1.0485-1.2815 MA`.
  - step 40000: `t=3.597952 us`, `I=1.103840 MA`, `Lp=10.142424 nH`
  - step 50000: `t=4.301412 us`, `I=1.137455 MA`, `Lp=13.357013 nH`
  - step 60000: `t=5.023197 us`, `I=1.149869 MA`, `Lp=16.792884 nH`
  - final step 75181: `I=1.144742 MA`, `V=11.965346 kV`, `crowbar=0`,
    `z=47.273209 cm`, `Lp=21.569092 nH`
- Source-scoped no-crowbar 8 us radial/pinch probe after the fix:
  standalone `pf1000_akel` with `DPF_MLX_PROBE_TARGET_US=8` exited `PASSED`,
  reaching `t=8.000071 us` in `105978` steps with the same final peak
  `1.150507 MA` at `t=5.250198 us`.
  - step 80000: `t=6.295662 us`, `phase=radial`, `I=1.136085 MA`,
    `r=14.036811 cm`, `Lp=23.220228 nH`, `dLdt=6.427329 nH/us`
  - step 90000: `t=6.949331 us`, `phase=radial`, `I=1.053240 MA`,
    `r=7.214705 cm`, `Lp=29.929093 nH`, `dLdt=17.453964 nH/us`
  - step 100000: `t=7.599487 us`, `phase=pinch`, `I=0.825187 MA`,
    `r=3.078976 cm`, `Lp=38.512458 nH`, `dLdt=-12.436875 nH/us`
  - final step 105978: `phase=pinch`, `I=0.767598 MA`, `V=10.479495 kV`,
    `r=4.600753 cm`, `Lp=34.464097 nH`, `dLdt=-8.323167 nH/us`
- Updated status:
  standalone source-scoped no-crowbar M2 is now closed for the `6 us` and
  `8 us` probes. The previous no-crowbar `12 us` endurance evidence was
  generated before the radial/reflected `fcr_eff` circuit-loading correction,
  so rerun the conservative `12 us` probe before claiming post-8us endurance
  is current. S1/S2 remain blocked until accepted same-scope digitized Akel
  waveform and dip evidence with uncertainty exist.
- Next checks:
  rerun the source-scoped no-crowbar `12 us` probe with the current
  circuit-loading fix; update the Metal v2 DoD/spec/review docs so they no
  longer call M2 a low-side near miss once the rerun evidence is complete; then
  continue Akel current-trace digitization for S1/S2.

### 2026-05-07: Current-Factor-Corrected 12 us Akel Probe And Doc Cleanup

- Rerun:
  standalone `pf1000_akel` no-crowbar probe with `DPF_MLX_PROBE_TARGET_US=12`,
  `DPF_MLX_PROBE_STEPS=220000`, `DPF_MLX_PROBE_PRINT_INTERVAL=20000`, and
  `DPF_MLX_PROBE_MEMORY=1` exited `PASSED`.
- Current-factor-corrected 12 us evidence:
  the run reached `t=12.000000 us` in `159912` steps.
  - Final reported `peak_I_MA=1.150507` at `peak_t_us=5.250198`, inside the
    Akel shot-12581 M2 band `1.0485-1.2815 MA`.
  - step 20000: `t=2.012285 us`, `I=0.880408 MA`, `phase=rundown`,
    `Lp=3.827323 nH`
  - step 40000: `t=3.597952 us`, `I=1.103840 MA`, `Lp=10.142424 nH`
  - step 60000: `t=5.023197 us`, `I=1.149869 MA`, `Lp=16.792884 nH`
  - step 80000: `t=6.295662 us`, `phase=radial`, `I=1.136085 MA`,
    `r=14.036811 cm`
  - step 100000: `t=7.599487 us`, `phase=pinch`, `I=0.825187 MA`,
    `r=3.078976 cm`
  - step 120000: `t=9.147131 us`, `I=0.709868 MA`, `r=8.958310 cm`
  - step 140000: `t=10.597788 us`, `I=0.763376 MA`, `r=14.469201 cm`
  - final step 159912: `I=0.811876 MA`, `V=8.228613 kV`, `crowbar=0`,
    `r=15.200000 cm`, `Lp=22.417737 nH`, `dLdt=0.000000 nH/us`
  - MLX memory telemetry stayed flat (`active=0.288 MB`, cache about
    `10.333 MB`, peak about `9.794 MB`).
- Documentation cleanup:
  `docs/METAL_V2_DOD.md`, `docs/METAL_V2_SPEC.md`, and
  `docs/SPRINT4_VALIDATION_REVIEW.md` now record that standalone
  source-scoped no-crowbar M2 and conservative M6 endurance are current after
  the Lee current-factor circuit-loading fix. They no longer call M2 a
  low-side near miss.
- Gate wording cleanup:
  `tests/test_mlx_pf1000.py` no longer says the long xfailed gate is blocked by
  M6/CFL duration stability. The remaining blocker is source closure for S1/S2:
  accepted same-scope digitized Akel current waveform and current-dip evidence
  with uncertainty.
- Next checks:
  continue Akel current-trace digitization for S1/S2, then decide whether to
  convert the long PF-1000 fixture from an `xfail(run=False)` scientific gate
  into an opt-in endurance/regression path with a large enough step cap.

### 2026-05-07: Akel Fig. 1 Extraction Status

- Progress made:
  promoted the local Akel 2021 Fig. 1 page-3 crop into
  `KnowledgeReference/figures/akel-2021-fig1-current-waveform-shot-12581.png`.
- Figure provenance:
  the crop was made from the parity-verified local Akel PDF page-3 render at
  300 dpi. Its SHA-256 is
  `4c574525f1de413e54cd02bd06aa35d549db700270281310a3809edc54ab255e`.
- OCR/axis check:
  the extracted panel preserves the `0-10 us` x-axis, `0-1400 kA` y-axis, and
  legend entries for measured `PF1000 D2 Meas. curr. kA 1.2 Torr shot 12581`
  and computed `PF1000 D2 comp. curr. kA 1.2 Torr`.
- Draft vector extraction route:
  the current `pdftocairo` page-3 SVG separates a measured-current candidate
  as filled black paths `1987-2280` (`294` compact path elements,
  approximately `0.02-9.98 us`) and a computed-current candidate as black
  stroke paths `1942-1975` (`34` path elements, approximately `0.01-10.01 us`).
  Filled black paths `2345-2411` are legend glyphs in the white legend box and
  must be excluded. This is extraction metadata only.
- Queue update:
  `scientific_closure_digitization_queue()` now reports
  `akel_2021_fig1_current_waveform_shot_12581` as
  `extracted_not_digitized`, with figure path/hash, candidate axis calibration
  points, and draft vector path-separation metadata. The other Akel figure
  tasks remain `not_extracted`.
- Verification status:
  `python3 -m py_compile src/dpf/validation/digitization.py tests/test_digitization.py`
  passed; `python3 -m pytest tests/test_digitization.py -q` passed
  (`11 passed`);
  `python3 -m pytest tests/test_digitization.py tests/test_quality_assessment.py::TestQualityAssessment::test_scientific_accuracy_gap_report_lists_remaining_work -q`
  passed (`12 passed`); the Fig. 1 PNG SHA-256 matched the queue hash;
  `git diff --check` passed; trailing-whitespace scan over touched text files
  found no matches.
- Scientific boundary:
  this does not close S1/S2. The extracted figure is a provenance artifact
  only; accepted waveform evidence still requires measured/computed series
  arrays, overlay residuals, and independent review through
  `digitization_verification_evidence()`.
- Next checks:
  export the separated measured/computed candidate paths into arrays, keep the
  packet draft until overlay residuals and independent review exist, then add a
  comparator that can report S1/S2 as blocked-by-review rather than
  blocked-by-missing-figure.

### 2026-05-07: Akel Fig. 1 Draft Arrays

- Progress made:
  exported the separated Fig. 1 measured/computed vector candidates into a
  draft packet at
  `KnowledgeReference/digitization/akel-2021-fig1-current-waveform-shot-12581-draft-packet.json`.
- Draft packet provenance:
  SHA-256 is
  `0b8fae6147480392fcbe77eabeebc915a6a9561ec994daec32dea22859878017`.
  The loader `akel_fig1_draft_digitization_packet()` attaches the expected and
  actual packet hash and reports `draft_packet_hash_verified=True` when the
  artifact matches.
- Candidate arrays:
  measured current has `294` points from filled black paths `1987-2280`;
  computed current has `34` points from black stroke paths `1942-1975`.
  Legend glyphs `2345-2411` remain excluded (`67` filled path elements).
- Gate result:
  `digitization_verification_evidence(akel_fig1_draft_digitization_packet())`
  fails on exactly `independent_review_missing`,
  `overlay_residual_too_large`, and `review_status_not_accepted`. It does not
  fail on source, figure image, axis calibration, or required-series checks.
- Status update:
  `scientific_closure_digitization_status([akel_fig1_draft_digitization_packet()])`
  now reports `failed_task_count=1`, `open_task_count=5`, and
  `accepted_task_count=0`, so Fig. 1 is no longer reported as a missing packet.
  Gap reporting distinguishes this as a draft/failed packet needing review or
  correction.
- Verification status:
  `python3 -m py_compile src/dpf/validation/digitization.py src/dpf/validation/quality_assessment.py src/dpf/validation/__init__.py tests/test_digitization.py tests/test_quality_assessment.py`
  passed; focused digitization/gap-report pytest slice passed (`16 passed`);
  full touched-file pytest slice passed (`68 passed`); `git diff --check`
  passed; trailing-whitespace scan over touched files found no matches; draft
  packet `shasum -a 256` matched
  `0b8fae6147480392fcbe77eabeebc915a6a9561ec994daec32dea22859878017`.
- Scientific boundary:
  this remains `draft_unreviewed` evidence only. S1/S2 stay blocked until
  overlay residuals are measured, independent review is completed, and the
  review status is accepted.

### 2026-05-07: Akel Fig. 1 Internal Overlay Residual

- Progress made:
  archived the page-3 SVG used for vector extraction at
  `KnowledgeReference/digitization/akel-2021-page3.svg` and measured an
  internal round-trip residual for the Fig. 1 draft arrays.
- Source SVG provenance:
  SVG SHA-256 is
  `b045c3b7033e50bd355e025ecf7c40d96edc1ffc7fcb6ef26832fe065fe99d3f`.
- Draft packet hash:
  adding overlay metadata changed
  `KnowledgeReference/digitization/akel-2021-fig1-current-waveform-shot-12581-draft-packet.json`
  to SHA-256
  `abe4a283ee154f84f6061da8ea508d3871faf3b14dddb2d1cfc8a7a0a5f8e0e7`.
- Overlay residual method:
  reprojected draft data arrays through the Fig. 1 axis calibration and
  compared them with transformed `pdftocairo` SVG path bounding-box centers.
  This is internal vector round-trip evidence, not an independent review.
- Overlay residual result:
  combined `328` candidate points had RMS residual `0.213455189 px` and max
  residual `2.733560259 px`. Computed-current RMS was `0.000027947 px` over
  `34` points; measured-current RMS was `0.225460245 px` over `294` points.
- Gate result:
  `digitization_verification_evidence(akel_fig1_draft_digitization_packet())`
  now fails only on `independent_review_missing` and
  `review_status_not_accepted`; `overlay_residual_too_large` is no longer
  present.
- Verification status:
  `python3 -m py_compile src/dpf/validation/digitization.py src/dpf/validation/quality_assessment.py src/dpf/validation/__init__.py tests/test_digitization.py tests/test_quality_assessment.py`
  passed; `python3 -m pytest tests/test_digitization.py tests/test_quality_assessment.py -q`
  passed (`68 passed`); `git diff --check` passed; trailing-whitespace scan
  over touched files found no matches; packet and SVG SHA-256 checks matched
  the values above.
- Scientific boundary:
  S1/S2 remain blocked. The packet is still `draft_unreviewed` until
  independent review accepts it.

### 2026-05-07: Akel Waveform Digitization Readiness Status

- Progress made:
  added `pf1000_16kv_current_waveform_digitization_candidate_evidence()` as a
  data-readiness helper for the Akel Fig. 1 waveform packet.
- Current helper result:
  with the local draft packet, the helper returns `passed=False`,
  `waveform_digitization_status="blocked_by_review"`, required series present,
  overlay RMS `0.213455189 px`, and missing checks
  `["independent_review_missing", "review_status_not_accepted"]`.
- Verification status:
  `python3 -m py_compile src/dpf/validation/digitization.py src/dpf/validation/quality_assessment.py src/dpf/validation/kr_targets.py src/dpf/validation/__init__.py tests/test_digitization.py tests/test_quality_assessment.py tests/test_kr_targets.py`
  passed; `python3 -m pytest tests/test_digitization.py tests/test_quality_assessment.py tests/test_kr_targets.py -q`
  passed (`146 passed`); `git diff --check` passed; trailing-whitespace scan
  over touched files found no matches.
- Status boundary:
  downstream code can now distinguish draft waveform data blocked by review
  from missing waveform data. This is not a simulation-vs-trace comparator and
  does not close S1/S2 or tier-1 waveform validation.
