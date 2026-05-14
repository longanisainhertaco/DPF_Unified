# Validated Physics Pipeline Plan

Date: 2026-05-12

Status: planning baseline for review and implementation

This plan defines the pipeline for promoting local source material into
validated physics evidence. It is intentionally conservative. A local source,
typed target, plot digitization, table extraction, formula check, or uncertainty
budget can support validation only after it passes the gates below and remains
inside the same device, shot, operating condition, observable, and diagnostic
scope.

The scientific source rule remains unchanged: local `KnowledgeReference/`
records are the only scientific source authority. This document is a plan and
does not accept any new target value, plotted curve, table value, formula,
uncertainty value, validation threshold, or simulation result.

## Current Inputs

| Input | Current status | Pipeline use |
| --- | --- | --- |
| `docs/USER_PDF_MAY12_SOURCE_VALIDATION_2026_05_12.md` | 28 promoted source records, 7 stage-only records, 5 source-validated target candidates, 0 failures | Source-authority intake gate |
| `docs/USER_PDF_MAY12_TARGET_TRIAGE_2026_05_12.md` | 5 target-extraction candidates and 23 method/context records | Target-review priority queue |
| `docs/TARGET_EXTRACTION_DIGITIZATION_2026_05_11.md` | 5 earlier target records started, 36 crop candidates, 0 accepted validation packets | Existing target/digitization workbench |
| `docs/A14_REMAINING_EXTRACTION_BACKLOG_2026_05_11.md` | 9 reviewable draft packets, 18 ready-not-started crops, 9 manual-review crops, 1 blocked crop | Independent review and extraction backlog |
| `src/dpf/validation/digitization.py` | Existing `digitization_verification_evidence()` gate | Provenance and review gate for figures/tables |
| `src/dpf/validation/kr_targets.py` | Typed KR target helper surface | Target registry and same-scope validation input |
| `src/dpf/validation/uncertainty_budget.py` | Existing UQ audit requirements | UQ completeness and propagation gate |
| `docs/FORMULARY_CODE_AUDIT_2026_05_11.md` | Local-formulary code audit with fixed and blocked formula surfaces | Formula validation backlog and method |
| `docs/DPF_REQUIREMENTS_BASELINE.md` | Candidate SRS/RTM baseline, Doorstop not initialized | Future traceability import |
| `docs/FIRST_PRINCIPLES_FINISH_LINE_PLAN.md` | Active first-principles execution specification from engineering probe to accepted PF-1000/Akel simulator | Orders package-native execution, limiter removal, numerical verification, startup, coupling, dimensionality, physics closure, evidence, UQ, and certificate work |

## Relationship To First-Principles Development

This document defines how evidence becomes accepted validation evidence.
`docs/FIRST_PRINCIPLES_FINISH_LINE_PLAN.md` defines what must be built, which
implementation gates must pass, and in what order for the first-principles
PF-1000/Akel path. The two plans are intentionally separate:

- the finish-line plan owns solver/startup/coupling/physics implementation
  sequencing;
- this validated-physics pipeline owns source review, digitization, UQ,
  comparator binding, same-scope packet assembly, and certificate gates;
- neither plan can promote an engineering probe or draft packet without accepted
  same-scope evidence.

## Output Definition

A scientific item is `validated_physics_evidence` only when all required gates
for its evidence type pass and an evidence packet can answer these questions:

- What exact local source record, hash, page, line range, figure, table, or
  formula supports the claim?
- What observable is being validated?
- What device, shot, operating condition, geometry, fill gas, and diagnostic
  scope apply?
- What unit system and dimensional normalization were used?
- What uncertainty applies, how was it derived, and how is it propagated to the
  pass/fail rule?
- Who extracted it, who independently reviewed it, and what changed after
  review?
- Which simulation output field or diagnostic is compared against it?
- Which validation tier or readiness gate can cite it?

If any answer is missing, the item remains a candidate or blocker.

## Acceptance States

| State | Meaning | Can support physics validation |
| --- | --- | --- |
| `stage_only_non_authority` | Local file is retained but is not physics authority | No |
| `source_validated` | Source identity, hash, text parity, and classification are checked | No |
| `source_line_reviewed` | Candidate lines/pages/figures/tables/formulas are reviewed and accepted as relevant | No |
| `typed_target_draft` | Target is structured with device/scope/units but not independently reviewed | No |
| `digitization_draft` | Figure/table packet exists with hashes, units, axis/table metadata, and arrays | No |
| `formula_audit_draft` | Formula mapping to local source exists but is not accepted | No |
| `independent_review_ready` | Packet has all technical fields and is ready for reviewer decision | No |
| `accepted_source_extraction` | Source extraction or digitization passed independent review | Yes, for source data only |
| `comparator_bound` | Accepted source data is attached to a tested simulation-output comparator | Yes, for that observable |
| `same_scope_packet_ready` | Required observables for one device/scope are assembled with UQ and comparators | Yes, for packet-level validation |
| `validated_physics_evidence` | Same-scope packet, comparator, UQ, source, and certificate gates pass | Yes |

No state may be skipped. Automation should fail closed on unknown states.

## Canonical Evidence Fields

Every evidence packet should use these fields or an equivalent typed dataclass:

| Field | Required for | Notes |
| --- | --- | --- |
| `evidence_id` | all | Stable ID, for example `VP-AKEL-2021-FIG1-S12581-CURRENT` |
| `evidence_type` | all | `scalar_target`, `curve_target`, `table_target`, `formula`, `uncertainty`, `method_context`, `comparator`, `same_scope_packet` |
| `status` | all | One of the acceptance states above |
| `validation_scope` | all | Device/shot/campaign scope, never generic when used for pass/fail |
| `device`, `shot`, `bank`, `fill_gas`, `pressure`, `geometry` | physics targets | Include `unknown` only if not used for same-scope closure |
| `observable_group` | physics targets | Circuit, phase, spatial, neutron, detector, field coupling, UQ, formula |
| `observable_name` | physics targets | Exact compared quantity |
| `source_path` | all scientific evidence | Must be under `KnowledgeReference/` |
| `source_sha256` | all scientific evidence | Must match local file |
| `source_pdf_path`, `source_pdf_sha256` | figure/table/formula where available | Required when source geometry or visual data matters |
| `source_lines` | target/formula/uncertainty | Required for text-derived values |
| `page`, `figure_id`, `table_id`, `equation_id` | visual/table/formula items | Required when applicable |
| `raw_value`, `raw_units`, `normalized_value`, `normalized_units` | scalar/table/formula | Preserve both source and SI-normalized forms |
| `digitized_series` | curves | Arrays with quantities, units, and point count |
| `axis_calibration` | curves | Calibration points, units, transform, residuals |
| `table_matrix` | tables | Extracted values plus header/unit mapping |
| `formula_source_latex_or_text` | formulas | Source form before code mapping |
| `code_symbol_map` | formulas | Mapping from source symbols to code variables |
| `dimensional_check` | formulas | SI unit proof or explicit source-unit conversion |
| `uncertainty_budget` | validation targets | Measurement, input, numerical, model-form, shot-to-shot, propagated |
| `acceptance_rule` | comparators | Tolerance rule and uncertainty use |
| `review` | all accepted evidence | Reviewer, date, decision, notes, defects, resolution |
| `linked_tests` | comparator/certificate | Test paths and commands |
| `linked_requirements` | SRS/RTM rows | Candidate IDs until Doorstop import |

## Pipeline Stages

### Stage 0: Source Authority Intake

Goal: prove that the local source exists, is correctly identified, and is
eligible for scientific review.

Inputs:

- Local PDF or existing local `KnowledgeReference/` record.
- Promotion/source-fidelity reports.
- Hashes and text parity outputs.

Methods:

- SHA-256 identity check.
- Duplicate detection by hash, not just title.
- Bad-title filtering for generic publisher cover pages.
- Text parity and source-fidelity recovery for captions, tables, formula-like
  lines, numeric contexts, uncertainty contexts, and image blocks.

Guardrails:

- Do not promote stage-only, AI-only, social-science, or unrelated files to
  physics authority.
- Do not treat recovered table/text snippets as accepted target values.
- Do not treat review/context papers as primary target sources unless a
  specific source-backed validation need is mapped.

Acceptance:

- Source has local KR Markdown/JSON or is explicitly stage-only.
- Hash and source mapping are stable.
- Status is `source_validated`.

Artifacts:

- Intake, promotion, source-fidelity, and source-validation reports.

### Stage 1: Source-Line Review

Goal: identify exactly which source locations may support scientific targets,
formulas, uncertainty values, or digitization packets.

Inputs:

- `source_fidelity_review` JSON sections.
- Candidate triage reports.
- Local PDF renders when visual geometry matters.

Methods:

- Human or agent-assisted line-window review.
- Page, figure, table, equation, and caption mapping.
- Rejection reason logging for non-target or context-only passages.

Guardrails:

- Use page/line/figure/table identity before extracting numbers.
- Keep review papers separate from primary experimental target papers.
- Keep source scopes separate, especially Akel 16 kV versus full-energy
  PF-1000 material.

Acceptance:

- Each candidate has `source_lines` or page/figure/table/equation identity.
- Each accepted candidate has an observable group and target role.
- Each rejected candidate has a reason.

Artifacts:

- `docs/*_SOURCE_LINE_REVIEW_*.md` and `.json`.
- Candidate target backlog updates.

### Stage 2: Typed Target Extraction

Goal: convert reviewed source locations into typed observable targets without
yet declaring them accepted validation evidence.

Inputs:

- Source-line review packets.
- Existing `kr_targets.py` conventions.
- Unit and device-scope metadata.

Methods:

- Dataclass or JSON-schema target records.
- Unit normalization to SI while preserving source units.
- Source field coverage checks.
- Duplicate and cross-scope detection.

Guardrails:

- A typed target is not a comparator and not a validation pass.
- Do not fill missing uncertainty from assumptions unless clearly marked as
  model-form or provisional.
- Do not average or combine shots unless the source gives a valid aggregation
  rule and uncertainty.

Acceptance:

- Target has source path, hash, line/page identity, device/scope, observable,
  units, value/array/table reference, status, and review state.
- Tests assert source paths exist, hashes match, line windows are in bounds, and
  required fields are present.

Artifacts:

- `src/dpf/validation/kr_targets.py` records or a future generated target
  registry.
- `tests/test_kr_targets.py` coverage.

### Stage 3: Figure And Plotted-Curve Digitization

Goal: convert plotted curves into auditable digitized arrays with visual and
numeric provenance.

Inputs:

- Reviewed figure candidates.
- Local source PDF and KR record.
- Rendered page and cropped figure image.

Methods:

- Deterministic page rendering, crop hashing, and source-PDF hashing.
- Axis calibration with residuals and explicit units.
- Series extraction with point count and labels.
- Overlay residual check against the source image or vector geometry.
- Independent review of extracted arrays and overlay.

Guardrails:

- Crop candidates are not digitized data.
- Axis scaffolds with no `digitized_series` cannot pass.
- Overlapping or ambiguous curves must become blockers, not guessed arrays.
- Legend glyphs, annotations, and gridlines must be explicitly excluded.

Acceptance:

- Packet passes `digitization_verification_evidence()`.
- `independent_review_count >= 1`.
- `review_status == "accepted"`.
- Figure packet matches the queue task ID, source path, hash, page, figure ID,
  source line window, and required series names.

Artifacts:

- `KnowledgeReference/digitization/*.json`.
- Figure/crop files under `KnowledgeReference/figures/`.
- Review handoff and accepted review reports.
- `tests/test_digitization.py` coverage.

### Stage 4: Table Digitization And Table Extraction

Goal: convert tables into typed matrices or scalar target groups with table
provenance.

Inputs:

- Reviewed table candidates.
- Local PDF render/crop if table geometry matters.
- Extracted table matrices from source-fidelity review.

Methods:

- Header, unit, row, and column mapping.
- Table crop hash when visual table layout is needed.
- Cross-check automated extraction against rendered table.
- Independent review of row/column interpretation.

Guardrails:

- OCR or auto-extracted matrices are draft until reviewed.
- Multi-row headers and unit rows require explicit mapping.
- Tables copied from review papers still need primary-source status decisions.

Acceptance:

- Table packet passes the table branch of `digitization_verification_evidence()`
  or an equivalent table-specific gate.
- Header/unit mapping, table ID, page, source hash, and review metadata are
  present.

Artifacts:

- Table extraction packets in `KnowledgeReference/digitization/`.
- Tests covering table hash checks and required metadata.

### Stage 5: Formula Validation

Goal: prove that coded formulas match local source formulas, including units,
constants, regimes of validity, and implementation assumptions.

Inputs:

- Local source formula lines, equation IDs, and context.
- Current code modules and tests.
- Existing formulary audit reports.

Methods:

- Source formula transcription with source line/equation identity.
- Symbol-to-code mapping.
- SI dimensional analysis and source-unit conversion.
- Numeric regression fixtures over representative inputs.
- Regime/validity metadata attached to code where relevant.

Guardrails:

- Correct formula implementation is not the same as DPF validation.
- Unknown-provenance empirical fits remain fail-closed.
- Formulas from general references can validate a local model component but do
  not create same-shot experimental validation.
- Multiple source conventions must be resolved before code changes.

Acceptance:

- Formula packet identifies source equation, symbol mapping, units, validity
  domain, code path, tests, and residual or exact-match logic.
- Tests fail if constants, units, or dimensional assumptions drift.
- Blocked surfaces remain labeled as not source-closed.

Artifacts:

- Formula audit packets.
- Focused tests such as the existing formulary audit tests.
- Module metadata for source status.

### Stage 6: Uncertainty Extraction And Propagation

Goal: attach defensible uncertainty to every target and validation decision.

Inputs:

- Published error bars, standard deviations, calibration uncertainty, detector
  response uncertainty, input uncertainty, numerical uncertainty, model-form
  uncertainty, and shot-to-shot variation.
- Existing `uncertainty_budget.py` audit requirements.

Methods:

- Extract uncertainty values with source lines and units.
- Normalize to the compared observable.
- Separate measured uncertainty from assumed/model uncertainty.
- Propagate uncertainty through interpolation, derived diagnostics, and
  comparator rules.
- Record the acceptance rule, for example overlap with confidence interval,
  normalized residual, NRMSE with confidence band, or sigma-scaled tolerance.

Guardrails:

- Missing uncertainty blocks validation unless a documented tier explicitly
  permits a qualitative or bounded claim.
- Do not infer uncertainty from plot thickness or visual noise without a
  reviewed method packet.
- Do not mix shot-to-shot variability with measurement error unless the source
  supports it.

Acceptance:

- Every comparator-bound target has a populated uncertainty budget or an
  explicit blocker.
- UQ audit passes for the validation tier being claimed.

Artifacts:

- UQ packets linked to target IDs.
- `tests/test_uncertainty_budget.py` coverage.

### Stage 7: Comparator Binding

Goal: connect accepted targets to simulation output fields with tested
comparison logic.

Inputs:

- Accepted source extraction or formula packet.
- Simulation output manifest and diagnostics.
- Units, interpolation rules, time base, and uncertainty budget.

Methods:

- Comparator registry by observable group.
- Unit conversion and time/space alignment.
- Interpolation method with uncertainty impact recorded.
- Pass/fail rule that uses uncertainty.
- Result classification that fails closed when evidence is incomplete.

Guardrails:

- Engineering probes may test stability only; they do not satisfy scientific
  comparators.
- MLX/Metal preview output cannot become Reference without accepted same-scope
  evidence and authority promotion.
- Cross-device or cross-shot comparisons can be context only unless explicitly
  validated as a transfer rule.

Acceptance:

- Comparator has source-backed target, tested output-field mapping, unit
  normalization, uncertainty-aware metric, and manifest/certificate linkage.

Artifacts:

- Comparator code and tests.
- Per-run evidence JSON.
- Validation certificate inputs.

### Stage 8: Same-Scope Packet Assembly

Goal: assemble enough accepted evidence to validate one claimed scope without
mixing incompatible sources.

Inputs:

- Comparator-bound accepted targets.
- Readiness tier definitions.
- Source acquisition and closure queue.

Methods:

- Scope matrix for device, shot, pressure, voltage, geometry, fill gas,
  diagnostic, observable group, and source family.
- Evidence completeness check by tier.
- Duplicate/conflict resolution.
- Blocker report for missing observables.

Guardrails:

- Akel 16 kV evidence cannot close full-energy PF-1000 claims.
- Method/context records cannot close experimental target tiers.
- A same-scope packet must not silently fall back to generic Lee/RADPF example
  targets.

Acceptance:

- Packet passes same-scope grouping.
- Required tiers for the claimed result pass.
- Missing groups are explicit blockers.

Artifacts:

- `docs/*_SAME_SCOPE_PACKET_*.md` and `.json`.
- Tests around same-scope packet validation.

### Stage 9: Certificate And Release Gate

Goal: allow user-facing validation claims only when evidence, comparators, UQ,
scope, and requirements traceability all pass.

Inputs:

- Same-scope packet.
- Run manifest.
- Result classification.
- Candidate SRS requirement IDs.

Methods:

- Validation certificate generation.
- Requirement-to-evidence trace links.
- CI checks that block promotion from draft/candidate states.
- User-facing labels: Reference, Preview, Derived Diagnostic, Exploratory,
  Superseded, or Invalid.

Guardrails:

- No certificate for blocked, draft, cross-scope, or missing-review evidence.
- No release claim without linked artifacts and commands.
- Findings docs remain active until Doorstop is initialized and validated.

Acceptance:

- Certificate writes only after all linked gates pass.
- RTM/Doorstop links identify verification method and acceptance evidence.

Artifacts:

- Validation certificate JSON.
- SRS traceability rows.
- Findings updates.

## Automation Backlog

| ID | Priority | Task | Goal | Guardrails | Acceptance evidence |
| --- | --- | --- | --- | --- | --- |
| VP-P0-001 | P0 | Define canonical evidence dataclasses and JSON schema | Make all target, curve, table, formula, UQ, comparator, and packet evidence machine-checkable | Unknown fields warn; missing required fields fail closed | Schema tests and sample packets |
| VP-P0-002 | P0 | Build source-line review generator | Convert source-fidelity contexts into reviewed or rejected candidate packets | No auto-acceptance from OCR or captions | `*_SOURCE_LINE_REVIEW_*.json` plus tests |
| VP-P0-003 | P0 | Build typed target validator | Enforce source path/hash/line/scope/unit fields for target records | Typed target still cannot pass validation alone | `tests/test_kr_targets.py` expansion |
| VP-P0-004 | P0 | Generalize digitization packet validator | Cover figure and table packets beyond Akel/A14 hardcoded helpers | Review status and hashes remain mandatory | `tests/test_digitization.py` expansion |
| VP-P0-005 | P0 | Build formula evidence registry | Tie coded formulas to source formulas, units, validity, and tests | Formula correctness is not DPF run validation | Formula audit JSON plus focused tests |
| VP-P0-006 | P0 | Build uncertainty packet validator | Require UQ components or explicit blockers for every comparator-bound target | No hidden tolerance defaults | `tests/test_uncertainty_budget.py` expansion |
| VP-P0-007 | P0 | Build comparator registry | Map accepted targets to simulation output fields and uncertainty-aware metrics | No engineering-probe promotion | Comparator tests and evidence JSON |
| VP-P0-008 | P0 | Build same-scope packet assembler | Prevent cross-scope mixing and show missing observables | Akel/full-energy PF-1000 separation enforced | Same-scope matrix tests |
| VP-P0-009 | P0 | Bind validation certificate to same-scope packets | Allow certificate only after packet, comparator, UQ, and review gates pass | Fail closed on drafts and blockers | Certificate negative/positive tests |
| VP-P1-010 | P1 | Add Doorstop import plan for validation requirements | Move candidate IDs into a versioned trace tree after review | Do not import unstable IDs prematurely | Doorstop tree validation |
| VP-P1-011 | P1 | Add reviewer handoff workflow | Give independent reviewers stable packets and defect/resolution fields | Reviewers cannot edit source truth silently | Handoff report plus accepted/rejected decisions |
| VP-P1-012 | P1 | Add CI promotion guard | Prevent accepted labels unless all gates pass | Draft packets remain blocked in CI | CI or local script gate |

## Candidate Requirement Additions

These IDs are proposed for the SRS/RTM after review. They should not be imported
into Doorstop until the team accepts the ID scheme.

| ID | Requirement | Status | Verification method |
| --- | --- | --- | --- |
| DPF-VV-011 | The system shall represent every scientific target, curve, table, formula, uncertainty value, comparator, and same-scope packet as typed evidence with local source provenance. | planned | test, inspection |
| DPF-VV-012 | The system shall require independent accepted review before any digitized figure or table data can support validation. | partial | test, review |
| DPF-VV-013 | The system shall require formula evidence packets for source-closed coded formulas, including source equation, symbol mapping, units, validity domain, and regression tests. | planned | analysis, test |
| DPF-VV-014 | The system shall require uncertainty extraction and propagation before any quantitative validation pass/fail threshold can be accepted. | planned | analysis, test |
| DPF-VV-015 | The system shall bind accepted targets to simulation output through tested comparators before any validation certificate can be written. | planned | test, analysis |
| DPF-VV-016 | The system shall assemble same-scope validation packets and reject cross-device, cross-shot, or cross-configuration evidence mixing unless a reviewed transfer rule exists. | planned | test, inspection |

## Immediate Execution Order

1. Finish independent review handoff for the existing A14 reviewable draft
   packets and record accepted/rejected decisions without changing the source of
   truth.
2. Promote no A14 or Akel digitization packet until
   `digitization_verification_evidence()` passes with accepted review metadata.
3. Start source-line review for the five May 12 target candidates:
   `10.1088@1742-6596@370@1@012059.pdf`, `kasperczuk2002.pdf`,
   `kubes2020.pdf`, `trunk1975.pdf`, and `lindemuth1982.pdf`.
4. Build the canonical evidence schema and typed target validator before adding
   more target records, so new extraction work lands in the final shape.
5. Extend table and formula pipelines in parallel with curve digitization, but
   keep them out of same-scope validation until UQ and comparator binding exist.
6. Add UQ packet checks before any target becomes comparator-bound.
7. Build one narrow same-scope packet first, then expand tiers after the
   certificate path proves it can fail closed.

## Definition Of Done

The pipeline is implementation-complete when:

- Source-intake, source-line review, typed target extraction, digitization,
  formula evidence, UQ, comparator binding, same-scope assembly, and certificate
  gates are all executable from scripts or tests.
- Every gate writes machine-readable evidence and a reviewer-readable Markdown
  report.
- Draft and blocked packets fail closed in tests.
- Accepted packets require independent review metadata.
- Same-scope validation rejects Akel/full-energy PF-1000 mixing and other
  cross-scope combinations.
- `CodexFindings.md` and `CortexFindings.md` identify the current pipeline
  stage for each active target candidate.
- Candidate SRS/RTM rows are ready for Doorstop import after review.

## Current Blockers

- No plotted-curve or table digitization packet is accepted for validation yet.
- A14 and Akel packets with draft arrays still require independent review.
- May 12 target candidates require source-line review before typed extraction.
- Formula code audits closed several mismatches, but source-closed formula
  packets and a general formula registry do not exist yet.
- Uncertainty values are indexed in source-fidelity reviews, but not yet
  normalized, propagated, or attached to comparator pass/fail rules.
- Comparator binding and same-scope packet assembly are not yet implemented as a
  general pipeline.
