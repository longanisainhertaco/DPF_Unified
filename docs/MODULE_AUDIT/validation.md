# Module Audit: Validation

Status date: 2026-05-11

Audit scope:

- `src/dpf/validation/`
- validation-facing tests such as `tests/test_kr_targets.py`,
  `tests/test_digitization.py`, `tests/test_quality_assessment.py`,
  `tests/test_mhd_numerical_fidelity.py`, `tests/test_physics_fidelity.py`,
  `tests/test_uncertainty_budget.py`, `tests/test_circuit_field_coupling.py`,
  and `tests/test_validation_artifacts.py`

These notes are advisory only. They are not source-of-truth science.

## Intended Behavior

The validation module should be the project's evidence gate, not the project's
physics authority. Its intended job is to:

- identify which `KnowledgeReference/` sources support a claim;
- extract typed, line-referenced targets from those local sources;
- reject draft, cross-scope, missing, reconstructed, or review-blocked evidence;
- distinguish code verification from experimental validation;
- label results as `Reference`, `Preview`, `Exploratory`, `Invalid`, or other
  non-promoting classes;
- produce run manifests and validation certificates only when the linked
  evidence is accepted and same-scope;
- report exactly what is missing before a predictive or high-fidelity claim can
  be made.

This module should never decide that a simulation is scientifically valid merely
because it has a plausible waveform, an analytic verification test, a generated
dataset, or a successful optimizer run.

## Source-Of-Truth Support

Useful support found:

- `src/dpf/validation/digitization.py` explicitly says digitized data can support
  validation only when source document, figure image, axis calibration, arrays,
  and review status are traceable; it also says the module audits packets rather
  than performing interactive digitization.
- The Akel digitization queue is scoped to
  `KnowledgeReference/radiation-physics-and-chemistry-188-2021-109633.md` and
  marks the current Figure 1 packet as draft/unreviewed.
- `src/dpf/validation/artifacts.py` has fail-closed result labels and certificate
  rules: only `Reference` results with `accepted` validation status can support
  validation claims. Certificates now carry classification/readiness context and
  accepted certificates reject blocker lists. HDF5 run metadata can also carry a
  compact readiness/source-blocker summary when the run summary provides it.
- `src/dpf/validation/kr_targets.py` currently has 45 coded target factories.
  The local source audit and semantic marker audit pass, but target coverage and
  same-scope closure do not pass.
- `src/dpf/validation/source_acquisition.py` correctly states that candidate
  links are not source-of-truth until user acquisition and local
  `KnowledgeReference/` ingestion.
- `src/dpf/validation/mhd_numerical_fidelity.py` explicitly labels Tier-3 MHD
  evidence as code numerical verification and says it cannot substitute for Tier
  4 spatial or Tier 5 neutron validation.

Current local report snapshot from this audit:

- KR target count: 45.
- KR target source audit: passed.
- KR target semantic marker audit: passed.
- KR target coverage: failed because these groups are missing or partial:
  `circuit_waveform`, `phase_timing`, `spatial_temperature`, `neutron_yield`,
  and `uncertainty`.
- Same-scope target closure: failed; no passed scopes.
- Widest available scope:
  `pf1000_full_energy_2007_gribkov_scholz`.
- Widest-scope blocker groups:
  `circuit_waveform`, `neutron_anisotropy`, `neutron_detector_response`,
  `neutron_spectrum`, `neutron_timing`, `neutron_yield`, `phase_timing`,
  `spatial_magnetic_or_em`, `spatial_temperature`, and `uncertainty`.
- Akel digitization queue: 6 tasks, 0 accepted, 6 open.

## Current Implementation Summary

The package is a mix of at least nine responsibilities:

1. Public aggregation API: `src/dpf/validation/__init__.py` exports a broad set
   of validation, calibration, target, artifact, and analytic helpers from one
   namespace.
2. KR target extraction: `kr_targets.py` encodes line-referenced local targets
   and reports coverage, semantic marker checks, and same-scope closure.
3. KR corpus inventory: `kr_corpus.py` inventories DPF-relevant local markdown
   and tracks whether reviewed sources are represented by targets or explicit
   decisions.
4. Digitization gates: `digitization.py` manages Akel figure queues and verifies
   source/image/hash/review packet metadata.
5. Readiness and quality gates: `quality_assessment.py` creates validation tier
   reports, predictive-readiness reports, high-fidelity-readiness reports, and
   scientific gap reports.
6. Artifact governance: `artifacts.py` defines result labels, run manifests,
   validation certificates, and HDF5 metadata embedding.
7. Numerical verification: `mhd_numerical_fidelity.py`, `riemann_exact.py`,
   `sedov_exact.py`, `magnetized_noh.py`, and related files support code
   verification and analytic test problems.
8. Experimental comparison/calibration: `experimental_devices.py`,
   `experimental_comparison.py`, `lee_model_comparison.py`, and calibration
   modules compare waveforms, optimize Lee factors, and register devices.
9. Physics diagnostic helpers: modules such as `pinch_physics.py`,
   `dynamic_zpinch.py`, and `bennett_equilibrium.py` compute analytic or
   semi-empirical diagnostics that are useful but not automatically validated.

The newer parts are mostly blocker-aware. The older comparison/calibration and
diagnostic parts are much easier to misuse because they can produce numbers
without requiring accepted same-scope evidence.

Calibration output now has a shared fail-closed provenance helper:
`calibration_provenance_metadata()` labels optimized fits as
`optimized_parameter_fit`, `Calibration Fit`, and `not_validation_evidence`.
The Gradio calibration bridge attaches those fields to calibration output and
prints a source-authority note in calibration markdown.

2026-05-11 formulary/local-KR update:

- `src/dpf/validation/pinch_physics.py::coulomb_mean_free_path()` now defaults
  to the NRL electron-ion Coulomb-log branches for its electron-ion mfp helper
  instead of the electron-electron expression.
- `src/dpf/validation/lee_model_comparison.py` now applies Lee axial `fc`
  scaling to circuit-facing axial inductance and `dLp_dt`.
- `src/dpf/validation/lee_model_comparison.py` now keeps radial `fcr` separate
  from axial `fc` and applies it to radial inductance, radial `dLp/dt`,
  radial/reflected force, frozen/post-crowbar radial inductance, and metadata.
  This closes the helper-level current-factor audit only; waveform validation
  remains blocked by accepted same-scope evidence.

## Why It Does Not Work Yet

The module has good guardrails, but the project is not validation-ready because
the evidence behind those gates is incomplete.

- The package can report KR source authority for coded targets, but source
  authority is not the same as target completeness.
- The semantic audit only checks that cited line windows contain expected terms.
  `kr_targets.py` explicitly notes that this is not a replacement for human
  scientific review.
- The current Akel plot data are not accepted evidence. The queue has six local
  figure tasks and no accepted digitization packets.
- Same-scope closure fails. Cross-device or cross-shot coverage is not enough
  for predictive end-to-end DPF claims.
- Calibration can still optimize against device registry values that include
  reconstructed or estimated waveforms. It now carries fail-closed provenance
  labels, but those labels do not prove the model or source-close the registry.
- `ExperimentalDevice` data mix measured, reconstructed, reference-only, and
  unverified fields in one object. Some entries state KR verification while
  individual fields remain estimated or inherited.
- `quality_assessment.py` is a useful gate, but its older `assess_quality`
  section still computes grades from plausibility checks. Those grades must not
  be treated as scientific validation.
- `pinch_physics.py` cites several papers and formulas in module comments; under
  the current rule, those claims must be re-checked against local
  `KnowledgeReference/` records before they drive validation tasks.
- The 2026-05-11 formulary pass fixed one mfp convention issue in
  `pinch_physics.py`, but it did not source-verify every analytic helper in
  validation.
- MHD numerical verification is useful, but it is Tier 3 code verification only.
  It does not validate density, magnetic field, temperature, neutron timing,
  spectrum, anisotropy, detector response, or UQ.

## Stale Or Inaccurate Assumptions

- The package-level docstring says "Validation suite for DPF simulations against
  experimental data"; this is too broad. Much of the package is actually target
  management, code verification, calibration, or readiness reporting.
- `experimental_devices.py` contains historical comments and values inherited
  from older debates, IPFS/RADPF presets, reconstructions, and Type B estimates.
  These must be individually re-audited.
- PF-1000 16 kV waveform fields are marked as reconstructed from a 27 kV Scholz
  waveform in the device registry. That should never be used as measured Akel
  S1/S2 waveform evidence.
- Calibration code now adds fail-closed provenance/status fields to the active
  UI calibration outputs. Remaining legacy calibration classes should still be
  treated cautiously if consumed outside that path.
- Some validation tests create synthetic accepted packets. Those tests are good
  for gate behavior but do not prove that real Akel data are accepted.
- Some analytic helper modules include formula citations that may be accurate
  but are not automatically local-KR-verified in their current implementation.

## Trustworthy Tests

These tests are useful because they preserve guardrails or blocker behavior:

- `tests/test_digitization.py`: verifies hash/path/review requirements and keeps
  draft digitization from passing.
- `tests/test_validation_artifacts.py`: verifies result classification,
  manifest consistency, certificate context persistence, and certificate
  rejection for blocked/cross-scope/blocker-carrying evidence.
- `tests/test_kr_targets.py`: verifies target registry shape, source audit, and
  same-scope target reports.
- `tests/test_quality_assessment.py`: useful where it checks that readiness
  remains blocked without required evidence.
- `tests/test_mhd_numerical_fidelity.py`: useful for Tier-3 packet behavior,
  especially where it preserves code-verification boundaries.
- `tests/test_physics_fidelity.py`, `tests/test_uncertainty_budget.py`, and
  `tests/test_circuit_field_coupling.py`: useful where they verify missing
  components stay non-passing.
- `tests/test_calibration_provenance.py`: verifies calibration metadata and UI
  markdown keep optimized fits out of validation claims.

## Suspect Tests

These tests should be read carefully before trusting their meaning:

- Tests using synthetic complete packets prove validator mechanics only.
  They do not prove real-world validation data exist.
- Tests comparing to `ExperimentalDevice` registry values may inherit stale or
  reconstructed data from that registry.
- Calibration tests can prove optimizer behavior without proving that the
  optimized parameter is scientifically meaningful. Provenance-label tests prove
  claim hygiene only.
- Any test that accepts a scalar yield, peak current, or broad temperature range
  as sufficient validation is suspect unless it also verifies same-scope source,
  uncertainty, and required observables.

## Future-Agent Notes

These notes are not authoritative. Re-check current code before acting.

- Treat `validation` as two modules pretending to be one: strict evidence gates
  and legacy validation/calibration utilities. The strict gates should become
  the only path to user-facing validation claims.
- Do not delete legacy helpers immediately. First classify them as `accepted`,
  `verification_only`, `diagnostic_only`, `calibration_only`, `reference_only`,
  `blocked`, or `unverified`.
- Prefer adding adapters that wrap legacy outputs in explicit classifications
  over allowing raw legacy dictionaries to flow into reports.
- The highest-value next work is not more thresholds. It is closing source
  packets: accepted Akel digitization, same-scope target completeness, detector
  response, and UQ.
- Keep code-verification and experimental validation vocabulary separate in
  every API and UI surface.
- Any future "validation ready" signal must include the validation scope, source
  URIs, line ranges, evidence packet hashes, review status, and uncertainty.

## Backlog Entries

See `BACKLOG.md` entries `VAL-001` through `VAL-013`.

Priority interpretation:

- `VAL-003`, `VAL-004`, and `VAL-005` protect against false validation claims.
- `VAL-006` is the main scientific closure path.
- `VAL-001` and `VAL-002` remain architecture/product hygiene needed before
  future agents can work safely. `VAL-010` is now closed for propagation
  guardrails. `VAL-007` is now closed for active calibration outputs, with
  broader legacy-class cleanup still subject to current-code review before new
  consumers rely on it.
- `VAL-008` and `VAL-009` are evidence-quality improvements.
