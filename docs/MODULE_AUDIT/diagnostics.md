# Diagnostics Module Audit

Status date: 2026-05-11

This note is advisory. It is not source-of-truth science and must be re-checked
against current code and reviewed `KnowledgeReference/` evidence before any
implementation work.

## Scope

Primary files reviewed:

- `src/dpf/diagnostics/`
- selected app-layer diagnostic helpers in `app_engine.py` and `app_mhd.py`
- diagnostic-adjacent tests for yield, nTOF, energy balance, Thomson,
  synthetic diagnostics, scaling laws, regimes, plasmoids, and filamentation

## Intended Behavior

The diagnostics package is intended to provide scalar and synthetic diagnostic
outputs around DPF runs:

- neutron yield and neutron time-of-flight outputs
- beam-target and beam-tracker estimates
- yield accumulation
- energy bookkeeping
- Thomson scattering, interferometry, and x-ray synthetic diagnostics
- HDF5/checkpoint output
- regime, instability, plasmoid, filamentation, runaway-electron, and topology
  helpers

## Source-Of-Truth Support Found Or Missing

Supported locally:

- Local Lee/Saw material supports a beam-target yield form and beam-energy
  relationship used by `src/dpf/diagnostics/beam_target.py`.
- Local KR material supports the broad DPF framing that neutron production can
  include both thermonuclear and beam-target/kinetic contributions.
- Local PANDA material supports qualitative nTOF/anisotropy diagnostic concepts
  and calibration limitations.
- Local Bosch-Hale material supports DD cross-section/reactivity tables and
  validity ranges. That supports the coefficient source for DD reactivity, but
  not every downstream neutron diagnostic claim.

Missing or not yet strong enough:

- Thomson, nTOF spectra, x-ray filters/emissivity, instability, regime,
  plasmoid, shear, and runaway formulas need exact local KR source closure.
- Bosch-Hale support does not by itself validate total DPF neutron yield,
  timing, anisotropy, detector response, or uncertainty.
- App-layer neutron estimates use simplified volume, confinement, and pinch
  assumptions that are not validation-grade.

## Current Implementation Summary

Diagnostics currently mix several classes of behavior:

- source-backed or partially source-backed formulas, such as Bosch-Hale DD and
  Lee/Saw beam-target estimates
- synthetic diagnostics for optical, x-ray, and neutron detector-like outputs
- state bookkeeping for energy, HDF5, checkpoints, and derived fields
- broad heuristic classifiers for regimes, instabilities, plasmoids, filaments,
  and runaway electrons

This is useful as an implementation surface. The package now carries a
fail-closed evidence manifest in `src/dpf/diagnostics/evidence_manifest.py`
that marks every public diagnostics surface as blocked-by-review, missing,
engineering-probe, or synthetic-only. It has no accepted validation entries.

2026-05-09 implementation update:

- `BeamTracker` no longer passes joules as `V_pinch` to
  `beam_target_yield_rate()`. It converts mean kinetic energy to the
  beam-target helper's `V_pinch` equivalent, exposes `equivalent_V_pinch`, and
  labels the result as `engineering_estimate_not_validation`.
- HDF5 `max_div_B` remains a compatibility scalar, but it is now explicitly
  marked as `rough_array_metric_not_physical_divergence` with `T/cell` units
  and `validation_status="not_validation_evidence"`.

2026-05-10 implementation update:

- Added `src/dpf/diagnostics/evidence_manifest.py`, a conservative manifest
  that classifies each diagnostics module/output group as
  `blocked-by-review`, `missing`, `engineering-probe`, or `synthetic-only`.
  No diagnostics entry is marked accepted or allowed to support validation
  claims.
- The manifest includes the public symbols for each diagnostics module so
  formula/output coverage can be tested without importing every diagnostic
  implementation.
- Added `src/dpf/diagnostics/test_lanes.py` and pytest collection markers for
  diagnostics-oriented tests. Current diagnostics tests are classified as
  engineering-smoke, source-component-check, source-blocked, or synthetic-only;
  none are marked as source-backed diagnostics validation tests.

2026-05-11 formulary update:

- `src/dpf/diagnostics/plasma_regime.py::magnetic_reynolds_number()` now uses
  the centralized corrected Spitzer resistivity instead of duplicating an
  uncorrected classical expression.
- `src/dpf/diagnostics/regime_classifier.py` now uses the NRL electron-ion
  Coulomb-log branches and a corrected diagnostic Spitzer resistivity form for
  the regime classifier.
- These fixes improve formula correctness for regime diagnostics only; they do
  not source-close Thomson, nTOF, x-ray filter/emissivity, instability,
  plasmoid, shear, runaway, or detector-response validation.

## Why It Likely Fails Or Is Unverifiable

- `src/dpf/diagnostics/beam_tracker.py` now has a unit guardrail for the
  beam-target helper call, but its yield remains an engineering estimate based
  on mean kinetic energy and placeholder current/confinement assumptions.
- `src/dpf/diagnostics/hdf5_writer.py` now labels `max_div_B` as a rough array
  metric. It is not a reliable physical divergence diagnostic until a separate
  geometry/grid-spacing-aware diagnostic is implemented.
- `YieldTracker` uses peak temperature and capped peak density shortcuts in some
  paths; those are useful engineering summaries but weak validation evidence.
- Several diagnostics are tested for shape, positivity, monotonicity, or broad
  magnitude rather than source-backed acceptance.
- The plasma-regime `ND < 1` MHD-valid claim cites material not yet established
  as local source-of-truth in this pass.
- Regime helper outputs remain source-component diagnostics, not global
  validation gates.

## Stale Or Inaccurate Assumptions

- `src/dpf/diagnostics/Troubleshooting.md` now has a 2026-05-09 audit-status
  preface, but older entries remain historical review notes and are not
  authority for current validation claims.
- Beam-target anisotropy and dwell/transit assumptions need explicit source
  status.
- Synthetic diagnostics can easily be mistaken for measurement validation if
  output labels do not distinguish modeled observables from validated detector
  response.

## Trustworthy Tests Versus Suspect Tests

More trustworthy as engineering smoke tests:

- `tests/test_energy_balance.py`
- `tests/test_yield_tracker.py`
- `tests/test_thomson_api.py`
- basic pusher checks in `tests/test_beam_tracker.py`
- `tests/test_diagnostics_evidence_manifest.py` for evidence-lane coverage and
  fail-closed validation labels
- `tests/test_diagnostics_test_lanes.py` for diagnostics test-lane coverage

Limited or suspect as science validation:

- `tests/test_neutron_yield.py`
- `tests/test_synthetic_diagnostics.py`
- `tests/test_scaling_laws.py`
- `tests/test_regime_classifier.py`
- `tests/test_plasmoid.py`
- `tests/test_filamentation.py`

These mostly prove mechanics or broad plausibility, not accepted diagnostic
truth.

## Future-Agent Notes

These notes are not authority. Re-check all line numbers and current behavior
before editing.

- Keep the diagnostics evidence manifest and diagnostics test-lane manifest
  updated before changing formulas or adding tests.
- Separate detector synthesis, scalar summaries, engineering probes, and
  validation-grade diagnostics in user-facing labels.
- Preserve local Bosch-Hale coefficient support while still blocking total
  neutron-validation claims until same-scope yield/timing/spectrum/anisotropy/
  detector/uncertainty packets exist.
- Keep `BeamTracker` yield and HDF5 `max_div_B` in engineering/non-validation
  lanes unless later same-scope source packets and geometry-aware diagnostics
  are added.

## Backlog Links

See `BACKLOG.md` entries `DIA-001` through `DIA-008`.
The 2026-05-11 formulary pass added `DIA-009`.
