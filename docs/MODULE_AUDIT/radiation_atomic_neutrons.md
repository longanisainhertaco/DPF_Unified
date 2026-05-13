# Radiation/Atomic/Neutrons Module Audit

Status date: 2026-05-11

This note is advisory. It is not source-of-truth science and must be re-checked
against current code and reviewed `KnowledgeReference/` evidence before any
implementation work.

## Scope

Primary files reviewed:

- `src/dpf/radiation/`
- `src/dpf/atomic/`
- neutron and yield diagnostics in `src/dpf/diagnostics/`
- MLX line-radiation mirror surfaces where relevant
- tests for bremsstrahlung, improved radiation, QMF, ionization, neutron yield,
  p-B11, PIC yield, and radiation metadata

## Intended Behavior

This module family appears intended to provide:

- bremsstrahlung, recombination, cyclotron, line-radiation, and FLD transport
  estimates
- ionization and ablation source terms
- thermonuclear DD, beam-target DD, p-B11 estimate, PIC kinetic yield, nTOF
  spectra, and aggregate yield metadata

## Source-Of-Truth Support Found Or Missing

Supported locally:

- NRL bremsstrahlung support exists in `KnowledgeReference/plasma-formulary.md`
  and is directly tested against implementation coefficients.
- NRL recombination and cyclotron formulas are present locally and partially map
  to `src/dpf/radiation/improved_radiation.py`.
- Saha, coronal, and ionization concepts exist locally in the plasma formulary,
  though the implementation goes beyond the simple local support.
- Bosch-Hale DD cross-section/reactivity tables and validity ranges exist in
  `KnowledgeReference/bosch-hale-1992-fusion-reactivity.md`.
- Lee/Saw beam-target phenomenology and calibration exist locally for
  beam-target estimates.

Missing or blocked:

- CHIANTI, ADAS, and Post cooling coefficients are not currently source-closed
  in local KR; the line-radiation module marks the coefficients as empirical and
  unknown-provenance.
- QMF bremsstrahlung suppression is explicitly unverified in-module.
- p-B11 reactivity/yield tables are explicitly outside the verified local corpus.
- High-Z predictive radiation needs opacity, EOS, charge-state kinetics, and
  source tables before validation claims are allowed.

## Current Implementation Summary

Bremsstrahlung is the strongest scoped radiation component. It has an
NRL-derived coefficient and tests that compare SI and eV/K forms against the
local source.

Line radiation is an empirical piecewise cooling system for H, Ne, Ar, Cu, W,
and interpolated generic-Z behavior. Metadata already marks it as not
high-Z-predictive.

2026-05-09 implementation update:

- Line-radiation metadata now exposes `source_status`,
  `validation_status="not_validation_evidence"`, and an engineering claim
  scope. CPU and MLX line-radiation surfaces now agree that the coefficients
  are unknown-provenance empirical fits, not direct CHIANTI/ADAS/Post source
  tables.
- QMF suppression now exposes metadata that marks the suppression formula as
  `free_free_suppression_source_missing` and
  `unverified_not_design_evidence`.

2026-05-10 implementation update:

- `QMFDiag` now carries the same fail-closed authority status as
  `qmf_model_metadata()`: heuristic diagnostic role, missing free-free source
  status, `validation_status="not_validation_evidence"`, and no validation or
  design-claim support. This closes the quarantine path for `RAD-005`; a future
  source derivation packet is still required before any promotion.

2026-05-11 formulary update:

- NRL Eq. 30 bremsstrahlung handling in `src/dpf/fluid/ionization.py` now uses
  the `W/cm^3` formulary coefficient, the correct SI conversion, and the
  quasi-neutral single-effective-charge reduction `sum Z^2 N(Z) = Z_eff * Ne`.
- NRL Eq. 33 recombination radiation now uses
  `C_REC = 1.69e-38 * sqrt(13.6)` in `line_radiation.py` and
  `improved_radiation.py`.
- NRL Eq. 34 cyclotron radiation now treats `B_mag` as a magnitude so negative
  input signs do not zero a `B^2` power term.
- NRL Eq. 13 radiative recombination is now implemented with the bracket term
  in `src/dpf/atomic/ionization.py`.
- `src/dpf/radiation/transport.py` now exposes
  `radiation_transport_model_metadata()` and labels the FLD/Rosseland/Kramers
  path as source-packet-missing and non-validation evidence.
- `src/dpf/diagnostics/pb11_yield.py::pb11_model_metadata()` now explicitly
  separates local-NRL-supported reaction/Q-value bookkeeping from missing
  p-B11 reactivity-table source support.

Improved radiation combines bremsstrahlung, recombination, and cyclotron
components. The cooling update should be reviewed carefully because the wording
suggests implicit behavior while the implementation computes old-state power and
then subtracts energy with a floor.

Ionization has useful structure, but rates/constants are mixed simplified
models and need field-by-field source closure.

Yield tracking is metadata-aware but approximate; thermonuclear paths use
temperature/density reductions and beam-target paths can use fallback pinch
lengths.

## Why It Likely Fails Or Is Unverifiable

- High-Z radiation is not source-closed. Line coefficients, opacities, EOS, and
  charge-state kinetics are not validated locally.
- Total neutron claims are not validation-grade without yield, timing, spectrum,
  anisotropy, detector response, and uncertainty closure.
- The DD thermonuclear branch returns combined DD reactivity, then neutron yield
  paths need careful branch handling so D(d,n) neutron production is not
  confused with combined DD fusion reactivity.
- App-layer neutron estimates use crude final-state volume, confinement, and
  pinch-length assumptions.
- QMF and p-B11 modules carry explicit non-predictive or unverified status and
  should remain quarantined from design claims.

## Stale Or Inaccurate Assumptions

- `line_radiation.py` and `src/dpf/metal/mlx_line_radiation.py` now reconcile
  the main provenance language and metadata, but the underlying coefficients
  remain source-blocked until local tables or accepted source packets exist.
- README-level statements about radiation transport may not match the existence
  of an FLD scaffold. Scaffold existence is implementation evidence, not
  validation evidence.
- Any future claim that opacity/FLD/Kramers/Rosseland behavior is
  "formulary-backed" must point to a separate local source packet; the
  2026-05-11 formulary pass did not promote that path.
- p-B11 reaction/Q-value support does not promote p-B11 reactivity, yield, or
  DPF feasibility claims.

## Trustworthy Tests Versus Suspect Tests

More trustworthy:

- `tests/test_bremsstrahlung_nrl.py` checks NRL coefficient conversion and K/eV
  consistency against local KR.
- `tests/test_radiation_model_metadata.py` checks that empirical/high-Z-limit
  metadata is exposed.
- `tests/test_qmf_suppression.py` now checks that QMF diagnostic outputs are
  quarantined from validation/design claims.

Limited or suspect:

- `tests/test_qmf_suppression.py` validates behavior of a module that explicitly
  says its suppression formula is unverified.
- `tests/test_improved_radiation.py` mostly checks monotonicity, positivity,
  bounds, and component sums.
- p-B11, line-radiation, PIC-yield, nTOF, and aggregate yield tests should be
  treated as mechanics/regression tests unless separately source-closed.

## Future-Agent Notes

These notes are not authority. Re-check all line numbers and current behavior
before editing.

- Keep QMF, p-B11, high-Z line radiation, app-layer neutron yield, and synthetic
  nTOF as estimate/scaffold/unverified until source packets and validation tests
  exist. QMF is currently quarantined as diagnostic-only, not source-closed.
- Use metadata/status labels as guardrails, not acceptance evidence.
- Do not convert NRL/Bosch-Hale coefficient support into broad DPF validation
  claims.
- Keep CPU and MLX radiation metadata in parity whenever either backend changes
  line-radiation source/provenance wording.

## Backlog Links

See `BACKLOG.md` entries `RAD-001` through `RAD-008`.
The 2026-05-11 formulary pass added `RAD-009` through `RAD-011`.
