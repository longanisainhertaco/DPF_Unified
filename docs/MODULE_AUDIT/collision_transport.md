# Collision/Transport Module Audit

Status date: 2026-05-11

This note is advisory. It is not source-of-truth science and must be re-checked
against current code and reviewed `KnowledgeReference/` evidence before any
implementation work.

## Scope

Primary files reviewed:

- `src/dpf/collision/spitzer.py`
- `src/dpf/fluid/anisotropic_conduction.py`
- Braginskii transport surfaces in `src/dpf/metal/metal_transport.py`
- related formulary tests in `tests/test_formulary_transport_audit.py` and
  existing physics/consolidated tests

## Intended Behavior

The collision/transport layer should provide NRL/Braginskii-scoped collision,
resistivity, temperature-relaxation, and thermal-conductivity helpers for code
verification and engineering simulation support. It should not imply that a
full DPF run is experimentally validated.

## Source-Of-Truth Support Found Or Missing

Supported locally:

- The local NRL formulary contains Coulomb-log branch formulas, transverse
  Spitzer resistivity, and Braginskii electron heat-flux coefficients,
  including the high-field perpendicular conductivity coefficient `4.7`.

Missing or blocked:

- `nu_ee` needs an explicit public convention before edit. The local formulary
  includes multiple electron-electron collision/relaxation rows, while current
  tests expect `nu_ee = sqrt(2) * nu_ei` for `Z=1`.
- MLX directional conduction now computes the NRL high-field perpendicular
  coefficient when field components are supplied. It still falls back to
  isotropic conduction when field components are absent.

## Current Implementation Summary

2026-05-11 formulary update:

- `braginskii_kappa()` now preserves the unmagnetized limit and matches the NRL
  high-field perpendicular coefficient `4.7`.
- `braginskii_kappa_perp()` and Metal transport direct high-field helpers now
  use `4.7` instead of `4.66`.
- `src/dpf/collision/Troubleshooting.md` now marks its older blanket
  "all formulas correct" verdict as superseded.
- `src/dpf/metal/mlx_transport.py::apply_thermal_conduction()` now replaces the
  fixed `kappa_perp_ratio = 1e-6` path with an NRL electron-ion Coulomb-log and
  Braginskii `4.7` high-field perpendicular conductivity calculation when
  field components are available.

## Why It Likely Fails Or Is Unverifiable

- Collision-frequency naming can be ambiguous unless the API states whether it
  means a collision frequency, momentum relaxation rate, energy exchange rate,
  or one of the NRL shorthand rows.
- Approximate backend-specific conduction paths can be stable engineering
  choices without being source-closed Braginskii implementations. The remaining
  fallback path without field components is still a mechanics fallback.

## Trustworthy Tests Versus Suspect Tests

More trustworthy:

- `tests/test_formulary_transport_audit.py` checks the Braginskii high-field
  limit, unmagnetized limit, diagnostic resistivity path, and electron-ion
  Coulomb-log branch behavior.

Limited or suspect:

- Tests that assert only positivity or suppression of `kappa_perp` prove broad
  behavior, not exact formulary coefficients.

## Future-Agent Notes

These notes are not authority. Re-check all line numbers and current behavior
before editing.

- Do not change `nu_ee` until the intended convention is named in code and
  tests.
- Keep direct NRL coefficient tests near any future Braginskii transport edit.
- Keep direct NRL coefficient coverage for MLX conduction; do not reintroduce a
  fixed cross-field ratio in field-aware conduction.

## Backlog Links

See `BACKLOG.md` entries `COL-001`, `COL-002`, and `MLX-010`.
