# Supplemental Physics Helpers Audit

Status date: 2026-05-11

These notes cover physics-bearing helper modules that were not called out
individually in the first module-audit pass. They are planning notes for future
agents, not source-of-truth science.

## Files Checked

- `src/dpf/atomic/ablation.py`
- `src/dpf/fluid/two_temperature.py`
- `src/dpf/fluid/viscosity.py`
- `src/dpf/fluid/nernst.py`
- `src/dpf/sheath/bohm.py`
- `src/dpf/turbulence/anomalous.py`
- `src/dpf/turbulence/subgrid.py`
- `src/dpf/experimental/civ_breakdown.py`
- `src/dpf/experimental/poloidal_bfield.py`
- `src/dpf/verification/sedov_cylindrical.py`

## Intended Behavior

These modules provide optional or diagnostic physics helpers around startup,
transport, turbulence, material sources, and code-verification problems:

- electrode ablation and impurity-source scaffolding;
- operator-split electron energy and electron-ion relaxation;
- Braginskii-style ion viscosity and Nernst/Ettingshausen transport;
- Bohm/Child-Langmuir sheath utilities;
- anomalous resistivity and CIV/Paschen startup estimates;
- Auluck/GV poloidal-field research utilities;
- cylindrical Sedov-Taylor code-verification support.

## Source-Truth Support Found Or Missing

Found local support:

- The NRL formulary contains transport-coefficient rows for electron heat
  flux/conductivity and ion-viscosity coefficients, plus transport validity
  limits.
- The NRL formulary contains collision/equilibration rows that can support a
  future convention audit.
- The local corpus contains Auluck/GV material and general plasma/sheath
  background, but that does not by itself validate DPF startup or late-pinch
  predictions.

Missing or incomplete local support:

- Constant ablation efficiencies, shielding/fluence bounds, and material
  ejection regimes are not source-closed.
- Two-temperature equilibration needs a line-by-line convention audit against
  the local NRL rows before it can support validation claims.
- Ion-viscosity collision time currently uses the shared Coulomb-log helper;
  the ion-ion Coulomb-log and `tau_i` convention need a dedicated source audit.
- Nernst/Ettingshausen coefficients use Epperlein-Haines-style fits without a
  reviewed local source packet.
- Anomalous resistivity thresholds and alpha ranges are still model-form
  assumptions for DPF.
- CIV/Paschen gas coefficients, magnetization limits, and startup applicability
  require reviewed source packets.
- Sedov cylindrical `gamma=5/3` normalization still needs local Kamm/Timmes
  extraction or quadrature before quantitative accuracy claims use it.

## Current Implementation Summary

The 2026-05-11 file-level pass added fail-closed metadata helpers to:

- `ablation_model_metadata()`
- `two_temperature_model_metadata()`
- `braginskii_viscosity_model_metadata()`
- `nernst_model_metadata()`
- `sheath_model_metadata()`
- `anomalous_resistivity_model_metadata()`
- `civ_breakdown_model_metadata()`

Each reports `validation_status="not_validation_evidence"` and
`can_support_validation_claims=False`.

## Why It Is Not Yet Trustworthy

These modules mix useful mechanics with source-missing empirical or
model-form assumptions. Passing unit tests proves mechanics, finiteness, or
guardrails. It does not prove that the effect is valid for PF-1000, Akel
shot 12581, p-B11 scenarios, high-Z radiation, startup, or final-pinch neutron
physics.

## Stale Or Inaccurate Assumptions

- Any comment that cites external papers not promoted to `KnowledgeReference/`
  is an acquisition lead, not local scientific evidence.
- The Nernst and anomalous-resistivity helpers should not be described as
  source-closed Braginskii/DPF physics until the missing packets exist.
- Sedov cylindrical verification is code-verification support only; it does
  not validate DPF physics.

## Tests That Are Trustworthy Versus Suspect

Trustworthy:

- `tests/test_unreviewed_physics_metadata.py` checks that these helpers fail
  closed as non-validation evidence.
- Existing high-fidelity physics-fidelity tests verify that active ablation,
  Nernst, sheath, radiation, and other helpers remain unvalidated in readiness
  summaries.

Limited:

- Mechanics tests in `tests/test_physics.py`, `tests/test_civ_breakdown.py`,
  `tests/test_turbulence_subgrid.py`, `tests/test_poloidal_bfield.py`, and
  verification tests show code behavior, not scientific acceptance.

## Future-Agent Notes

These notes are not authority. Re-check code and source lines before editing.

- Prefer metadata/source-status guardrails before formula changes in source-
  missing helper modules.
- Do not source-close Nernst, anomalous resistivity, CIV/Paschen, or ablation
  from external citations alone. Promote and review the exact sources locally
  first.
- Keep Sedov/Taylor and Athena/AthenaK method work in the numerical
  verification lane, not the experimental validation lane.

## Backlog Links

See `BACKLOG.md` entries `PHX-001` through `PHX-006`.
