# Circuit/Snowplow Module Audit

Status date: 2026-05-11

This note is advisory. It is not source-of-truth science and must be re-checked
against current code and reviewed `KnowledgeReference/` evidence before any
implementation work.

## Scope

Primary files reviewed:

- `src/dpf/circuit/`
- `src/dpf/fluid/snowplow.py`
- `src/dpf/metal/mlx_snowplow.py`
- `src/dpf/engine/circuit_coupling.py`
- related circuit, snowplow, Akel preset, and PF-1000 tests

## Intended Behavior

The circuit/snowplow layer should provide the reduced circuit-facing dynamics
for DPF runs:

- implicit midpoint RLC integration with variable `Lp`, `dL_dt`,
  plasma resistance, and optional back EMF
- Lee-style axial rundown and radial implosion through mass/current factors
- current-factor-scaled circuit loading without corrupting geometric
  inductance coefficients
- optional MHD-derived coupling only when the field state is trustworthy
- explicit phase and provenance information for downstream diagnostics

The CPU snowplow is broader than the MLX snowplow. CPU code includes axial,
radial, reflected, pinch, and post-pinch behavior. MLX is reduced to axial,
radial inward, and pinch termination.

## Source-Of-Truth Support Found Or Missing

Supported locally:

- Local Akel KR material supports the PF-1000 shot 12581 circuit/device setup
  and Lee factors used by the `pf1000_akel` preset.
- Local Lee/Saw material supports distinct axial/radial current factors and
  axial/radial mass factors.
- Local Lee/Saw material supports shock-front and piston separation and the
  five-phase Lee-model framing.
- Local Lee/Saw material includes deuterium gross `rmin = 0.13a`, `z = 0.7a`
  guidance.

Missing or not yet strong enough:

- `CircuitCoupler` density-weighted effective radius is now explicitly labeled
  as engineering scaffolding, not local validation evidence. Same-scope source
  support is still required before scientific use.
- CPU radial circuit inductance uses shock radius `r_s`; MLX uses piston radius
  `r_p`. This difference now has declared metadata and is explicitly not
  cross-backend-equivalent validation evidence.
- CPU PF-1000-specific `r_pinch_min = 0.17a` and MLX `0.13a` are now
  scope-separated in radius-convention metadata.
- Post-pinch resistance multipliers are now labeled as empirical engineering
  continuity knobs with missing source provenance and non-validation status.

## Current Implementation Summary

`src/dpf/circuit/rlc_solver.py` implements an implicit midpoint RLC solver with
variable plasma inductance, derivative history, plasma resistance, and optional
back EMF.

`src/dpf/circuit/coupler.py` derives circuit feedback from MHD state using
density-derived sheath position, density-weighted effective radius, monotonic
`Lp`, BDF2 `dLp/dt`, and back-EMF clamping. It now exposes fail-closed
authority metadata for those outputs.

`src/dpf/fluid/snowplow.py` provides the CPU Lee-style snowplow and
circuit-facing scaling helpers. `L_coeff` remains geometric while
`_axial_circuit_inductance()`, `_radial_circuit_inductance()`, and
`_radial_circuit_dL_dt()` apply current-factor scaling for circuit loading.

`src/dpf/metal/mlx_snowplow.py` provides the reduced MLX axial/radial/pinch
path.

CPU and MLX snowplow classes now expose `radius_convention` metadata. CPU
metadata labels radial inductance as shock-front-radius (`r_s`) loading with a
PF-1000/0.14-0.17-band `r_min` scope. MLX metadata labels radial inductance as
piston-radius (`r_p`) loading with reduced deuterium gross `0.13a` termination
and no full Lee five-phase coverage.

`src/dpf/engine/circuit_coupling.py` subcycles the circuit/snowplow system and
can blend trusted MHD-derived `Lp`.

2026-05-11 formulary/local-KR update:

- `CircuitCoupler` no longer returns `back_emf = I*dLp_dt` while also sending
  `dLp_dt` to `RLCSolver`. The RLC solver already contains the inductive
  `-I*dLp/dt` term, so the coupler now clamps `dLp_dt` by equivalent voltage
  and returns `back_emf=0.0` unless a future distinct motional-EMF model is
  added.
- `src/dpf/validation/lee_model_comparison.py` now applies axial `fc` scaling
  and radial `fcr` scaling to circuit-facing inductance, `dLp_dt`, radial force,
  reflected phase, and frozen/post-crowbar radial inductance. Device
  `lee_fcr` overrides are applied per run and restored afterward.

## Why It Likely Fails Or Is Unverifiable

- Prior blocker addressed 2026-05-09:
  CPU snowplow docstrings/comments now describe `L_coeff` as geometric and
  circuit-facing `L_plasma` as current-factor scaled.
- Prior blocker addressed 2026-05-09:
  `auto` MHD coupler mode now requires a resolved MHD signal such as nonzero
  field/velocity or dynamic density, not merely a positive initial density
  array. Explicit `density_weighted` remains caller-controlled.
- Density-weighted `CircuitCoupler` behavior may be useful engineering
  scaffolding, but it is not yet source-validated as a scientific method.
  `circuit_coupler_authority()` and engine summary metadata make that status
  visible to callers.
- Prior blocker addressed 2026-05-09:
  CPU and MLX snowplow radius conventions are documented as intentionally
  different approximations and are pinned by tests as not cross-backend
  equivalent validation evidence.
- Prior blocker addressed 2026-05-09:
  post-pinch CPU resistance multipliers expose `post_pinch_resistance_authority`
  and remain `not_validation_evidence`.

## Stale Or Inaccurate Assumptions

- "Lee model" comments can hide implementation-specific approximations. Each
  approximation needs explicit source status.
- Current-factor scaling is subtle. Future edits must preserve the boundary
  between geometric inductance and circuit-facing loading; tests now assert
  that `L_coeff` does not change with `current_fraction`.
- CPU and MLX snowplow paths are not interchangeable validation evidence.
  Radius-convention metadata now makes the `r_s`/`r_p` and `0.17a`/`0.13a`
  split explicit.
- Post-pinch resistance multipliers are empirical engineering continuity
  parameters until source/provenance closure exists.
- Placeholder and xfailed waveform tests are useful blockers, not acceptance.
- Do not reintroduce `back_emf = I*dLp_dt` into a path that also passes
  `dLp_dt` to `RLCSolver`; that double-counts the inductive term.
- Preserve the boundary between geometric inductance and circuit-facing
  current-factor loading: axial uses `fc`; radial phases use `fcr`.

## Trustworthy Tests Versus Suspect Tests

More trustworthy as engineering guardrails:

- `tests/test_circuit_coupler.py` for monotonicity, clamping, and synthetic MHD
  coupling mechanics, including fail-closed authority labels.
- `tests/test_snowplow_consolidated.py` tests that preserve the current-factor
  circuit-loading behavior.
- `tests/test_pf1000_akel_preset.py` for source-scoped Akel preset parameters.
- `tests/test_snowplow_post_pinch_audit.py` for blocker preservation.
- `tests/test_snowplow_consolidated.py::TestSnowplowInstantiation` now pins
  radius-convention and post-pinch resistance authority metadata.

Limited or suspect:

- Synthetic density-coupler tests prove mechanics, not source validation.
- `tests/test_circuit_consolidated.py` includes placeholder/pass behavior.
- Xfailed Akel waveform-current tests should remain blocked until accepted
  digitization and same-scope evidence packets exist.

## Future-Agent Notes

These notes are not authority. Re-check all line numbers and current behavior
before editing.

- Preserve `L_coeff` as unscaled geometry and apply `fc`/`fcr_eff` only in
  circuit-facing helpers unless a reviewed source says otherwise.
- Do not strengthen density-weighted MHD coupling claims before source closure.
- Keep CPU and MLX snowplow result comparisons scoped to their different phase
  models.
- If updating comments, use source-status language rather than broad claims.

## Backlog Links

See `BACKLOG.md` entries `CIR-001` through `CIR-010`.
