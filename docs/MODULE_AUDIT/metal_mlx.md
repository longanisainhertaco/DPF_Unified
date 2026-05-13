# Metal/MLX Module Audit

Status date: 2026-05-11

This note is advisory. It is not source-of-truth science and must be re-checked
against current code and reviewed `KnowledgeReference/` evidence before any
implementation work.

## Scope

Primary files reviewed:

- `src/dpf/metal/`
- `scripts/run_mlx_pf1000_probe.py`
- `tests/test_mlx_*.py`
- related PF-1000 and cross-backend tests

## Intended Behavior

The MLX path appears intended to provide a pure-MLX PF/DPF engine with:

- an implicit midpoint RLC circuit
- a reduced snowplow phase model
- optional MHD field solve
- trust-gated circuit coupling
- current, inductance, phase, and coupling provenance fields

The reduced MLX snowplow model handles axial rundown, radial inward shock, and
pinch termination. It does not model the full reflected shock, radiative pinch,
or expanded column behavior.

The MLX MHD solver advertises a broad numerical surface: cylindrical
finite-volume MHD, HLL/HLLD, PLM/WENO, SSP RK, dual energy, electrode boundary
handling, diffusion, conduction, radiation, Hall/PIC hooks, and stabilization
logic.

## Source-Of-Truth Support Found Or Missing

Supported locally:

- Local Lee/Saw KR material supports reduced Lee-style gross snowplow constants
  and phase simplifications, including `rmin = 0.13a`, `z = 0.7a`, axial
  snowplow/radial slug framing, and reflected-shock limitations.
- Local KR material supports limiting ordinary MHD claims for late pinch and
  particle-yield behavior. MHD can support earlier dynamics, but late neutron
  behavior needs kinetic or particle-aware evidence.
- `scripts/run_mlx_pf1000_probe.py` is correctly marked as non-scientific
  endurance/regression policy rather than acceptance evidence.

Missing or not yet traceable:

- PF-1000 Akel 16 kV shot constants used in tests remain suspect unless traced
  to accepted KR evidence.
- Density-weighted MHD inductance coupling is not yet locally validated as a
  field-derived method.
- Sun 2025, Auluck-path, and Poynting/flux coupling references still need local
  source review before they can support validation claims; the comments and
  runtime metadata now fail closed instead of presenting those paths as
  source-backed authority.

## Current Implementation Summary

`src/dpf/metal/mlx_engine.py` seeds an MHD solver around the snowplow state,
runs the snowplow every step for phase detection, advances MHD using a
CFL-limited substep, and optionally blends MHD-derived `Lp`, `dLdt`, and
resistance into the circuit after finite, positive, and engineering eligibility
checks. The blend gate now records phase eligibility, finite/positive/comparable
`Lp`, finite `dLdt`, and finite/nonnegative resistance checks, while preserving
`can_support_scientific_claims=False`.

`src/dpf/metal/mlx_circuit.py` is an implicit midpoint RLC integrator with
variable plasma inductance/resistance and optional back EMF.

`src/dpf/metal/mlx_solver.py` contains significant stabilization machinery:
post-step floors, NaN sanitation, velocity clipping, magnetic cleaning,
diffusion/conduction/radiation, and coupling update hooks.

`src/dpf/metal/mlx_timestepper.py` includes a stage helper and `_apply_floors()`
path intended to avoid fake mass injection. As of 2026-05-09, the direct helper
and zero-`dt` RK2/RK3 paths are covered by no-density-injection tests.

2026-05-11 formulary/local-KR update:

- Metal/MLX Braginskii high-field perpendicular conductivity helpers now use
  the NRL `4.7` coefficient where the full inputs are available.
- Cylindrical MHD source terms in `metal_riemann.py`, `mlx_timestepper.py`,
  `mlx_riemann.py`, and `mlx_sources.py` were corrected to the r-weighted
  conservative form with `p_total`, inward toroidal hoop stress, correct theta
  momentum sign, and no second density multiply for conserved source arrays.
- `mlx_transport.py::apply_thermal_conduction()` now computes the NRL
  electron-ion Coulomb log and Braginskii high-field perpendicular
  conductivity when field components are supplied. The no-field fallback
  remains mechanics-only.

## Why It Likely Fails Or Is Unverifiable

- Prior blocker addressed 2026-05-09:
  MLX coupling comments no longer call voltage-flux or Poynting-voltage paths
  "CORRECT" or "first-principles" authority. `coupling_method_authority()`
  classifies density-weighted `Lp`, voltage-flux, and Poynting-voltage methods
  as non-validation evidence, and `run_mlx_discharge()` returns
  `mhd_coupling_authority` so the engine surface carries the same guardrail.
- Prior blocker addressed 2026-05-09:
  `compute_upf_voltage_flux()` no longer contains the dead radial-coordinate
  expression.
- Prior blocker addressed 2026-05-09:
  `_apply_floors()` no longer applies a `B^2/va_max^2` density floor, and full
  zero-`dt` RK2/RK3 tests now guard against floor-driven fake mass injection.
- Prior blocker labeled 2026-05-09:
  the circuit step in `mlx_engine.py` still passes `back_emf=0.0`, and
  `run_mlx_discharge()` now exposes `back_emf_authority` plus `back_emf_V` so
  this cannot be mistaken for separate motional back-EMF coupling.
- Prior blocker labeled 2026-05-09:
  `run_mlx_discharge()` now exposes `phase_model_authority`, marking the pure
  MLX snowplow path as reduced axial/radial/pinch coverage, not full Lee
  five-phase coverage.
- Prior blocker addressed 2026-05-09:
  MHD-derived coupling is no longer gated only by finite/positive `Lp`.
  `evaluate_mhd_coupling_gate()` separates engineering blend eligibility from
  scientific authority, and `run_mlx_discharge()` returns an
  `mhd_coupling_gate` summary that remains `not_validation_evidence`.
- PF-1000 current/dip/phase tests are intentionally blocked or xfailed; they
  must stay non-authoritative until accepted source packets exist.
- Field-aware MLX conduction now has direct NRL coefficient coverage. Do not
  cite no-field fallback conduction as a full Braginskii implementation.

## Stale Or Inaccurate Assumptions

- Any future "correct", "first-principles", or "trust-gated" coupling language
  is suspect unless backed by accepted local KR and a validation gate. A
  focused claim-guard test now scans the MLX coupling/engine surfaces for those
  overclaim markers.
- MLX finite-state smoke tests and endurance probes can support implementation
  stability, not DPF scientific acceptance. A non-slow policy test now checks
  that the standalone PF-1000 MLX probe remains `non_scientific` and
  source-blocked.
- Hardcoded Akel/PF-1000 constants in tests are not automatically accepted
  evidence. They need source lineage and review status.
- Broad MHD solver feature claims may overstate what is active in a given MLX
  run mode.
- Operator ownership is now reported explicitly for GPU backends. This prevents
  requested Nernst, diffusion, radiation, and transport flags from silently
  implying both backend-native and Python-side application.

## Trustworthy Tests Versus Suspect Tests

More trustworthy as engineering checks:

- `tests/test_mlx_snowplow.py` for scalar snowplow behavior.
- `tests/test_mlx_circuit_coupling.py` for synthetic coupling mechanics.
- `tests/test_mlx_cross_backend.py` for qualitative parity behavior.
- `tests/test_mlx_pf1000_probe.py` for opt-in finite-state endurance policy.
- `tests/test_mlx_claim_guardrails.py` for fail-closed MLX coupling authority
  metadata and source-text claim hygiene.

Limited or suspect for scientific acceptance:

- `tests/test_mlx_pf1000.py` current, dip, and phase acceptance tests are
  intentionally xfailed or blocked by source closure.
- `tests/test_mlx_acceptance.py` has explicitly xfailed diffusion convergence
  behavior needing grid investigation.
- Synthetic Bennett/coupling tests prove plumbing and numerical behavior, not
  device validation.

## Future-Agent Notes

These notes are not authority. Re-check all line numbers and current behavior
before editing.

- Do not promote MLX endurance or cross-backend parity evidence to scientific
  validation.
- Keep PF-1000 S1/S2 source closure blocked until accepted KR-backed packets
  exist.
- Audit operator ownership before changing Nernst, Hall, diffusion, radiation,
  or transport paths. The current guardrail reports backend-owned, fallback, or
  Python-owned behavior; future changes should update those diagnostics and
  tests in lockstep.
- If comments cite missing papers, either add reviewed local source support or
  downgrade the claim.

## Backlog Links

See `BACKLOG.md` entries `MLX-001` through `MLX-010`.
