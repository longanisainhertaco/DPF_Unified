# Engine/Core Module Audit

Status date: 2026-05-11

This note is advisory. It is not source-of-truth science and must be re-checked
against current code and reviewed `KnowledgeReference/` evidence before any
implementation work.

## Scope

Primary files reviewed:

- `src/dpf/engine/core.py`
- `src/dpf/engine/backend_dispatch.py`
- `src/dpf/engine/backend_capabilities.py`
- `src/dpf/engine/circuit_coupling.py`
- `src/dpf/engine/physics_operators.py`
- `src/dpf/engine/state_management.py`
- `src/dpf/config.py`
- `src/dpf/constants.py`
- `src/dpf/presets.py`
- `app_engine.py`
- related engine, preset, backend, and PF-1000 tests

## Intended Behavior

`SimulationEngine` appears intended to be the central orchestration layer:
validate configuration, choose the backend, construct circuit/fluid/PIC/
diagnostic surfaces, initialize state, step circuit and MHD/fluid operators,
then emit diagnostics, manifests, and summary metrics.

The UI-facing `app_engine.py` is a separate adapter. It exposes a Lee/snowplow
helper path and an MHD wrapper path intended to return data in a UI-compatible
shape.

## Source-Of-Truth Support Found Or Missing

Supported locally:

- The `pf1000_akel` preset has direct local support for Akel shot 12581 bank,
  tube, operating parameters, and Lee factors in
  `KnowledgeReference/radiation-physics-and-chemistry-188-2021-109633.md`.
- Local Lee/Saw material supports gross Lee/RADPF phase limits, including
  deuterium `rmin = 0.13a`, `z = 0.7a`, reflected-shock handling, and
  deuterium radiative-phase limitations.
- Local KR material supports a limited claim for ordinary MHD in late pinch and
  yield behavior: MHD may be useful for earlier phases, while later neutron
  behavior needs kinetic or particle-aware treatment.
- Bosch-Hale D-D reactivity support exists for helper paths through local
  `KnowledgeReference/bosch-hale-1992-fusion-reactivity.md`.

Missing or not yet strong enough:

- Most broad preset comments are not line-traceable to local KR evidence.
- Backend labels such as `"production"` are engineering labels, not scientific
  readiness labels.
- The engine does not yet show a single, audited path from accepted validation
  packets through UI/API/export readiness claims.
- Breakdown configuration exists, but engine integration is not evident from
  current source search.

## Current Implementation Summary

`src/dpf/config.py` is a broad typed Pydantic configuration surface with many
physics toggles. It validates backend names, radii, geometry, grid shape, and
some ranges, but most science references in config text are narrative rather
than line-traced requirements.

`src/dpf/engine/backend_dispatch.py` maps backend names and labels some paths as
`"production"`. Treat that as implementation maturity, not validation status;
`backend_authority_labels()` now exposes that boundary explicitly.

`src/dpf/engine/backend_capabilities.py` reports skipped Athena-family physics
features and reports MLX/Metal diffusion fallback behavior.

`src/dpf/engine/circuit_coupling.py` owns circuit subcycling, snowplow/MHD
plasma-inductance blending, boundary conditions, radial initialization, ohmic
gap correction, and dynamic sheath pressure.

`src/dpf/engine/physics_operators.py` applies collision, radiation, Nernst,
Powell, viscosity, and diffusion operators with backend skips embedded in the
operator path.

`app_engine.py` wraps full-engine output for the GUI. As of 2026-05-09,
full-engine failures raise by default and Lee fallback requires an explicit
caller opt-in.

2026-05-10 implementation update:

- `src/dpf/presets.py` now exposes `preset_value_authority()` and
  `preset_authority_manifest()` so every config leaf in every named preset has
  an explicit source/value authority record.
- Preset authority records fail closed with
  `validation_status="not_validation_evidence"` and
  `can_support_validation_claims=False`. Broad narrative, empirical, derived,
  and source-blocked presets remain useful scaffolds but not validation
  evidence.

2026-05-11 formulary/local-KR update:

- Conservative-MHD energy flux in `src/dpf/fluid/mhd_solver.py` and
  `src/dpf/fluid/cylindrical_mhd.py` now uses the SI form
  `(E + p_total)v - B(v dot B)/mu_0`.
- This is a formula-correctness fix in shared fluid/MHD machinery. It does not
  source-close backend feature toggles or turn numerical runs into experimental
  validation evidence.

## Why It Likely Fails Or Is Unverifiable

- Prior blocker addressed 2026-05-09:
  `app_engine.run_mhd_simulation_core(..., n_steps=1000)` now passes the bound
  to `SimulationEngine.run(max_steps=...)`.
- Prior blocker addressed 2026-05-09:
  `app_engine.py` no longer silently converts full-engine exceptions into
  Lee-only results unless the caller sets explicit fallback opt-in metadata.
- Prior blocker addressed 2026-05-09:
  MLX/Metal operator ownership is now explicit in backend diagnostics. Requested
  Nernst, transport, and bremsstrahlung paths are reported as backend-owned;
  requested GPU `implicit`/`sts` diffusion reports explicit fallback; line
  radiation is reported as Python-operator-owned unless a backend-native source
  packet is wired.
- Prior blocker addressed 2026-05-09:
  Python-side Nernst and implicit/STS diffusion are skipped for `metal`/`mlx`
  backends so GPU-owned paths are not double-applied.
- `BreakdownConfig.enabled` defaults true, but current engine search only finds
  config/test references, not a clear engine integration path. Engine summaries
  now label this as `config_only_not_applied`.
- State sanitation repairs NaN/Inf values up to a cumulative threshold by
  default. As of 2026-05-09, the engine preserves first/last/recent nonfinite
  event evidence and supports fail-fast probe mode before repair.
- Formula-correctness fixes should not be reported as backend/readiness
  promotion. They only remove one known implementation error.

## Stale Or Inaccurate Assumptions

- Broad presets such as general PF-1000, MJOLNIR, FAETON, POSEIDON, and NX2
  contain comments marked empirical or unverified. `preset_value_authority()`
  now exposes fail-closed records for every preset value; treat those records as
  scaffolding labels, not accepted device definitions.
- `src/dpf/constants.py` now labels its constants as standards-scoped
  implementation constants, not KR validation inputs, and has direct SciPy
  authority tests.
- Backend maturity labels can be misread as validation labels unless UI/API
  surfaces keep them separate.
- Config feature toggles can imply implemented physics even when the backend
  skips or partially applies the feature.

## Trustworthy Tests Versus Suspect Tests

More trustworthy as engineering guardrails:

- `tests/test_pf1000_akel_preset.py` for source-scoped Akel preset ratchets.
- `tests/test_backend_capabilities.py` for backend warning/flag plumbing.
- `tests/test_memory_preflight.py` for memory preflight/telemetry summaries.
- `tests/test_engine_timestep_limits.py` for timestep floor regressions.
- `tests/test_mlx_pf1000_probe.py` as an opt-in, non-scientific endurance probe.
- `tests/test_preset_source_scope.py` for preset source-scope and per-value
  non-validation authority coverage.

Limited or suspect for scientific acceptance:

- Preset smoke tests prove construction and one-step current behavior, not
  source validity.
- MLX integration tests are mostly environment-skipped no-crash or shape checks.
- Some circuit handoff tests mirror production logic manually and can drift from
  implementation behavior.

## Future-Agent Notes

These notes are not authority. Re-check all line numbers and current behavior
before editing.

- Start by separating execution readiness, backend capability, and scientific
  validation readiness in any user-facing output.
- Do not make GUI fallback quieter. Engine failures should remain visible.
- Preserve the PF-1000/Akel source-scope distinction; do not use broad PF-1000
  preset behavior as Akel shot-12581 validation evidence.
- Treat `app_engine.py` as a high-risk audit surface because it bridges
  implementation details into user-facing claims.

## Backlog Links

See `BACKLOG.md` entries `ENG-001` through `ENG-009`.
