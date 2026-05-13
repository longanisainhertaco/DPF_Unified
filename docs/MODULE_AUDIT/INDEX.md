# Module Audit Index

Status date: 2026-05-11

These notes are planning and review artifacts. They are not source-of-truth
science, and they must not be cited as validation evidence. Scientific claims
remain gated by reviewed local `KnowledgeReference/` records and accepted
same-scope evidence packets.

## Rules

1. Treat all non-`KnowledgeReference/` project material as suspect until traced
   to a reviewed local source and a passing gate.
2. Do not edit source modules during audit passes.
3. Add one note file per module, plus backlog entries in `BACKLOG.md`.
4. Separate how the module should work from why the current implementation is
   not yet trustworthy.
5. Preserve blocker states. A blocker is a useful result, not a failure of the
   audit.
6. Future-agent notes are advisory only. They are not gospel and must be
   re-checked against current code and `KnowledgeReference/`.

## Audit Order

| Order | Module | Note file | Status |
| --- | --- | --- | --- |
| 1 | Validation | `validation.md` | Initial deep audit complete. |
| 2 | Engine/core | `engine_core.md` | Initial scout audit integrated. |
| 3 | Metal/MLX | `metal_mlx.md` | Initial scout audit integrated. |
| 4 | Circuit/snowplow | `circuit_snowplow.md` | Initial scout audit integrated. |
| 5 | Diagnostics | `diagnostics.md` | Evidence manifest and test-lane guardrails complete; source-blocked. |
| 6 | Radiation/atomic/neutrons | `radiation_atomic_neutrons.md` | Initial scout audit integrated. |
| 7 | Collision/transport | `collision_transport.md` | 2026-05-11 formulary audit integrated. |
| 8 | IO/export | `io_export.md` | Initial scout audit integrated. |
| 9 | AI/WALRUS | `ai_walrus.md` | Guardrail pass complete; source-blocked. |
| 10 | Server/GUI/CLI | `server_gui_cli.md` | Guardrail pass complete; no open SGC code-ready items remain. |
| 11 | Supplemental physics helpers | `supplemental_physics_helpers.md` | File-level guardrail pass complete; source-blocked. |

## Shared Audit Template

Each module note should include:

- intended behavior
- source-of-truth support found or missing
- current implementation summary
- why it likely fails or is unverifiable
- stale or inaccurate assumptions
- tests that are trustworthy versus suspect
- future-agent notes, explicitly non-authoritative
- module backlog entries with blocker status

## Current Cross-Cutting Findings

- The repository contains newer fail-closed validation guardrails mixed with
  older model, calibration, and diagnostic code that still carries historical
  assumptions.
- Passing tests often prove that blocker states are preserved, not that the
  science is validated.
- Code-verification evidence is useful, but it cannot replace same-scope DPF
  experimental validation.
- Local generated data, WALRUS data, and historical review documents are not
  validation evidence unless promoted through the KR and review gates.
- The 2026-05-11 formulary pass corrected several concrete coded formula
  mismatches, but it did not globally source-verify the full simulation stack.

## Source-Truth Verification Status

No module is globally verified against the source of truth as of 2026-05-11.
Recent implementation work has closed engineering guardrails and prevented
known suspect paths from presenting as validation evidence. It has not converted
the full physics stack into source-verified science.

| Module | Current source-truth status | Why it is not fully verified |
| --- | --- | --- |
| Validation | Blocked / guarded | KR gates, target extraction, digitization, source-line semantics, certificates, readiness propagation, and calibration provenance labels exist in guarded form, but many device fields and target groups still need accepted same-scope packets. |
| Engine/core | Engineering-guarded / not globally source-verified | Runtime labels, fail-visible behavior, constants authority, nonfinite evidence, and per-preset value-authority records are guarded. Breakdown behavior, backend ownership, and physics feature application still require source/provenance closure. |
| Metal/MLX | Engineering-guarded / not source-verified | MLX density floors, probe labels, reduced phase-model metadata, coupling authority wording, MHD-derived coupling gate summaries, and field-aware NRL Braginskii perpendicular conduction are guarded. PF-1000 constants and scientific acceptance of MHD-derived circuit coupling remain source-blocked. |
| Circuit/snowplow | Engineering-guarded / source-blocked | Current-factor loading boundaries, Lee comparison `fc`/`fcr` loading, auto-coupler trust gates, density-weighted coupler source-status labels, CPU/MLX radius-convention metadata, and post-pinch resistance provenance labels improved. Akel waveform tests remain blocked. |
| Collision/transport | Formulary-partial / convention-blocked | NRL Braginskii perpendicular conductivity, MLX field-aware perpendicular conduction, and several electron-ion diagnostic branches were corrected on 2026-05-11. `nu_ee` remains convention-blocked. |
| Diagnostics | Engineering-guarded / source-blocked | BeamTracker unit handling, HDF5 `max_div_B` labels, the diagnostics evidence manifest, and diagnostics test-lane markers are guarded. Thomson, nTOF, x-ray, instability/regime/plasmoid/shear/runaway formulas and full neutron diagnostic packets remain blocked. |
| Radiation/atomic/neutrons | Formulary-partial / source-blocked | NRL Eq. 30, Eq. 33, Eq. 34, and Eq. 13 mismatches were corrected on 2026-05-11. Line cooling tables, opacity/FLD, QMF, p-B11 data, ionization/ablation constants, and neutron packets remain blocked. |
| IO/export | Product/export guarded / not scientific evidence | HDF5/Well lifecycle and metadata improved; Well HDF5 now carries fail-closed artifact classification metadata, CLI classification flags, and config/API-driven engine HDF5/Well/run-manifest/batch-Well/checkpoint/dataset-manifest propagation. Well/WALRUS/The Well source authority, certificate readiness context, strict validator coverage, and existing training artifact provenance remain blocked. |
| AI/WALRUS | Guarded / source-blocked | Strict validator/export/model-status guardrails now prevent common overclaims. Local WALRUS data and model claims remain non-validation evidence until source, license, checkpoint, formatter, and accepted validation packets exist. |
| Server/GUI/CLI | Product-label guarded / source-blocked | Backend contracts, CLI authority display, TopBar time/version labels, renderer local-first asset checks, legacy Gradio claim wording, run/global readiness-scope metadata, and PF-1000 preset source-scope labels are guarded. These labels do not promote API/UI output to scientific validation. |
| Supplemental physics helpers | Guarded / source-blocked | Ablation, two-temperature, viscosity, Nernst, sheath, anomalous-resistivity, and CIV/Paschen helpers now expose fail-closed metadata. Formula/source conventions and same-scope validation packets remain missing for predictive use. |

Use this table as a routing aid only. A module can be engineering-improved while
still scientifically blocked. Scientific status changes only through reviewed
local `KnowledgeReference/` evidence and accepted same-scope validation packets.
