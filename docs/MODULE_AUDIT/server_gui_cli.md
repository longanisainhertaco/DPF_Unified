# Server/GUI/CLI Module Audit

Status date: 2026-05-09

This note is advisory. It is not source-of-truth science and must be re-checked
against current code and reviewed `KnowledgeReference/` evidence before any
implementation work.

## Scope

Primary files reviewed:

- `src/dpf/server/`
- `src/dpf/cli/main.py`
- `gui/src/renderer/`
- `app.py`
- `app_mhd.py`
- related server, GUI, CLI, readiness, local-first, and project tests

## Intended Behavior

The server is intended to provide a local FastAPI/WebSocket simulation API with
health, config, presets, project lifecycle, metadata, and fail-closed readiness
payloads.

The CLI is intended to run simulations with backend overrides, serve the local
API, launch the Gradio UI, export Well data, and run legacy validation commands.

The GUI is intended to talk to the localhost API/WebSocket, show backend/status/
readiness state, and expose project and simulation controls.

## Source-Of-Truth Support Found Or Missing

Supported locally:

- Local Akel KR material supports the PF-1000 shot 12581 context and table
  values.
- Server readiness currently folds in the Akel Fig. 1 digitization blocker,
  which preserves the draft/not-accepted status.

Missing or suspect:

- Gradio/UI text that previously used "validated against 7+ published devices",
  "publication-grade", "WORKING", and "97x demonstrated" has been downgraded
  to Preview/source-gated wording. Future product copy must still route through
  validation authority before Reference use.
- CLI validation now keeps PASS/FAIR/POOR peak-current grades separate from
  source authority by showing Authority and Blockers columns plus an explicit
  source-authority note.
- Backend availability/status contracts are now aligned across config, CLI,
  server health, GUI types, and GUI controls for the known backend names.
- PF-1000 presets now expose source-scope labels so the broad mixed-scope
  `pf1000` preset, source-scoped `pf1000_akel` shot-12581 preset, and derived
  `pf1000_20kv` trend preset are not conflated.

## Current Implementation Summary

Server readiness is fail-closed and now carries explicit scope metadata. It
classifies API results as Preview/not-promoted unless accepted validation
evidence exists, includes the Akel Fig. 1 draft digitization blocker, and labels
whether that blocker applies to the current run scope or only to the global
source-closure queue.

Project lifecycle APIs enforce local project roots, config hashes, and archive/
duplicate provenance.

CLI simulation exposes more backend choices than some GUI/server surfaces.

GUI/server/CLI backend contracts now include `python`, `athena`, `athenak`,
`metal`, `mlx`, `hybrid`, and `auto` where applicable. Health/status reports
availability for `mlx` and `hybrid`; selectors and TypeScript types can carry
the same names. Availability is still not validation authority.

Preset listing now carries `source_scope`, `source_scope_status`,
`source_scope_note`, and `validation_scope`. These fields are API/UI labels; the
raw simulation configs returned by `get_preset()` do not include them.

## Why It Likely Fails Or Is Unverifiable

- Backend contract mismatch for `mlx`/`hybrid` has been closed at the
  health/type/selector level. Scientific backend authority remains separate.
- GUI time display previously treated metadata/state seconds as nanoseconds.
  `TopBar` now formats seconds correctly as ns/us/ms/s by magnitude.
- Legacy Gradio backend copy now uses Preview/source-gated wording; CLI
  validation displays source authority beside peak-current grades. Remaining
  product risk is mainly readiness scope wording and PF-1000 preset/source-scope
  labeling.
- Local-first policy now scans renderer files for non-local HTTP assets. The
  renderer HTML no longer loads external Google font assets and its CSP allows
  only self/local API/WebSocket/font data sources.
- API readiness now states whether the Akel digitization blocker applies to the
  declared run validation scope or is only a global source-queue blocker.
- PF-1000 preset confusion has been reduced at the listing/UI level. Broad
  PF-1000 and 20 kV trend presets remain non-validation evidence.

## Stale Or Inaccurate Assumptions

- GUI version text now comes from `gui/package.json` via Vite instead of the
  old hardcoded `v1.0.0` label.
- Gradio PF-1000 defaults mix values from different scopes unless explicitly
  labeled. Current preset listings now carry source-scope labels for the PF-1000
  family, but the underlying defaults still need source-by-field closure before
  any validation claim.
- "WORKING", "validated", "publication-grade", and "97x demonstrated" are now
  blocked by claim-hygiene tests for the legacy Gradio UI.

## Trustworthy Tests Versus Suspect Tests

More trustworthy for narrow behavior:

- `tests/test_cli_backend_options.py` for backend contract names and CLI
  source-authority display.
- `tests/test_gradio_claims.py` for legacy Gradio claim-hygiene copy gates.
- `tests/test_server_readiness.py` for fail-closed readiness and explicit
  run/global source-blocker scope metadata.
- `tests/test_server_projects.py` for root containment and config hash behavior.
- `tests/test_project_lifecycle.py` for duplicate/archive preservation.
- `tests/test_web_ui_consolidated.py` for renderer/server backend status
  contract coverage, including `mlx` and `hybrid`.
- `tests/test_local_first_security.py` now covers renderer external asset scans
  and the current renderer HTML local-first posture.
- `tests/test_preset_source_scope.py` for PF-1000 preset source-scope labels
  and REST preset metadata.

Incomplete or suspect:

- `npm run typecheck` in `gui/` covers the updated TopBar formatter interface,
  but there is still no dedicated renderer unit test runner for display values.
- Preset source-scope labels are product/API guardrails; they do not prove each
  preset value has a line-level KR source.

## Future-Agent Notes

These notes are not authority. Re-check all line numbers and current behavior
before editing.

- Treat all UI science language as suspect until it routes through readiness and
  source-authority helpers.
- Do not promote Akel Fig. 1 until independent review metadata passes.
- Align backend contracts before adding more UI controls.
- Keep product status, engineering readiness, global source blockers, run-scope
  blockers, and scientific validation as separate labels.

## Backlog Links

See `BACKLOG.md` entries `SGC-001` through `SGC-008`.
