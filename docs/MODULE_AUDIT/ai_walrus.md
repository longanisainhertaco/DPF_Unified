# AI/WALRUS Module Audit

Status date: 2026-05-09

This note is advisory. It is not source-of-truth science and must be re-checked
against current code and reviewed `KnowledgeReference/` evidence before any
implementation work.

## Scope

Primary files reviewed:

- `src/dpf/ai/`
- WALRUS training/export scripts
- `tests/test_walrus_consolidated.py`
- `tests/test_ai_disclosure_claims.py`
- `docs/WALRUS_MHD_TRAINING_DATA_REVIEW_2026_05_09.md`

## Intended Behavior

The AI module intends to provide WALRUS-backed surrogate modeling, data export,
batch generation, dataset validation, inverse design, realtime API, and training
workflows.

The module also attempts to bridge DPF states into The Well/WALRUS-style HDF5
and model input formats.

## Source-Of-Truth Support Found Or Missing

Supported locally:

- Local KR material supports that DPF MHD modeling is hard, limited, and
  validation-sensitive.
- Local KR material supports cautious ML/surrogate use only with adequate data,
  metadata, uncertainty quantification, and applicability bounds.

Missing:

- No local `KnowledgeReference/` support was found for WALRUS, The Well, CATS,
  Polymathic, `MHD_64`, `MHD_256`, or `IsotropicModel`.
- The WALRUS training-data review document says external WALRUS/The Well/CATS
  findings are leads only until acquired, hashed, reviewed into
  `KnowledgeReference/`, and mapped to specific claims.
- Current local WALRUS/MHD training artifacts are not accepted validation data.

## Current Implementation Summary

`src/dpf/ai/well_exporter.py` exports DPF states into Well-style HDF5 and now
writes fail-closed root metadata: `validation_status="not_validation_evidence"`,
`result_label="Preview"`, and `source_status="not_source_backed"`. It writes the
configured root `grid_type`, labels sanitized non-finite field data with
`nonfinite_sanitized`, and records root sanitation counts. This preserves ML
interchange compatibility without hiding numerical failure as validation data.

`src/dpf/ai/dataset_validator.py` checks schema, field/scalar NaN/Inf, field
statistics, and energy drift. Its strict mode now adds fail-closed checks for
required energy/time datasets, monotonic time, root provenance/classification,
geometry consistency, sanitized non-finite datasets, saturation-scale values,
and all-zero magnetic fields.

The WALRUS adapter documents a single-temperature mismatch: `Te` and `Ti` share
one channel. The current guard rejects large `Te/Ti` divergence, but the 10%
threshold is not source-closed locally.

`DPFSurrogate` can enter placeholder mode after a missing checkpoint, missing
WALRUS dependency, or load failure. Prediction raises in placeholder mode, and
status/reporting now separates `placeholder_loaded`, `real_model_loaded`, and
`source_backed_model_loaded`. Source-backed WALRUS model status remains
fail-closed until checkpoint/version/license/source evidence is reviewed.

## Why It Likely Fails Or Is Unverifiable

- Current local HDF5 training data are not defensible as validation or
  publication data. The review document records non-finite current/voltage,
  all-zero B fields, missing energy conservation, missing metadata, and
  coordinate/root mismatches.
- Generic Well MHD pretraining is not DPF validation. It may support ML
  pretraining or benchmark comparison only after source/version/license/hash
  review.
- Exporter sanitation is now labeled, but any sanitized file remains
  non-validation evidence and should fail strict validation.
- Validator passes outside strict mode are still schema/interchange checks, not
  scientific validation.
- Surrogate tests use mocked or identity behavior for some paths; those are
  implementation checks, not physics validation.

## Stale Or Inaccurate Assumptions

- `src/dpf/metal/mlx_surrogate.py` still has its own placeholder path and should
  be reviewed separately before any MLX surrogate claims are made.
- `_walrus_base.py` claims 10% `Te/Ti` tolerance is negligible relative to DPF
  measurement uncertainty, but no local KR support was found for that threshold.
- `scripts/generate_walrus_data.py` now says it writes JSON exploratory
  candidate summaries and marks future manifests as `not_validation_evidence`.
- `src/dpf/ai/Troubleshooting.md` is historical routing material, not evidence.

## Trustworthy Tests Versus Suspect Tests

More trustworthy for implementation mechanics:

- Dataset validator schema/NaN/energy tests in `tests/test_walrus_consolidated.py`.
- Exporter structure and scalar/unit tests in `tests/test_walrus_consolidated.py`.
- Strict validator and exporter labeling tests for scalar non-finites,
  provenance gaps, geometry mismatch, all-zero magnetic fields, sanitized
  non-finites, and fail-closed root metadata.
- API/surrogate status tests that keep placeholder models separate from real
  and source-backed models.
- Disclosure tests in `tests/test_ai_disclosure_claims.py` as text gates.

Limited or suspect:

- Surrogate validation tests that use mocked/identity behavior.
- Real WALRUS tests now skip unless `dpf.ai.HAS_WALRUS` is true and the local
  checkpoint exists; even then they check broad sanity only.
- Any test that treats local generated data as validation data.

## Future-Agent Notes

These notes are not authority. Re-check all line numbers and current behavior
before editing.

- Treat `docs/WALRUS_MHD_TRAINING_DATA_REVIEW_2026_05_09.md` as a useful audit
  note, not source-of-truth science.
- Do not use current local WALRUS/HDF5 data for scientific validation or
  publication claims.
- Keep `source_backed_model_loaded` false until a reviewed source packet records
  checkpoint hash, version, license, source, local behavior, and accepted
  validation scope.
- Keep external WALRUS/The Well/CATS claims blocked until the exact records are
  acquired and promoted through local source review.

## Backlog Links

See `BACKLOG.md` entries `AI-001` through `AI-008`.
