# Project Folder Restructure Audit - 2026-05-09

## Scope

This pass inspected the project folder for material that is not yet verifiable,
material that is stale or generated, and folder-structure changes that can be
made without touching the source-of-truth corpus.

Source-of-truth guardrail:

- `KnowledgeReference/` was treated as read-only.
- No files were moved into, out of, or within `KnowledgeReference/`.
- Findings documents were inspected as active plan/status context, but this pass
  did not use external material to promote any scientific claim.

## Current Structure Classes

| Class | Paths | Status |
| --- | --- | --- |
| Source of truth | `KnowledgeReference/`, `KnowledgeReference/digitization/`, `KnowledgeReference/figures/`, `KnowledgeReference/reference-images/` | Read-only for this pass. |
| Active source/test surface | `src/`, `tests/`, `scripts/`, `gui/`, top-level runnable configs | Left in place. |
| Active plan/status | `CortexFindings.md`, `CodexFindings.md`, `docs/SOURCE_ACQUISITION_NEEDED.md`, SRS/RTM docs | Left in place. |
| Source-intake and provenance staging | `archive_reference_OLD/`, `downloaded_books_papers/`, local source-search docs | Left in place because they support acquisition/provenance work. |
| Active generated validation evidence | `results/mhd_*`, tracked `training/vv_campaign/*` | Left in place; code-verification status only unless separately source-validated. |
| Unverified local data/model assets | `training_data/`, `docs/walrus_training_pf1000.h5`, `models/walrus-pretrained/`, `walrus/` | Left in place because scripts/tests reference these paths, but they are not validation-grade. |
| Stale/generated archive | `output/archive/2026-05-09/` | New archive location for generated clutter moved in this pass. |

## Cannot Be Verified Yet

These items should remain blocked, exploratory, or engineering-only until they
have accepted provenance, source mapping, and validation evidence.

| Item | Current location | Verification gap | Allowed use before closure |
| --- | --- | --- | --- |
| Local WALRUS/DPF HDF5 training sets | `training_data/dpf_v2/`, `training_data/dpf_batch_50/` | Missing full manifests; audit found missing energy-conservation fields, non-finite circuit scalars, suspicious saturated field values, geometry metadata mismatches, and sampled all-zero magnetic fields. | Schema tests, negative tests, pipeline development, exploratory ML. |
| PF-1000 WALRUS HDF5 file | `docs/walrus_training_pf1000.h5` | Not valid Well/MHD training data; contains current-waveform style data only and lacks required validation metadata. | Exploratory loader tests only. |
| WALRUS checkpoint | `models/walrus-pretrained/walrus.pt` | Exact model hash, source version, license record, model card, and inference behavior are not promoted into a project evidence packet. | Local smoke tests only. |
| Local `walrus/` checkout | `walrus/` | Ignored external checkout; useful as implementation reference but not DPF scientific evidence. | Engineering reference, subject to license/provenance review. |
| Public The Well/WALRUS/CATS leads | Acquisition queue only | Not local source-of-truth evidence and not same-scope DPF validation. | Acquisition and benchmark planning only. |
| Tier-3 numerical packet | `results/mhd_tier3_numerical_packet.json` and related `results/mhd_*` evidence | Code verification only; does not close PF-1000 spatial, neutron, physics-fidelity, or predictive-readiness validation. | Numerical verification reporting only. |
| Akel S1/S2 waveform closure | Active findings/source queue | Still blocked by review/source acceptance status where same-scope digitization is not accepted. | Blocker reporting only. |
| Tier 2 phase, Tier 4 spatial, Tier 5 neutron validation | Active findings/source queue | Missing same-scope KR-backed targets, uncertainty values, and accepted review packets. | Blocker reporting and extraction planning only. |
| Historical review/planning docs | `docs/PHD_DEBATE_*`, `docs/CYCLE*`, `docs/SPRINT*`, `docs/PHASE*`, `docs/PR_B_*`, `docs/UAT_*` | Many are snapshots against older code or older plans; not automatically current against the present implementation. | Historical context only until re-audited. |

## Archived In This Pass

The following generated or stale folders/files were moved under
`output/archive/2026-05-09/`. This was done instead of deletion so the move is
reversible.

| New archive folder | Contents moved |
| --- | --- |
| `generated-output/` | Old `output/shinka_*` runs, Shinka logs, and `output/sweep`. |
| `root-generated/` | Top-level ignored run scratch such as `.last-*`, `.exp04-results.json`, `:memory:`, `correct.json`, `diagnostics.h5`, `dpf_output.h5`, `dpf_cylindrical_output.h5`, `metrics.json`, and temporary Office lock files. |
| `ui-screenshots/` | Old March 2026 browser screenshots from the former `screenshots/` root folder. |
| `generated-sites/` | Generated static/build outputs: `site/`, top-level `dist/`, and `gui/dist/`. |
| `caches/` | Python/test/temp caches: `.pytest_cache`, `.ruff_cache`, root/script/test `__pycache__`, `tmp/`, and related cache folders. |
| `frontend-cache/` | Reflex generated frontend cache: `frontendv2/.web`, `frontendv2/.states`, and `frontendv2/__pycache__`. |
| `legacy-experiments/` | Ignored `spark33/` experiment folder. |
| `training-runs/` | Ignored training run logs and generated sweep folders, excluding tracked `training/vv_campaign/*` evidence files. |

## Stale But Not Moved Yet

These are tracked or referenced docs. They should be archived in a follow-up
that updates indexes and references at the same time.

| Candidate archive group | Paths | Reason not moved in this pass |
| --- | --- | --- |
| Debate/review snapshots | `docs/PHD_DEBATE_*_VERDICT.md`, `docs/PHASE_R_DEBATE_VERDICT.md` | `docs/RESEARCH_INDEX.md` and other docs reference the root paths. |
| Prototype cycle docs | `docs/CYCLE1_*`, `docs/CYCLE2_DEEP_REVIEW.md`, `docs/CYCLE3_FINAL_PROTOTYPES.md` | Some scripts/docs still cite them as historical implementation context. |
| Sprint/phase plans | `docs/SPRINT*`, `docs/PHASE_B_*`, `docs/PHASE1_ASSESSMENT_REPORT.md` | Several docs cross-reference these risk and implementation notes. |
| PR/UAT snapshots | `docs/PR_B_*`, `docs/UAT_A_ENHANCED.md`, `docs/UAT_B_ENHANCED.md` | Useful historical acceptance context; move only with an index redirect. |
| Old calibration/research logs | `docs/CALIBRATION_RUN_2026_03_25.md`, `docs/calibration_run_2026_03_25_raw.log`, `docs/PHYSICS_AUDIT_2026_03_31.md`, `docs/LOCAL_PDF_SOURCE_AUDIT_2026_05_06.md` | Keep until the active source-acquisition docs link to an archive location. |
| Old reference staging | `archive_reference_OLD/` | Name is stale, but it still contains provenance-relevant local source material such as the LeVeque promotion source path. |

## Folder Rules Going Forward

1. Keep `KnowledgeReference/` immutable except during explicit source-ingestion
   work.
2. Keep active requirement/status artifacts in predictable root or `docs/`
   locations until references are updated.
3. Put generated runs, screenshots, build products, caches, and temp files under
   `output/` or another ignored generated-artifact path.
4. Keep unverified local data in place only when code expects those paths, but
   mark it as `Exploratory` or `Unverified`.
5. Do not use public WALRUS/The Well material, local HDF5 data, or historical
   review docs as scientific support unless they are promoted through the local
   evidence process.
6. When tracked historical docs are archived, move them as a batch with:
   updated `docs/RESEARCH_INDEX.md`, updated SRS references, and a redirect
   index under `docs/archive/`.

## Recommended Follow-Up Structure

This pass created the generated-artifact archive. The next doc-structure pass
should be tracked and reference-aware:

```text
docs/
  active/
    srs/
    validation/
    source-acquisition/
  archive/
    historical-reviews/
    prototype-cycles/
    sprint-phase-plans/
    pr-uat-snapshots/
  indexes/
    research-index.md
    archive-index.md
```

Do not perform that tracked-doc move until references are updated in the same
change.
