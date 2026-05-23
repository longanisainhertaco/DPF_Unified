# SS12 P1 Phase 5-D Figure Crop Artifacts — Evaluate / Learn / Continue

Date: 2026-05-22 UTC
Phase: Crop region selection and crop artifact generation

## Evaluate

Implemented Phase 5-D crop-region planning and crop artifact generation.

Created/updated:

- `docs/SS12_P1_PHASE5D_CROP_REGION_PLAN_2026_05_22.json`
- `scripts/build_ss12_phase5d_crop_artifacts.py`
- `docs/SS12_P1_PHASE5D_CROP_ARTIFACT_MANIFEST_2026_05_22.json`
- `tests/test_ss12_phase5d_crop_artifacts.py`

Generated crop PNG artifacts under:

- `artifacts/ss12_phase5d/crops/`

Crop rows map one-to-one to Phase 5-C page renders:

1. `pf1000_recent_progress_fig6_current_waveform`
2. `pf1000_scholz_fig4_density_distribution`
3. `pf1000_krauz_fig8_magnetic_probe_current`
4. `pf1000_scholz_fig9_neutron_timing`

The crop regions are visual-estimate regions from the rendered PDF pages. They are image artifacts only. They are not digitized and do not carry accepted observable values.

## Guardrails implemented

Phase 5-D builder fails closed when:

- output manifest path is outside `docs/`;
- crop output directory is outside `artifacts/`;
- crop plan top-level acceptance flags are not false;
- crop row acceptance flags are not false;
- duplicate page-render artifact IDs are present;
- duplicate page-render `figure_source_id` values are present;
- duplicate crop IDs are present;
- duplicate crop `figure_source_id` values are present;
- crop plan IDs do not match the Phase 5-C page render manifest one-to-one;
- page image path resolves outside `artifacts/`;
- page image is missing;
- page image SHA-256 does not match the Phase 5-C manifest;
- crop bbox is malformed;
- crop bbox includes boolean coordinates;
- crop bbox is outside page bounds.

Every crop artifact records:

- source page-render ID;
- page image path and SHA-256;
- crop pixel bounding box;
- crop basis;
- crop image path;
- crop image SHA-256;
- digitization hash as null;
- axis calibration status as missing;
- uncertainty budget status as missing;
- review certificate status as missing;
- all acceptance/promotion flags as false.

## Verification

Focused integrated verification:

```text
.venv312/bin/python -m pytest \
  tests/test_ss12_phase5d_crop_artifacts.py \
  tests/test_ss12_phase5c_page_render_crop_manifest.py \
  tests/test_ss12_phase5b_figure_asset_inventory.py \
  tests/test_first_principles_phase5b_figure_asset_packet_plan.py \
  tests/test_ss12_phase5_figure_source_manifest.py \
  tests/test_first_principles_phase5_figure_packet_builder.py \
  tests/test_first_principles_acceptance_shield_phase4d.py \
  tests/test_first_principles_figure_candidate_phase4c.py \
  tests/test_first_principles_circuit_power_port_phase4b.py \
  tests/test_first_principles_numerical_fidelity_phase4a.py \
  tests/test_ss12_phase2_source_packet_matrix.py \
  tests/test_ss12_phase3_transfer_candidate_matrix.py -q

77 passed in 3.35s
```

Lint:

```text
ruff check scripts/build_ss12_phase5d_crop_artifacts.py tests/test_ss12_phase5d_crop_artifacts.py

All checks passed!
```

Static scan:

```text
static_scan_findings 0
```

Acceptance-promotion JSON scan:

```text
0 matches
```

Independent final review:

```json
{
  "passed": true,
  "security_concerns": [],
  "logic_errors": [],
  "suggestions": [
    "Optional: split duplicate crop-region failures into distinct rule strings for duplicate crop ids versus duplicate crop figure_source_id values to improve diagnostics.",
    "Optional: add explicit schema-style missing-field checks for page/crop rows to avoid raw KeyError on malformed manifests."
  ]
}
```

## Learn

- Crop artifacts are a useful bridge between source PDFs and digitization, but they remain pre-calibration artifacts.
- The artifact chain now has hash continuity: PDF hash -> page PNG hash -> crop PNG hash.
- Manifest-generated artifacts need duplicate-ID checks before dict construction; otherwise duplicates can be silently overwritten.
- Python `bool` is a subclass of `int`; bbox validation must use `type(v) is int`, not `isinstance(v, int)`.

## Continue

Proceed to Phase 5-E: digitization packet scaffolding and calibration schema.

Next requirements:

1. define axis-calibration schema per crop;
2. require at least two calibration points per axis before any digitized series is accepted;
3. compute digitization hash from crop image hash + calibration + extracted points;
4. keep digitized values staged only;
5. require uncertainty budget and review certificate before any validation target can be considered;
6. add acceptance-shield regression for forged digitization acceptance claims.
