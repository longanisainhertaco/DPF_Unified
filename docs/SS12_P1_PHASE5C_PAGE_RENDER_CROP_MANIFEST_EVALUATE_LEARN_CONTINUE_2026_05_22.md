# SS12 P1 Phase 5-C Page Render / Crop Manifest — Evaluate / Learn / Continue

Date: 2026-05-22 UTC
Phase: PDF page rendering and crop-manifest staging

## Evaluate

Implemented Phase 5-C page-render artifact generation with crop/digitization fields deliberately left unselected and non-promoting.

Created/updated:

- `scripts/build_ss12_phase5c_page_render_manifest.py`
- `docs/SS12_P1_PHASE5C_PAGE_RENDER_CROP_MANIFEST_2026_05_22.json`
- `tests/test_ss12_phase5c_page_render_crop_manifest.py`
- `scripts/validate_ss12_phase5b_figure_asset_inventory.py`
- `tests/test_ss12_phase5b_figure_asset_inventory.py`

Generated page-render PNG artifacts under:

- `artifacts/ss12_phase5c/page_renders/`

The manifest contains four page-render artifacts mapped to Phase 5-B assets:

1. `pf1000_recent_progress_fig6_current_waveform`
2. `pf1000_scholz_fig4_density_distribution`
3. `pf1000_krauz_fig8_magnetic_probe_current`
4. `pf1000_scholz_fig9_neutron_timing`

Each render row records:

- source PDF path and SHA-256;
- page number;
- figure ID;
- render DPI;
- PNG path;
- PNG SHA-256;
- page width/height in pixels;
- crop fields as null;
- digitization hash as null;
- all acceptance/promotion flags as false.

## Guardrails implemented

Phase 5-C builder now fails closed before rendering when:

- Phase 5-B asset inventory validation fails;
- render DPI is outside 72..300;
- output manifest path is outside `docs/`;
- render directory path is outside `artifacts/`.

Phase 5-B validator now enforces path containment:

- `phase5_source_manifest` must resolve inside the repo;
- `repo_pdf` declared path must stay inside the repo;
- a repo-contained PDF symlink may resolve only inside the repo or under the explicitly allowed PDF vault `/Users/anthonyzamora/PDFs`;
- `external_pdf` must resolve under `/Users/anthonyzamora/PDFs`;
- PDF existence, file-ness, and SHA-256 are verified after containment checks.

No accepted state is created by rendering. Page images are source-location artifacts only.

## Tests added

Phase 5-C tests verify:

- builder generates the page-render manifest;
- one render artifact exists per Phase 5-B asset;
- rendered PNG files exist;
- rendered PNG SHA-256 values are recomputed from live bytes in tests;
- crop fields remain null;
- digitization hash remains null;
- acceptance and promotion flags remain false;
- invalid asset inventory blocks before render and creates no bad-render artifacts;
- render directory outside `artifacts/` is rejected;
- output manifest outside `docs/` is rejected;
- DPI lower/upper bounds are accepted at 72 and 300;
- out-of-bound DPI is rejected.

Phase 5-B tests verify:

- default inventory is accepted;
- acceptance flags are rejected at top level and row level;
- bad source PDF hash is rejected;
- Phase 5 source-manifest path escape is rejected;
- PDF path outside allowed roots is rejected;
- repo-contained symlink to the allowed PDF vault is accepted for the Krauz PDF;
- extracted packets require digitization hash;
- extracted-with-hash can exist only as an artifact state and still does not imply acceptance.

## Verification

Integrated focused verification:

```text
.venv312/bin/python -m pytest \
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

65 passed in 2.71s
```

Lint:

```text
ruff check scripts/validate_ss12_phase5b_figure_asset_inventory.py \
  scripts/build_ss12_phase5c_page_render_manifest.py \
  src/dpf/first_principles/figure_asset_inventory.py \
  tests/test_ss12_phase5b_figure_asset_inventory.py \
  tests/test_ss12_phase5c_page_render_crop_manifest.py \
  tests/test_first_principles_phase5b_figure_asset_packet_plan.py

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
  "suggestions": []
}
```

## Learn

- The Krauz PDF repo path is a symlink into `/Users/anthonyzamora/PDFs/01_DPF_Dense_Plasma_Focus/auxiliary/krauz2012.pdf`. The validator now models this explicitly instead of silently accepting arbitrary symlink escapes.
- Hash validation alone is not enough for scientific artifact pipelines. Containment must happen before file read/render operations.
- Page rendering is a useful intermediate artifact but still not evidence acceptance. The next phase must select figure crops, hash them, and record crop coordinates before digitization can be staged.

## Continue

Proceed to Phase 5-D: figure crop region selection and crop artifact generation.

Next requirements:

1. define crop region schema with PDF-point and pixel bounding boxes;
2. generate crop PNGs from the Phase 5-C page renders or source PDFs;
3. compute crop image hashes;
4. keep digitization values absent until axis calibration is explicit;
5. validate crop regions are inside page bounds;
6. reject crop output outside `artifacts/`;
7. add acceptance-shield regression ensuring crops do not promote first-principles acceptance.
