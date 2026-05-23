# SS12 P1 Phase 5-B Figure Asset Inventory — Evaluate / Learn / Continue

Date: 2026-05-22 UTC
Phase: Figure asset inventory and extraction packet plan

## Evaluate

Implemented Phase 5-B asset inventory, validator, and fail-closed extraction packet planner.

Created:

- `docs/SS12_P1_PHASE5B_FIGURE_ASSET_INVENTORY_2026_05_22.json`
- `scripts/validate_ss12_phase5b_figure_asset_inventory.py`
- `src/dpf/first_principles/figure_asset_inventory.py`
- `tests/test_ss12_phase5b_figure_asset_inventory.py`
- `tests/test_first_principles_phase5b_figure_asset_packet_plan.py`

Asset rows now map one-to-one to the Phase 5 figure-source manifest:

1. `pf1000_recent_progress_fig6_current_waveform`
   - PDF: `downloaded_books_papers/Research Papers/2026-05-20-user-supplied-papers/scholz_2001_recent_progress_1mj_pf_research.pdf`
   - SHA-256: `d3e51f6c56f734e871f657f950486be441f75df9b75660e4524675738b002c75`
   - Page hint: 4
   - Status: asset located, not extracted.

2. `pf1000_scholz_fig4_density_distribution`
   - PDF: `/Users/anthonyzamora/PDFs/01_DPF_Dense_Plasma_Focus/scholz-2006-pf1000-mega-joule.pdf`
   - SHA-256: `cb68ba976ce8d02da7e5f174d0b1732ce49aad00b606656799b5d3b7227d09e8`
   - Page hint: 3
   - Status: asset located, not extracted.
   - Boundary: computed/transfer figure only; not accepted experimental density history.

3. `pf1000_krauz_fig8_magnetic_probe_current`
   - PDF: `downloaded_books_papers/Research Papers/2026-05-11-user-ingest/krauz2012.pdf`
   - SHA-256: `705bcc836646b1887ea985d4d28e7fd0bbae6ce326797d0e1a8cdd70bdb162b2`
   - Page hint: 10
   - Status: asset located, not extracted.

4. `pf1000_scholz_fig9_neutron_timing`
   - PDF: `/Users/anthonyzamora/PDFs/01_DPF_Dense_Plasma_Focus/scholz-2006-pf1000-mega-joule.pdf`
   - SHA-256: `cb68ba976ce8d02da7e5f174d0b1732ce49aad00b606656799b5d3b7227d09e8`
   - Page hint: 5
   - Status: asset located, not extracted.
   - Boundary: timing candidate only; same-scope neutron spectrum remains missing.

## TDD record

RED:

```text
FileNotFoundError: docs/SS12_P1_PHASE5B_FIGURE_ASSET_INVENTORY_2026_05_22.json
```

GREEN:

```text
7 passed in 0.36s
```

Builder RED:

```text
ModuleNotFoundError: No module named 'dpf.first_principles.figure_asset_inventory'
```

Builder GREEN:

```text
5 passed in 0.51s
```

## Guardrails implemented

Validator enforces:

- inventory top-level acceptance boundary flags must be false;
- row-level acceptance flags must be false;
- figure asset rows must match the Phase 5 source manifest one-to-one;
- referenced PDF paths must exist;
- referenced PDF SHA-256 must match live files;
- duplicate asset IDs and duplicate source IDs are rejected;
- `not_extracted` packets may not carry digitization hashes;
- `extracted` packets require digitization hashes;
- extracted-with-hash is allowed only as a digitized artifact state, not as an accepted state.

Builder enforces:

- missing inventory returns `blocked_phase5b_asset_inventory_missing`;
- invalid inventory returns `blocked_phase5b_asset_inventory_invalid` before task generation;
- valid inventory returns non-promoting extraction tasks with required outputs:
  - figure-region crop;
  - region hash;
  - axis calibration;
  - digitized curve or observable candidate;
  - uncertainty budget;
  - review certificate reference.

## Independent review

Independent review passed with no security concerns or logic errors.

Suggestions from review were addressed:

- added top-level acceptance-boundary mutation regression;
- added future-facing extracted-with-hash regression confirming no acceptance promotion.

## Verification

Focused integrated verification:

```text
.venv312/bin/python -m pytest \
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

54 passed in 0.97s
```

Lint:

```text
ruff check scripts/validate_ss12_phase5b_figure_asset_inventory.py \
  src/dpf/first_principles/figure_asset_inventory.py \
  tests/test_ss12_phase5b_figure_asset_inventory.py \
  tests/test_first_principles_phase5b_figure_asset_packet_plan.py

All checks passed!
```

Static scan:

```text
static_scan_findings 0
```

Acceptance-promotion search over Phase 5 JSON docs:

```text
0 matches
```

## Learn

- The Scholz 2006 PF-1000 PDF is not currently under the repo path referenced by its KnowledgeReference JSON, but it exists in `/Users/anthonyzamora/PDFs/01_DPF_Dense_Plasma_Focus/`; Phase 5-B records this as `external_pdf` with live hash.
- Figure-source text evidence is insufficient for digitization. The next step needs actual image-region crops tied to source PDF hashes.
- Allowing `extracted` with a digitization hash is useful for future Phase 5-C, but acceptance still requires uncertainty and independent review.

## Continue

Proceed to Phase 5-C: actual PDF page rendering and region-crop artifact generation.

Next requirements:

1. render each source PDF page to an image artifact;
2. create crop manifests for figure regions;
3. compute page image hash and region hash;
4. keep all digitized values absent until axis calibration is explicit;
5. add tests that extracted crops still do not promote acceptance.
