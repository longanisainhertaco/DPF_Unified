# SS12 P1 Phase 5 Figure Source Manifest — Evaluate / Learn / Continue

Date: 2026-05-22 UTC
Phase: Figure-backed source manifest and staged packet builder

## Evaluate

Implemented Phase 5 manifest, validator, and staged packet builder.

Created:

- `docs/SS12_P1_PHASE5_FIGURE_SOURCE_MANIFEST_2026_05_22.json`
- `scripts/validate_ss12_phase5_figure_source_manifest.py`
- `src/dpf/first_principles/figure_source_manifest.py`
- `tests/test_ss12_phase5_figure_source_manifest.py`
- `tests/test_first_principles_phase5_figure_packet_builder.py`

Manifest rows staged:

1. `current_waveform` — PF-1000 Fig. 6 Rogowski/PIN/dI/dt candidate from `recent-progress-in-1-mj-plasma-focus-research-d3e51f6c.md` lines 169-173.
2. `density_history` — PF-1000 computed density distribution candidate from `scholz-2006-pf1000-mega-joule.md` lines 303-305.
3. `em_field_history` — PF-1000 magnetic-probe / PCS current candidate from `experimental-study-of-the-structure-of-the-plasma-current-sheath-on-the-pf-1000-facility-705bcc83.md` lines 568-612.
4. `neutron_timing_or_spectrum` — PF-1000 neutron timing candidate from `scholz-2006-pf1000-mega-joule.md` lines 400-419. This is explicitly timing, not an accepted neutron spectrum.

## TDD record

RED:

```text
FileNotFoundError: docs/SS12_P1_PHASE5_FIGURE_SOURCE_MANIFEST_2026_05_22.json missing
```

GREEN:

```text
6 passed in 0.27s
```

Builder RED:

```text
ModuleNotFoundError: No module named 'dpf.first_principles.figure_source_manifest'
```

Builder GREEN:

```text
4 passed in 0.51s
```

## Independent review and fixes

Independent review initially failed Phase 5 for three guardrail gaps:

1. Validator did not reject row-level `accepted_figure_claim=true` or `accepted_runtime_claim=true`.
2. Accepted-state matching was exact/case-sensitive.
3. Builder staged alternate manifests without first reusing validator path/line/status checks.

Fixes applied:

- Row-level acceptance flags now use the full `ACCEPTANCE_FLAGS` tuple.
- Status/review/scope accepted-state checks now trim and lowercase values.
- Builder imports and runs `validate_manifest()` before staging any row.
- Invalid manifests return `blocked_phase5_manifest_invalid` with validator rule IDs in `blocking_reasons`.
- Source path escape is blocked before staging.

Regression tests added:

- `test_phase5_validator_rejects_row_level_acceptance_flags`
- case/whitespace accepted-state rejection via `status = " Accepted "`
- `test_phase5_builder_rejects_invalid_manifest_before_staging`
- `test_phase5_builder_rejects_source_path_escape_before_staging`

Second independent review passed with no security concerns or logic errors.

## Verification

Focused integrated verification:

```text
.venv312/bin/python -m pytest \
  tests/test_ss12_phase5_figure_source_manifest.py \
  tests/test_first_principles_phase5_figure_packet_builder.py \
  tests/test_first_principles_acceptance_shield_phase4d.py \
  tests/test_first_principles_figure_candidate_phase4c.py \
  tests/test_first_principles_circuit_power_port_phase4b.py \
  tests/test_first_principles_numerical_fidelity_phase4a.py \
  tests/test_ss12_phase2_source_packet_matrix.py \
  tests/test_ss12_phase3_transfer_candidate_matrix.py -q

40 passed in 0.69s
```

Lint:

```text
ruff check scripts/validate_ss12_phase5_figure_source_manifest.py \
  src/dpf/first_principles/figure_source_manifest.py \
  tests/test_ss12_phase5_figure_source_manifest.py \
  tests/test_first_principles_phase5_figure_packet_builder.py

All checks passed!
```

Static added-line scan:

```text
static_scan_findings 0
```

## Learn

- The manifest itself being valid is not enough; any runtime builder that accepts alternate manifests must call the validator first.
- Candidate-only workflows need row-level acceptance flags as well as top-level boundary flags.
- Accepted-state comparisons must be normalized or an attacker/operator typo can bypass fail-closed diagnostics.
- The density figure candidate remains transfer/computed evidence, not same-source accepted experimental density history.
- The neutron row is timing evidence only; same-scope neutron spectrum remains missing.

## Continue

Proceed to Phase 5-B: reproducible figure-region inventory and extraction packets.

Next work:

1. locate corresponding PDF/image regions for each manifest row;
2. create figure asset inventory with source file hash and page/figure hints;
3. extract image-region candidates without numeric acceptance;
4. compute digitization input hashes;
5. add uncertainty/reviewer placeholders;
6. keep all staged packets non-promoting until review certificate exists.
