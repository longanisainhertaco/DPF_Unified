# SS12 P1 Phase 6-A UQ Budget Scaffold — Evaluate / Learn / Continue

Date: 2026-05-22 UTC
Phase: UQ budget scaffold from staged digitization packets

## Evaluate

Implemented Phase 6-A uncertainty-budget scaffolding for the four Phase 5-E digitization packets.

Created/updated:

- `docs/SS12_P1_PHASE6A_UQ_BUDGET_SCAFFOLD_2026_05_22.json`
- `scripts/validate_ss12_phase6a_uq_budget_scaffold.py`
- `tests/test_ss12_phase6a_uq_budget_scaffold.py`

The scaffold contains one UQ row per Phase 5-E digitization packet:

1. `pf1000_recent_progress_fig6_current_waveform`
2. `pf1000_scholz_fig4_density_distribution`
3. `pf1000_krauz_fig8_magnetic_probe_current`
4. `pf1000_scholz_fig9_neutron_timing`

Each row is intentionally blocked:

- `uq_status: blocked_digitization_not_reviewed`
- all uncertainty terms are null
- `combined_uncertainty: null`
- `review_certificate_status: missing`
- all acceptance and promotion flags are false

No uncertainty budget is complete or accepted in Phase 6-A.

## Guardrails implemented

The Phase 6-A validator fails closed when:

- top-level acceptance/promotion flags are not false;
- row-level acceptance/promotion flags are not false;
- UQ row IDs are duplicated;
- digitization packet IDs are duplicated in UQ rows;
- the Phase 5-E scaffold reference is anything other than the exact canonical string `docs/SS12_P1_PHASE5E_DIGITIZATION_SCAFFOLD_2026_05_22.json`;
- external, traversal, traversal-to-canonical, or absolute canonical Phase 5-E paths are supplied;
- UQ rows do not map one-to-one to canonical Phase 5-E digitization packets;
- UQ row `figure_source_id` or `crop_artifact_id` does not match the referenced Phase 5-E packet;
- blocked UQ rows carry uncertainty values;
- complete UQ rows lack any of the five uncertainty terms;
- complete UQ rows lack `combined_uncertainty`;
- complete UQ rows lack a complete review certificate;
- NaN/Infinity constants appear in JSON;
- arbitrary status strings are used;
- uncertainty values are negative, boolean, non-numeric, or non-finite.

Required uncertainty terms for complete rows:

- source uncertainty
- digitization uncertainty
- calibration uncertainty
- numerical uncertainty
- model-inadequacy uncertainty
- combined uncertainty

Allowed status values:

- UQ status: `blocked_digitization_not_reviewed`, `complete_not_accepted`
- review certificate status: `missing`, `complete`

## Verification

Focused Phase 6-A verification:

```text
.venv312/bin/python -m pytest tests/test_ss12_phase6a_uq_budget_scaffold.py -q

15 passed
```

Focused integrated Phase 2–6A verification:

```text
.venv312/bin/python -m pytest \
  tests/test_ss12_phase6a_uq_budget_scaffold.py \
  tests/test_ss12_phase5e_digitization_scaffold.py \
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

104 passed
```

Lint:

```text
ruff check scripts/validate_ss12_phase6a_uq_budget_scaffold.py tests/test_ss12_phase6a_uq_budget_scaffold.py

All checks passed!
```

Static scan:

```text
phase6a_static_scan_findings 0
```

Phase 6-A acceptance-promotion scan:

```text
0 matches for accepted_* true, promotes_acceptance true, or can_support_first_principles_acceptance true
```

Independent final review:

```json
{
  "passed": true,
  "security_concerns": [],
  "logic_errors": [],
  "summary": "Phase 6-A final review passed after exact canonical Phase 5-E reference hardening."
}
```

## Learn

- Linkage validators must pin upstream manifests by exact canonical reference, not only by resolved path; `docs/../docs/...` can resolve to the right file while still weakening the no-traversal guarantee.
- UQ budget completion must require all uncertainty terms and a review certificate before any row can be used downstream.
- Blocked UQ rows should not carry partially populated numeric uncertainty values; otherwise later code may accidentally consume them.
- JSON non-finite constants must be rejected at parse time and numeric terms must also be finite/nonnegative.

## Continue

Proceed to Phase 6-B: UQ budget completion gate and propagation packet scaffold.

Next requirements:

1. consume Phase 6-A rows without accepting them;
2. build propagation packet slots for source, digitization, calibration, numerical, and model-inadequacy terms;
3. fail closed while any upstream digitization/UQ/review state is missing;
4. reject any complete propagation row without canonical UQ linkage;
5. keep all acceptance and runtime flags false;
6. prepare Phase 7 review-certificate linkage but do not promote acceptance.
