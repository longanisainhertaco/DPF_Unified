# SS12 P1 Phase 6-B UQ Propagation Scaffold — Evaluate / Learn / Continue

Date: 2026-05-22 UTC
Phase: UQ propagation packet scaffold and completion gate

## Evaluate

Implemented Phase 6-B propagation scaffolding for the four Phase 6-A UQ budget rows.

Created/updated:

- `docs/SS12_P1_PHASE6B_UQ_PROPAGATION_SCAFFOLD_2026_05_22.json`
- `scripts/validate_ss12_phase6b_uq_propagation_scaffold.py`
- `tests/test_ss12_phase6b_uq_propagation_scaffold.py`

The scaffold contains one propagation packet per Phase 6-A UQ row:

1. `pf1000_recent_progress_fig6_current_waveform`
2. `pf1000_scholz_fig4_density_distribution`
3. `pf1000_krauz_fig8_magnetic_probe_current`
4. `pf1000_scholz_fig9_neutron_timing`

Each packet is intentionally blocked:

- `propagation_status: blocked_uq_budget_incomplete`
- `propagated_observable: null`
- `propagated_uncertainty: null`
- `review_certificate_status: missing`
- all acceptance and promotion flags are false

No propagation output is complete or accepted in Phase 6-B.

## Guardrails implemented

The Phase 6-B validator fails closed when:

- scaffold JSON is not an object;
- top-level acceptance/promotion flags are not false;
- packet-level acceptance/promotion flags are not false;
- propagation packet IDs are duplicated;
- UQ row references are duplicated;
- the Phase 6-A scaffold reference is anything other than the exact canonical string `docs/SS12_P1_PHASE6A_UQ_BUDGET_SCAFFOLD_2026_05_22.json`;
- propagation packets do not map one-to-one to canonical Phase 6-A UQ rows;
- propagation packet `figure_source_id` or `digitization_packet_id` does not match the referenced Phase 6-A row;
- blocked propagation packets carry propagated observable or uncertainty values;
- complete propagation packets lack finite observable values;
- complete propagation packets lack finite nonnegative propagated uncertainty;
- complete propagation packets lack a complete review certificate;
- NaN/Infinity constants appear in JSON;
- arbitrary propagation or review status strings are used.

Allowed status values:

- propagation status: `blocked_uq_budget_incomplete`, `complete_not_accepted`
- review certificate status: `missing`, `complete`

## Verification

Focused Phase 6-B verification:

```text
.venv312/bin/python -m pytest tests/test_ss12_phase6b_uq_propagation_scaffold.py -q

12 passed
```

Focused integrated Phase 2–6B verification:

```text
.venv312/bin/python -m pytest \
  tests/test_ss12_phase6b_uq_propagation_scaffold.py \
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

116 passed
```

Lint:

```text
ruff check scripts/validate_ss12_phase6b_uq_propagation_scaffold.py tests/test_ss12_phase6b_uq_propagation_scaffold.py

All checks passed!
```

Static scan:

```text
phase6b_static_scan_findings 0
```

Phase 6-B acceptance-promotion scan:

```text
0 matches for accepted_* true, promotes_acceptance true, or can_support_first_principles_acceptance true
```

Independent final review:

```json
{
  "passed": true,
  "security_concerns": [],
  "logic_errors": [],
  "summary": "Phase 6-B final review passed. No unsafe shell/path traversal or propagation acceptance leaks found."
}
```

## Learn

- Completion gates need explicit tests for arbitrary status strings and blocked rows carrying accidental values; validators alone are easy to under-test.
- Propagation must remain separate from acceptance: a finite propagated value is not accepted without review-certificate completion.
- Canonical upstream linkage should use exact string references and validator-owned canonical paths, not scaffold-controlled filesystem paths.
- Top-level JSON type checks prevent traceback-style failure and preserve machine-readable validation output.

## Continue

Proceed to Phase 6-C: power-port certification scaffold.

Next requirements:

1. consume canonical Phase 4-B circuit power-port packet and Phase 6-B propagation scaffold;
2. define crowbar timing, current-sheath acceleration, and pinch-focus dynamics certification rows;
3. fail closed while power-port evidence and UQ propagation remain incomplete;
4. keep all acceptance and runtime flags false;
5. prepare Phase 7 review-certificate linkage without promoting acceptance.
