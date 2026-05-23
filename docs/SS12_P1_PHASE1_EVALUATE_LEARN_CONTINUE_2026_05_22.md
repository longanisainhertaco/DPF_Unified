# SS12 P1 Phase 1 — Evaluate / Learn / Continue

Date: 2026-05-22 UTC
Phase: P1-0 active-artifact linter policy cleanup

## Evaluate

Codex implemented the current-behavior cleanup for the active results artifact hygiene contract.

Verified commands from `/Users/anthonyzamora/dpf-unified`:

```text
.venv312/bin/python -m pytest tests/test_results_artifact_hygiene.py -q
```

Result:

```text
16 passed in 0.41s
```

```text
.venv312/bin/python -m py_compile scripts/verify_active_results_artifact_hygiene.py tests/test_results_artifact_hygiene.py
```

Result: passed.

```text
.venv312/bin/python scripts/verify_active_results_artifact_hygiene.py --strict --check
```

Result:

```json
{
  "active_hit_count": 0,
  "clean": true,
  "ordinary_non_same_scope_source_fields": "allowed",
  "protected_key_chains": ["same_scope", "same_scope_source"]
}
```

```text
.venv312/bin/python -m pytest \
  tests/test_ss10_imported_pic_context_only_policy.py \
  tests/test_server_readiness.py \
  tests/test_ws9_runner_scope_source_geometry.py \
  tests/test_results_artifact_hygiene.py \
  tests/test_first_principles_acceptance_gate_dry_run.py -q
```

Result:

```text
85 passed in 23.13s
```

```text
ruff check scripts/verify_active_results_artifact_hygiene.py tests/test_results_artifact_hygiene.py
```

Result:

```text
All checks passed!
```

## Learn

The P0 safety property remains intact: no architecture/cross-scope material may appear under `same_scope` or `same_scope_source` evidence key chains in active result artifacts.

The P1-0 contract is now explicit: architecture or cross-scope evidence may appear in ordinary non-`same_scope` source fields, while approved context keys remain the recommended/canonical home for relocated cross-scope context. This avoids unnecessary schema churn before source-packet extraction while preserving the fail-closed same-scope boundary.

HeliosMatrix_KB WS3 eval completed, but the produced report mixed a stale cached 6-question BM25 mode with 29-question dense/hybrid modes. A forced WS3 rerun was started to recompute all four modes consistently.

## Continue

Proceed to Phase 2 source-packet matrix design and extraction once the forced Helios gold eval completes.

Next required artifacts:

- `docs/SS12_P1_PHASE2_SOURCE_PACKET_MATRIX_DESIGN_2026_05_22.md`
- a machine-readable PF-1000 full-energy source matrix
- line-cited channel decisions for geometry, bank/circuit, gas, current waveform, startup, density, EM, temperature/distribution, neutron scalar yield, neutron timing, spectrum, anisotropy, detector response, uncertainty, and review certificate

## Acceptance boundary

No physics blocker was closed in Phase 1. No acceptance state changed. `accepted_runtime_claim`, `can_support_first_principles_acceptance`, and certificate acceptance remain false.
