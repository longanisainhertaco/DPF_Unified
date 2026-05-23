# SS12 P1-0 Codex Handoff

Date: 2026-05-22
Scope: Active-results linter policy cleanup only

## Decision

Selected option 2, current behavior. Hybrid-PIC / architecture source slugs may
appear in ordinary non-`same_scope` source fields. The protected chains remain:

- `same_scope`
- `same_scope_source`

Approved context keys such as `architecture_or_schema_context_sources`,
`cross_scope_context_sources`, and `source_scope_context` remain the recommended
home for relocated cross-scope context, not the only permitted location.

## Changed Files

- `scripts/verify_active_results_artifact_hygiene.py`: added explicit JSON
  policy fields `ordinary_non_same_scope_source_fields="allowed"` and
  `protected_key_chains=["same_scope", "same_scope_source"]`; scan behavior is
  unchanged.
- `tests/test_results_artifact_hygiene.py`: added a policy-output regression
  test for the current-behavior contract.
- `docs/SPRINT12_P0_COMPLETION_MEMO_2026_05_21.md`: updated the recorded
  authority-policy JSON and P1-0 wording to match the explicit fields.
- `docs/SS12_P1_0_CODEX_HANDOFF_2026_05_22.md`: this handoff.

## TDD Evidence

- RED: `.venv312/bin/python -m pytest tests/test_results_artifact_hygiene.py::TestAuthorityPolicyJson::test_policy_json_records_current_behavior_and_protected_chains -q`
  - Failed as intended with `KeyError: 'ordinary_non_same_scope_source_fields'`.
- GREEN: `.venv312/bin/python -m pytest tests/test_results_artifact_hygiene.py::TestAuthorityPolicyJson::test_policy_json_records_current_behavior_and_protected_chains -q`
  - `1 passed in 0.43s`.

## Verification

- `.venv312/bin/python -m pytest tests/test_results_artifact_hygiene.py -q`
  - `16 passed in 0.64s`.
- `.venv312/bin/python -m py_compile scripts/verify_active_results_artifact_hygiene.py tests/test_results_artifact_hygiene.py`
  - Passed.
- `.venv312/bin/python scripts/verify_active_results_artifact_hygiene.py --strict --check`
  - `clean=true`, `active_hit_count=0`.

## Acceptance Boundary

No physics acceptance work was started. No acceptance flags were changed.
`accepted_runtime_claim`, `can_support_first_principles_acceptance`, and
certificate/promoted acceptance remain false for this P1-0 cleanup.
