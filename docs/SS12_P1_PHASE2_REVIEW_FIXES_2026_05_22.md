# SS12 P1 Phase 2 Review Fixes

Date: 2026-05-22 UTC
Scope: Source-packet extracted matrix and validator hardening after Claude/Codex/Hermes review.

## Evaluate

Independent review found three blocking issues:

1. The validator would allow a Phase 2 row marked `accepted` if it had `same_scope` and reviewed refs, even though this phase must remain non-promoting.
2. Source-path containment was lexical and did not resolve `..` before checking repository containment.
3. The extracted matrix global blocker still said `reviewed candidates`, contradicting the corrected `target_extraction_candidate` status.

## Fixes

- `scripts/validate_ss12_phase2_source_packet_matrix.py`
  - Resolves source paths with `Path.resolve(strict=False)` before containment checks.
  - Emits `accepted_row_forbidden_in_phase2` for any `status="accepted"` row.
  - Keeps the deeper accepted-row checks as defense-in-depth.

- `tests/test_ss12_phase2_source_packet_matrix.py`
  - Added mutation coverage proving an accepted row is rejected even with `scope_match="same_scope"` and `review_status="accepted"`.
  - Added mutation coverage proving `KnowledgeReference/../../outside-source.md` is rejected as outside-repo.
  - Keeps exact normalized quote equality for live source refs.

- `docs/SS12_P1_PHASE2_SOURCE_PACKET_MATRIX_EXTRACTED_2026_05_22.json`
  - Global blocker now says extracted values remain `target-extraction candidates`, not reviewed candidates.

## Verification

```text
.venv312/bin/python -m pytest tests/test_ss12_phase2_source_packet_matrix.py -q
9 passed in 0.50s
```

```text
.venv312/bin/python scripts/validate_ss12_phase2_source_packet_matrix.py --repo-root .
PASS /Users/anthonyzamora/dpf-unified/docs/SS12_P1_PHASE2_SOURCE_PACKET_MATRIX_EXTRACTED_2026_05_22.json: 0 source-packet matrix issues
```

```text
.venv312/bin/python -m pytest \
  tests/test_results_artifact_hygiene.py \
  tests/test_ss12_phase2_source_packet_matrix.py \
  tests/test_first_principles_acceptance_gate_dry_run.py -q
39 passed in 3.43s
```

```text
ruff check scripts/verify_active_results_artifact_hygiene.py \
  scripts/validate_ss12_phase2_source_packet_matrix.py \
  tests/test_results_artifact_hygiene.py \
  tests/test_ss12_phase2_source_packet_matrix.py
All checks passed!
```

## Learn

The current source-packet matrix is now intentionally non-promoting and fail-closed. It is an extraction staging artifact, not validation evidence sufficient for runtime acceptance.

## Continue

Next executable phase remains Helios-guided blocked-channel retrieval:

- density history
- EM field history
- temperature/distribution history
- startup BVP
- neutron spectrum or explicit same-scope absence proof
- uncertainty budget
- independent review certificate scaffold

Acceptance flags remain false.
