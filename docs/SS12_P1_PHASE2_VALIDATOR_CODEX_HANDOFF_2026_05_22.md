# SS12 P1 Phase 2 Validator Codex Handoff

Date: 2026-05-22 UTC
Scope: SS12 P1 Phase 2 source-packet matrix validator

## Implemented

- Added `scripts/validate_ss12_phase2_source_packet_matrix.py`.
- Added validator coverage to `tests/test_ss12_phase2_source_packet_matrix.py`.
- Left `docs/SS12_P1_PHASE2_SOURCE_PACKET_MATRIX_EXTRACTED_2026_05_22.json` unchanged.
- Did not inspect or mutate PDFs.
- Did not commit.

## Validator Gates

The script validates:

- all 15 required source-packet channels are present with no duplicate or unexpected channels;
- `acceptance_boundary.accepted_runtime_claim`,
  `acceptance_boundary.can_support_first_principles_acceptance`, and
  `acceptance_boundary.promotes_acceptance` are exactly `false`;
- any row marked `status="accepted"` must have `scope_match="same_scope"`;
- any accepted row must carry observable source refs whose `review_status` is
  `reviewed` or `accepted`;
- every text source ref resolves inside the repo;
- every text source ref has a valid 1-based line range that exists in the
  cited source file;
- PDF source refs fail closed rather than being read by this validator.

## Verification

```text
.venv312/bin/python -m pytest tests/test_ss12_phase2_source_packet_matrix.py -q
7 passed in 0.30s
```

```text
.venv312/bin/python scripts/validate_ss12_phase2_source_packet_matrix.py
PASS /Users/anthonyzamora/dpf-unified/docs/SS12_P1_PHASE2_SOURCE_PACKET_MATRIX_EXTRACTED_2026_05_22.json: 0 source-packet matrix issues
```

```text
.venv312/bin/python -m py_compile scripts/validate_ss12_phase2_source_packet_matrix.py tests/test_ss12_phase2_source_packet_matrix.py
```

```text
ruff check scripts/validate_ss12_phase2_source_packet_matrix.py tests/test_ss12_phase2_source_packet_matrix.py
All checks passed!
```

## Remaining Boundaries

- The extracted matrix remains non-promoting: no channel is accepted.
- Candidate rows are still candidates, even when line-cited.
- Independent review certificate, complete uncertainty budget, startup BVP,
  density history, EM history, temperature/distribution history, and same-scope
  neutron spectrum remain unclosed.
