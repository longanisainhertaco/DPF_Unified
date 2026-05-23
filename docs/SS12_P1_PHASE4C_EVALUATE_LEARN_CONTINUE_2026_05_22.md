# SS12 P1 Phase 4-C — Evaluate / Learn / Continue

Date: 2026-05-22 UTC
Phase: Figure-backed waveform/density target staging

## Evaluate

Added Phase 4-C TDD artifact:

- `tests/test_first_principles_figure_candidate_phase4c.py`

Added implementation:

- `src/dpf/first_principles/figure_candidate_staging.py`

Implemented behavior:

- `stage_figure_observable_candidate(...)` stages figure-derived observable candidates without promoting acceptance.
- Required metadata:
  - validation scope;
  - channel;
  - source path;
  - source line range;
  - figure id;
  - extraction method;
  - digitization hash;
  - uncertainty;
  - reviewer;
  - review state;
  - scope classification.
- Missing metadata returns `blocked_figure_candidate_incomplete`.
- Complete metadata returns `staged_figure_candidate_not_accepted`.
- Reviewed transfer candidates still cannot promote acceptance.
- Same-source candidates still require explicit `same_source_accepted` classification plus accepted certificate, and this staging helper still returns `accepted_observable_claim=false`.

TDD evidence:

```text
RED: ModuleNotFoundError: No module named 'dpf.first_principles.figure_candidate_staging'
GREEN: tests/test_first_principles_figure_candidate_phase4c.py -> 4 passed
```

Verification:

```text
.venv312/bin/python -m pytest tests/test_first_principles_figure_candidate_phase4c.py \
  tests/test_first_principles_circuit_power_port_phase4b.py \
  tests/test_first_principles_numerical_fidelity_phase4a.py \
  tests/test_ss12_phase2_source_packet_matrix.py \
  tests/test_ss12_phase3_transfer_candidate_matrix.py \
  tests/test_first_principles_acceptance_gate_dry_run.py -q

36 passed in 3.30s
```

```text
ruff check src/dpf/first_principles/figure_candidate_staging.py \
  src/dpf/first_principles/circuit_power_port.py \
  tests/test_first_principles_figure_candidate_phase4c.py \
  tests/test_first_principles_circuit_power_port_phase4b.py

All checks passed!
```

## Learn

The source retrieval phase found promising figure-backed transfer candidates, especially Rogowski/dI/dt traces and computed density plots. Phase 4-C prevents those from leaking into accepted observables before digitization uncertainty, reviewer identity, source scope, and certificate status are explicit.

## Continue

Next executable step: Phase 4-D cross-packet acceptance shield.

Required behavior:

- even if transfer linkage, circuit power-port packet, and figure staging packets exist, first-principles acceptance must remain false until source packet, numerical packet, uncertainty budget, and review certificate all pass;
- acceptance shield should reject any transfer-only or staged-only evidence bundle.

Acceptance flags remain false.
