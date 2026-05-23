# SS19 UQ, Comparator, and Certificate Pipeline

Date: 2026-05-23
Task: SS19 — UQ, comparator, and certificate pipeline

## Scope

SS19 adds a narrow certificate-pipeline evaluator that turns the existing comparator/UQ/certificate scaffolds into an executable fail-closed decision packet:

- comparator mapping completeness by observable;
- uncertainty-budget completeness for measurement, model, numerical, propagation, and observable uncertainty terms;
- source-packet and run-manifest SHA-256 provenance checks;
- upstream packet blocker scan;
- negative-control matrix;
- review-certificate gate;
- synthetic complete fixture positive path.

This is an engineering/certificate wiring step only. It does not accept any production DPF runtime, first-principles, full-3D, or validation claim.

## Implemented artifacts

- `src/dpf/first_principles/certificate_gate.py`
  - Added `build_ss19_certificate_pipeline(...)`.
  - Added SS19 required source-hash, comparator mapping, UQ, and negative-control matrices.
  - Added deterministic `certificate_artifact_hash` over the evaluated payload.
  - Preserved `accepted_runtime_claim=false`, `can_support_first_principles_acceptance=false`, and `promotes_acceptance=false` for every path.

- `tests/test_ss19_certificate_pipeline.py`
  - RED/GREEN coverage for incomplete production refusal with explicit blockers.
  - Complete production fixture refusal unless real review gate is later enabled.
  - Synthetic complete fixture acceptance for wiring only.
  - Synthetic fixture refusal when a required negative control is missing.

## Acceptance posture

Production remains refused even when all SS19 structural inputs are present:

- `status=refused_production_acceptance_disabled`
- `can_emit_certificate=false`
- `accepted_runtime_claim=false`
- `can_support_first_principles_acceptance=false`
- `promotes_acceptance=false`

The only positive status is `accepted_synthetic_complete_fixture`, which proves certificate wiring only and still keeps all runtime/first-principles acceptance flags false.

## Evaluate

Verification run:

```bash
python3 -m pytest tests/test_ss19_certificate_pipeline.py -q
# 4 passed in 0.56s

python3 -m pytest tests/test_ss19_certificate_pipeline.py tests/test_first_principles_certificate_negative_controls.py tests/test_uncertainty_budget.py -q
# 49 passed in 1.23s

python3 -m py_compile src/dpf/first_principles/certificate_gate.py tests/test_ss19_certificate_pipeline.py
# passed

python3 -m ruff check src/dpf/first_principles/certificate_gate.py tests/test_ss19_certificate_pipeline.py
# passed after import formatting fix

git diff --check -- src/dpf/first_principles/certificate_gate.py tests/test_ss19_certificate_pipeline.py
# passed
```

## Learn

- SS19 can now distinguish three states instead of a single static blocked packet: incomplete stack refusal, complete production refusal, and synthetic wiring-only acceptance.
- The complete production branch deliberately remains disabled because the required same-commit production evidence/review stack is not closed.
- The synthetic complete fixture is useful for preventing reviewer rubber-stamping: reviewers can inspect a positive wiring path while verifying that no runtime/public acceptance flag can flip.

## Continue

- Independent review should inspect the new `build_ss19_certificate_pipeline(...)` branch boundaries and verify that the synthetic fixture cannot promote production acceptance.
- SS20 should use this packet in the integration dry-run ledger and preserve `refused_production_acceptance_disabled` unless real same-scope evidence and review close.
- Future real acceptance work needs a separately reviewed production review gate; do not extend the synthetic fixture into production acceptance.

## Fix/Reverify Addendum — 2026-05-23

Parent review result: PASS; no certificate/UQ/comparator code changes were requested.

### Evaluate

Reverification run after review:

```bash
python3 -m pytest tests/test_ss19_certificate_pipeline.py tests/test_first_principles_certificate_negative_controls.py tests/test_uncertainty_budget.py -q
# 49 passed in 1.25s

python3 -m pytest tests/test_ss14_pf1000_source_packet_matrix.py tests/test_ss16_startup_bvp_evidence_packet.py tests/test_ss17_spatial_thermo_validation_packets.py tests/test_ss18_neutron_diagnostic_validation_stack.py tests/test_ss19_certificate_pipeline.py tests/test_first_principles_certificate_negative_controls.py tests/test_uncertainty_budget.py -q
# 83 passed in 1.33s

python3 scripts/validate_ss14_pf1000_source_packet_matrix.py && python3 scripts/validate_ss16_startup_bvp_evidence_packet.py && python3 scripts/validate_ss17_spatial_thermo_packet_matrix.py && python3 scripts/validate_ss18_neutron_diagnostic_packet.py
# PASS / OK for SS14, SS16, SS17, SS18 validators

python3 -m pytest tests/test_validation_artifacts.py tests/test_ss12_phase7a_review_certificate.py tests/test_first_principles_certificate_negative_controls.py tests/test_ss19_certificate_pipeline.py -q
# 64 passed in 1.72s

python3 scripts/audit_first_principles_artifacts.py 'results/**/*.json'
# PASS -- 81 JSON files scanned; 39 first-principles artifacts, 0 failed

python3 -m py_compile src/dpf/first_principles/certificate_gate.py tests/test_ss19_certificate_pipeline.py
python3 -m ruff check src/dpf/first_principles/certificate_gate.py tests/test_ss19_certificate_pipeline.py
# All checks passed
```

### Learn

- The post-review SS19 stack still refuses incomplete production evidence and complete production evidence, while allowing only `accepted_synthetic_complete_fixture` as wiring evidence.
- Packet validators for upstream SS14/SS16/SS17/SS18 evidence remain fail-closed and non-promoting.
- The artifact audit found no active first-principles artifact promotion failure.

### Continue

- Carry SS19 forward into SS20 as a non-promoting integration dry-run input.
- Preserve `accepted_runtime_claim=false`, `can_support_first_principles_acceptance=false`, and `promotes_acceptance=false` until real same-scope evidence, UQ, comparator, and review close in one reviewed production stack.
