# First-Principles DPF Sprint 3 Completion Audit

Date: 2026-05-19

Repository: `/Users/anthonyzamora/dpf-unified`

Branch reviewed: `codex/corpus`

HEAD reviewed: `269d7d1` (`docs: Sprint 3 final submission`)

Controlling contract:
`docs/FIRST_PRINCIPLES_SPRINT3_COMPLETION_HANDOFF_2026_05_19.md`

Latest observed periodic audit:
`/private/tmp/dpf-unified-audit-logs/20260519T203626Z/summary.md`

## Verdict

Sprint 3 is not complete under the completion handoff contract.

The submitted work is useful and much of it is fail-closed, but it does not yet
satisfy the contract for a completed Sprint 3. The periodic audit and focused
pytest suites passed, yet the tests currently prove a weaker contract than the
handoff requires. Several control-plane ledgers are stale, multiple packet paths
point to non-existent modules, and three runtime paths can still produce
misleading or broken authority signals.

Decision:

- Do not start Sprint 4 as if Sprint 3 is closed.
- Treat the next sprint as `Sprint 3R: Remediation and Completion Gate`.
- Sprint 3R exits only when every finding in this audit is fixed or converted
  into a typed fail-closed blocker with a regression test.
- No validation claim, engineering-firm-ready claim, or accepted whole-shot
  claim is allowed from this work.

## Audit Method

1. Re-anchored in the live repository and verified the worktree was clean.
2. Compared Sprint 3 final-submission claims against the controlling handoff.
3. Read the packet ledgers, Sprint 3 status files, source modules, and tests.
4. Ran the DPF validation and SRS traceability preflight scripts.
5. Used five Codex subagents to audit independent lanes:
   - S3.1/S3.9 packet and control-plane integrity;
   - S3.2/S3.3 PF-1000 geometry and `Sigma_p` plumbing;
   - S3.4/S3.5 startup and closure packets;
   - S3.6/S3.7/S3.8 neutron, numerical, and certificate gates;
   - S3.9 source-truth and traceability.
6. Preserved `KnowledgeReference/` and tracked verified extracts as the only
   scientific authority. External libraries and tests were treated as
   engineering checks, not physics authority.

Commands or evidence used during this audit included:

```bash
git status --short --branch
git log --oneline -12
python3 /Users/anthonyzamora/.codex/skills/dpf-validation/scripts/dpf_skill_preflight.py /Users/anthonyzamora/dpf-unified
python3 /Users/anthonyzamora/.codex/skills/srs-traceability/scripts/srs_trace_audit.py /Users/anthonyzamora/dpf-unified
.venv312/bin/python -m pytest tests/test_external_team_submission_package.py tests/test_srs_traceability_export.py -q -rx
```

The repository-level periodic audit passed at HEAD `269d7d1`, but passing that
suite is not sufficient for Sprint 3 completion because the suite misses several
handoff-specific negative controls identified below.

## Accepted Progress

The team did deliver real progress:

- Sprint 3 research/spec packets were converted into runtime-facing structures
  in multiple modules.
- Startup, closure, neutron, numerical, and certificate paths mostly preserve
  the intended fail-closed posture.
- Reduced Lee/snowplow/scalar-yield paths remain comparators or baselines, not
  first-principles authority.
- Requirement rows were updated to `partial` rather than promoted to
  `implemented` or `accepted`.
- Focused suites reported green during the audit.

This progress remains valuable, but it is not yet complete enough to close the
Sprint 3 contract.

## Stop-The-Line Findings

### A1 - Legacy startup gate can still promote acceptance

Severity: critical.

Contract area: S3.4 startup BVP packet.

Evidence:

- `src/dpf/first_principles/startup_bvp.py:239` computes `can_support` from
  caller-supplied mode, review status, missing-channel state, and payload
  eligibility.
- `src/dpf/first_principles/startup_bvp.py:393` reads caller-declared
  `accepted_channels`.
- `src/dpf/first_principles/startup_bvp.py:275` embeds the typed
  `build_startup_packet()` output, but the legacy acceptance path is not bound
  to the typed packet's `can_support_first_principles_acceptance=false`.

Impact:

The typed packet itself is fail-closed, but a caller can construct a complete
payload and get `accepted_startup_bvp_packet` even while the embedded typed
packet says startup remains blocked. That violates the no-promotion rule.

Required fix:

- Force `build_startup_bvp_packet()` to derive acceptance from the typed
  `StartupPacket`, not from caller-declared accepted channels.
- Until a real source-backed startup BVP exists, every whole-shot startup call
  must return blocked startup authority.
- Add a regression test that attempts the false accepted payload and expects
  `can_support_first_principles_acceptance=false`.

### A2 - Neutron authority can falsely accept scalar or target-only evidence

Severity: critical.

Contract area: S3.6 neutron authority packet.

Evidence:

- `src/dpf/first_principles/neutron_authority.py:966` maps
  `neutron_scalar_yield` to `same_scope_scalar_yield`.
- `src/dpf/first_principles/neutron_authority.py:1020` labels any accepted
  channel as `accepted_neutron_authority`.

Impact:

Scalar yield or target metadata can produce an accepted authority label even
when mechanism-resolved runtime histories are absent. This contradicts the
first-principles requirement that total-yield authority depends on mechanism
separation, thermonuclear history, beam-target/stopping authority, spectra,
anisotropy, detector response, and UQ.

Required fix:

- Keep scalar yield as `candidate_comparator_only`.
- Do not emit `accepted_neutron_authority` unless the runtime mechanism packet
  and the same-scope review packet are both present.
- Add negative tests for scalar-only evidence, target-only evidence, and missing
  mechanism histories.

### A3 - Beam-target neutron diagnostic is broken on the active Python 3.12 / NumPy 2 lane

Severity: high.

Contract area: S3.6 neutron authority and Python 3.12 runtime.

Evidence:

- `src/dpf/diagnostics/beam_target.py:92` uses
  `getattr(np, "trapezoid", np.trapz)`.
- In NumPy 2.x, the default argument `np.trapz` is evaluated before `getattr`
  returns, and `np.trapz` may not exist.
- Integration paths call this helper at `beam_target.py:610`, `:648`, and
  `:662`.

Impact:

The neutron diagnostic can raise `AttributeError` before reaching the intended
fail-closed authority logic.

Required fix:

- Use a lazy fallback:
  `integrator = np.trapezoid if hasattr(np, "trapezoid") else np.trapz`.
- Add a NumPy 2 regression test for `_trapezoid_integral()`.

### A4 - PF-1000 material masks are still heuristic projections

Severity: high.

Contract area: S3.2 PF-1000/Akel geometry and material masks.

Evidence:

- `src/dpf/fields/source_geometry.py:855`, `:886`, `:896`, and `:915` mark
  revision-sensitive geometry fields as conflicts or blocked values.
- The partition then uses grid heuristics:
  `cage_outer = 0.75 * domain_r` at `source_geometry.py:1062`, a radial cathode
  shell at `source_geometry.py:1071`, and an axial-decile insulator band at
  `source_geometry.py:1082`.
- The insulator material mask is synthesized at `source_geometry.py:1085`
  despite missing insulator outer radius/wall thickness authority.

Impact:

The manifest hashes are deterministic, but deterministic does not mean
source-grounded. Consumers could treat a heuristic mask as a reviewed PF-1000
material boundary.

Required fix:

- Do not synthesize `insulator_material_faces` or cathode-rod/cage boundaries
  from blocked dimensions.
- If a material class needs a blocked dimension, the mask must be blocked or
  explicitly labeled `candidate_projection_not_source_mask`.
- Add tests for 12 rods, cage radius handling, insulator dimensions, chamber
  wall/backplate handling, and blocked-dimension fail-closed behavior.

### A5 - Geometry under-resolution gates miss required features

Severity: medium.

Contract area: S3.2 geometry and numerical readiness.

Evidence:

- `source_geometry.py:1018-1031` checks rod diameter and anode radius only.
- The handoff requires under-resolution handling for rods, insulator surfaces,
  and source-tagged transition widths.

Impact:

The runtime can pass a geometry manifest even when a required source-backed
surface is not resolvable on the grid.

Required fix:

- Add under-resolution gates for every source-supported material feature that
  is used by a mask.
- If a feature is blocked, do not generate an accepted/source mask for it.
- Add tests that fail on under-resolved insulator and plasma-transition
  features when those features become source-supported.

### A6 - `SigmaPSurfacePacket` is not yet the handoff schema

Severity: high.

Contract area: S3.3 `Sigma_p` surface packet plumbing.

Evidence:

- `src/dpf/fields/source_geometry.py:1238` defines the packet, but it lacks
  required digest fields such as `sigma_p_face_set_sha256`,
  `moving_classification_sha256`, `omega_partition_sha256`, and
  `material_mask_sha256_by_class`.
- It also lacks raw face operand arrays/statuses for `B`, `E`, `J`, `v`, and
  `eta` in a form suitable for audit replay.
- `build_sigma_p_surface_packet()` computes a geometry hash at
  `source_geometry.py:1424-1436`, but the default blocked return at
  `source_geometry.py:1437-1440` does not preserve that hash.

Impact:

The packet cannot yet support a reviewer-grade moving-boundary audit, even in a
fail-closed state.

Required fix:

- Add all required digest and operand fields.
- Preserve source geometry hash and mask hashes on blocked packets.
- Add tests that serialize the packet and verify digest stability.

### A7 - Serialized `Sigma_p` packets are ignored by `power_port.py`

Severity: medium.

Contract area: S3.3 `Sigma_p` power-port integration.

Evidence:

- `src/dpf/first_principles/power_port.py:459-466` says a
  `SigmaPSurfacePacket` instance or `to_dict()` form is accepted.
- `power_port.py:469-473` only accepts the instance; dict-form packets are
  discarded and replaced with a fresh blocked packet.
- `_sigma_p_surface_term()` at `power_port.py:498` checks operands but does not
  enforce moving/stationary classification or a Sigma_p sign-convention packet.

Impact:

Runtime telemetry emitted as a manifest dictionary will not be consumed by the
power-port ledger, and required negative controls are absent.

Required fix:

- Implement dict-form reconstruction or explicitly reject serialized packets
  with a named blocker.
- Add stationary-face zeroing and missing-sign-convention negative controls.
- Add tests that fail if dict-form packet data is dropped silently.

### A8 - Closure top-level effects omit required registered blockers

Severity: medium.

Contract area: S3.5 closure registry and regime gates.

Evidence:

- `src/dpf/first_principles/closure_packet.py:37` includes
  `electron_inertia` and `stopping_collisions` in `REQUIRED_EFFECTS`.
- Registry records exist at `closure_packet.py:822` and `:869`.
- `build_physics_closure_packet()` starts the top-level `effects` dictionary at
  `closure_packet.py:1171`, but the derived matrix at `closure_packet.py:1312`
  is built from an incomplete effects dictionary.

Impact:

The registry names the blockers, but the top-level closure matrix can omit them
from status summaries and missing-effect outputs.

Required fix:

- Include `electron_inertia` and `stopping_collisions` in top-level effects,
  `closure_matrix_status_by_effect`, and `missing_or_unaccepted_effects`.
- Add tests that compare `REQUIRED_EFFECTS` against the top-level matrix keys.

### A9 - Cross-restart merged ledger drops extended S3.7 channels

Severity: medium.

Contract area: S3.7 numerical acceptance harness.

Evidence:

- Live manifests emit extended fields at
  `src/dpf/first_principles/segmented_whole_shot.py:421-427`.
- `src/dpf/first_principles/segmented_whole_shot_combine.py:241-249` omits
  `cumulative_field_energy_delta_J`, `cumulative_pml_removed_energy_J`,
  `cumulative_power_port_work_J`, and `cumulative_ionization_step_count`.

Impact:

Segmented whole-shot merge artifacts are not same-scope with the live run
manifest, so restart reproducibility and ledger audit cannot be accepted.

Required fix:

- Preserve every extended S3.7 cumulative field through merged ledgers.
- Add an N-restart regression test that asserts all extended fields survive
  merge and are numerically aggregated under the documented convention.

### A10 - Packet ledgers and status documents contradict the final submission

Severity: high.

Contract area: S3.1 packet hygiene and S3.9 traceability.

Evidence:

- `sprint_3/SPRINT_3_FINAL_SUBMISSION.md:12-37` claims S3.1-S3.9 runtime
  foundations were delivered.
- `sprint_3/SPRINT_3_STATUS_LEDGER.md:8-11` still says
  `runtime_implementation_delivered=false`.
- `sprint_3/SPRINT_3_STATUS_LEDGER.md:29-44` lists S3.2-S3.9 as awaiting
  implementation.
- `BLOCKER_MATRIX.csv:18-26` still marks Sprint 3 rows pending and points to
  deleted `sprint_3/PENDING.md`.
- `README.md` and `THREE_SPRINT_FINAL_SUMMARY.md` repeat the stale state.

Impact:

The package cannot be handed to another team or engineering reviewer as a
self-consistent evidence packet.

Required fix:

- Rewrite all packet ledgers to distinguish:
  `research_packet_delivered`, `runtime_foundation_delivered`,
  `accepted_physics_delivered`, and `validation_delivered`.
- Remove live references to deleted `sprint_3/PENDING.md`.
- Add S3.1 and S3.9 rows to `CLAIMS_LEDGER.csv` and `TEST_MAP.csv`.
- Update tests so they reject the stale `runtime_implementation_delivered=false`
  field after runtime commits exist.

### A11 - Shorthand citations remain in actionable packet content

Severity: high for source-truth discipline.

Contract area: S3.1 citation normalization.

Evidence:

- `WP_N5_CLOSURE_REGISTRY_SOURCE_AUDIT.md:418`, `:421`, `:424`, `:426`,
  `:547`, and `:648` still contain shorthand `[KR: ...]` references.
- `tests/test_external_team_submission_package.py:431-433` only rejects
  `[KR: same file ...]`, so it misses other shorthand citations.

Impact:

Actionable implementation content can still cite ambiguous source text.

Required fix:

- Replace every actionable shorthand citation with exact local paths and line
  ranges.
- Expand the test regex to catch all `[KR: ...]` shorthand forms that do not
  include a repo-relative path and line range.

### A12 - Traceability points to non-existent modules and findings docs are stale

Severity: medium.

Contract area: S3.9 traceability and durable findings.

Evidence:

- `docs/DPF_REQUIREMENTS_BASELINE.md:73`,
  `docs/SRS_TRACEABILITY_MATRIX.csv:37`, `CLAIMS_LEDGER.csv:15`,
  `BLOCKER_MATRIX.csv:30`, and `CHANGELOG.md:301` reference
  `src/dpf/first_principles/closures.py`.
- The actual implementation is `src/dpf/first_principles/closure_packet.py`.
- `CHANGELOG.md:320`, `CLAIMS_LEDGER.csv:18`, `BLOCKER_MATRIX.csv:33`, and
  `SPRINT_3_STATUS_LEDGER.md:43` reference
  `src/dpf/first_principles/certificate.py`.
- The actual implementation is `src/dpf/first_principles/certificate_gate.py`.
- `CodexFindings.md` and `CortexFindings.md` had no Sprint 3 completion-audit
  entry before this audit.

Impact:

The RTM and handoff files cannot be trusted as precise code-navigation or audit
evidence until the paths are corrected.

Required fix:

- Replace `closures.py` with `closure_packet.py`.
- Replace `certificate.py` with `certificate_gate.py`.
- Regenerate the SRS traceability exports.
- Update both findings documents with the Sprint 3R status.

## Package Status By Work Package

| Work package | Audit status | Reason |
| --- | --- | --- |
| S3.1 packet hygiene | fail | Status ledgers stale, deleted `PENDING.md` still referenced, shorthand citation tests too narrow. |
| S3.2 PF-1000 geometry | partial | Metadata exists, but masks still use heuristic projection and blocked dimensions. |
| S3.3 `Sigma_p` plumbing | partial | Terms stay blocked, but packet schema, dict ingestion, sign controls, and moving/stationary controls are incomplete. |
| S3.4 startup packet | fail | Typed packet is fail-closed, but legacy builder can still accept caller payloads. |
| S3.5 closure registry | partial | Registry exists; top-level matrix omits required registered blockers. |
| S3.6 neutron authority | fail | Scalar/target-only channels can be labeled accepted; beam-target helper breaks on NumPy 2. |
| S3.7 numerical harness | partial | Live manifest emits extended fields, merged ledger drops them. |
| S3.8 certificate scaffold | partial pass | Runtime gate appears fail-closed; docs point to wrong module path. |
| S3.9 traceability | fail/partial | RTM exists, but stale ledgers, wrong paths, and findings-doc drift remain. |

## Required Next Action

Create and execute Sprint 3R. Sprint 3R is a remediation sprint, not a new
feature sprint. It must close the audit findings above with code, tests, and
traceability updates. Sprint 4 work may be researched in parallel, but it must
not be used to bypass Sprint 3R completion.
