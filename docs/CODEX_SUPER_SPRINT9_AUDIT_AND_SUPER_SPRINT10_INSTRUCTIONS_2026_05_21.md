# Codex Super-Sprint 9 Audit and Super-Sprint 10 Instructions

Date: 2026-05-21  
Branch audited: `codex/corpus`  
Audited HEAD: `a4c28b9225ed45fc15f188d2f16810d340eda610`  
Prior baseline: `814ab10`

## 1. Source and Claim Boundary

This audit uses the project fail-closed rule:

- `KnowledgeReference/` and already promoted KR-derived packets are the only
  scientific authority.
- Raw PDFs, raw external repositories, PlasmaPy, and engineering probes can be
  acquisition/cross-check/runtime context only until target-extracted, reviewed,
  and consumed by code with tests.
- No reduced model, imported PIC state, text-only extraction, or short engineering
  probe can promote `accepted_runtime_claim` or
  `can_support_first_principles_acceptance`.

Super-Sprint 9 was not a validation sprint. It was supposed to make the PF-1000
full-energy package-native 3-D runtime path internally coherent as an
engineering-experiment surface.

## 2. Audit Verdict

**Verdict: not accepted as complete. Accept the P0 scope/source repair, but
reject the full Super-Sprint 9 completion claim until the corrections below
land.**

What is solid:

- The top-level PF-1000 full-energy runtime now carries
  `validation_scope=pf1000_full_energy_27_to_40_kv`.
- The top-level selected-machine source scope is now
  `pf1000_scholz_2000_2001_24rod_large_electrode_full_energy_source`.
- The LLNL-like hybrid-PIC source is retained as architecture/equation-method
  evidence, not as the selected-machine source scope.
- The Akel 16 kV helper is no longer matching generic PF-1000 full-energy scope.
- Bennett startup evidence is exposed as wrong-scope candidate context and does
  not promote startup acceptance.
- No accidental runtime acceptance promotion was found.

What is not complete:

- Runtime candidate evidence still contains a misleading
  `same_scope_3d_validation_packet` populated from the LLNL-like hybrid-PIC
  geometry packet.
- PF-1000 geometry-mask runtime telemetry does not emit the five blocked
  geometry fields required by WS9-6.
- Hollow-anode telemetry contradicts the selected deck: the deck deliberately
  leaves `anode_inner_radius_m=None`, while runtime feature telemetry marks the
  hollow anode as source-declared.
- The segmented whole-shot manifest does not directly carry the validation
  packet summary, same-scope summary, or power-port Sigma-p blocker summary
  required by WS9-7.
- The Sprint 9 memo and findings docs overstate completion and contain stale
  transcript details.

## 3. Commands and Evidence

Commands run by Codex:

```bash
git status --short
git log --oneline -8
.venv312/bin/python -m pytest \
  tests/test_ws9_runner_scope_source_geometry.py \
  tests/test_first_principles_same_scope_full_energy_packet.py \
  tests/test_git_status_clean_exception.py \
  tests/test_startup_breakdown_audit.py \
  tests/test_first_principles_startup_bvp.py \
  tests/test_sprint8_ws4_bennett_startup.py -q
.venv312/bin/python -m ruff check src tests
.venv312/bin/python scripts/run_codex_periodic_audit.py --timeout-seconds 900
.venv312/bin/python -m pytest tests/test_server_readiness.py -q
```

Observed:

- Focused Sprint 9/startup/git-status tests: `194 passed`.
- Ruff: `All checks passed!`.
- Periodic audit: `10/10 PASS` under the approved 146-line external-churn
  exception, log
  `/private/tmp/dpf-unified-audit-logs/20260521T043943Z`.
- Physical worktree is not clean: 145 PDF typechanges plus `m external/athenak`.
  The audit gate passes only because these are explicitly classified as approved
  external churn.
- `tests/test_server_readiness.py` still fails. Codex reproduced the first
  failure before pytest stopped at the first failure:
  `first_principles_startup_initialization.status` is
  `rejected_startup_mode_for_first_principles` instead of the expected
  `blocked_startup_bvp_packet_not_available`.

Direct runtime probe:

```text
deck.validation_scope = pf1000_full_energy_27_to_40_kv
deck.selected_machine_source_scope = pf1000_scholz_2000_2001_24rod_large_electrode_full_energy_source
validation_packet.source_scope = pf1000_scholz_2000_2001_24rod_large_electrode_full_energy_source
telemetry.architecture_source_scope = llnl_like_180ka_axisymmetric_hybrid_pic
same_scope.declared_scope = pf1000_full_energy_27_to_40_kv
can_support = False
```

This confirms the top-level scope/source repair.

It also confirmed the residual problems:

```text
candidate_evidence.same_scope_3d_validation_packet.source_scope =
  llnl_like_180ka_axisymmetric_hybrid_pic

conductor_mask blocked field names present =
  hollow_anode_bore_length: false
  insulator_wall_thickness: false
  backplate_radial_extent: false
  backplate_axial_thickness: false
  same_scope_reviewed_mask: false

manifest has validation_packet = false
```

## 4. Audit Findings

### A1 - High - Misleading same-scope 3-D candidate evidence remains

`src/dpf/first_principles/runner.py` still emits
`candidate_evidence.same_scope_3d_validation_packet` from
`HybridPICSourceGeometry`. That packet carries
`source_scope=llnl_like_180ka_axisymmetric_hybrid_pic`.

This is not a runtime acceptance leak because the packet remains
`can_support_first_principles_acceptance=false`, but it violates the Sprint 9
intent to separate architecture evidence from selected-machine same-scope
evidence.

Required correction:

- Rename or split the packet so LLNL-like hybrid-PIC evidence is emitted under an
  architecture-only key, for example `architecture_3d_geometry_candidate_packet`.
- Only emit a `same_scope_3d_validation_packet` when its `source_scope` is the
  selected-machine PF-1000 source scope and it represents same-scope validation
  evidence.
- Add a runtime test asserting that no key containing `same_scope` carries the
  LLNL-like architecture source scope for the PF-1000 full-energy preset.

### A2 - High - WS9-6 geometry blockers are not emitted by runtime telemetry

The PF-1000 deck builds a blocked-field manifest for:

- `anode_hollow_bore_length_m`
- `insulator_wall_thickness_m`
- `backplate_radial_extent_m`
- `backplate_axial_thickness_m`
- `same_scope_reviewed_geometry_mask`

The conductor-mask runtime telemetry does not expose those blocked fields, and
the segmented manifest does not expose them either.

Required correction:

- Thread the deck blocked-field manifest into
  `boundary_policy.conductor_mask.blocked_geometry_fields`.
- Include blocker IDs, blocked status, and the source/scope reason for each
  field.
- Add tests that inspect the actual runtime result and segmented manifest, not
  only the deck helper.

### A3 - High - Hollow-anode telemetry contradicts the selected deck

The PF-1000 full-energy deck explicitly leaves `anode_inner_radius_m=None` and
states that the anode is not declared hollow for this scope. Runtime telemetry
currently reports `hollow_anode_declared_by_source=true` inside
`pf1000_geometry_features` whenever the PF-1000 rod projection is active.

Required correction:

- Set `hollow_anode_declared_by_source` only when a KR-supported inner radius or
  hollow-bore geometry field is actually present for the selected scope.
- For the current deck, report:
  `hollow_anode_declared_by_source=false`,
  `hollow_anode_inner_radius_supplied=false`, and a blocked-field entry for the
  missing bore length/radius.
- Add a regression test asserting this exact state.

### A4 - High - WS9-7 segmented manifest does not carry required handoff evidence

The Sprint 9 instruction required the engineering probe manifest to record
source scope, same-scope scope, limiter counts, duration status, and power-port
blockers. The manifest records deck/plan/segment/blocker structures, but does
not directly embed a validation-packet summary, same-scope summary, or
power-port Sigma-p blocker summary.

Required correction:

- Add a compact, stable `first_principles_scope_summary` block to the segmented
  manifest:
  - `validation_scope`
  - `selected_machine_source_scope`
  - `architecture_source_scope`
  - `can_support_first_principles_acceptance=false`
  - `accepted_runtime_claim=false`
- Add `same_scope_summary` with declared scope and channel states.
- Add `power_port_summary` with Auluck/Sigma-p term status and terms II/IV/V/VI
  blocked on the reviewed face set.
- Add `geometry_blocker_summary` with the five blocked geometry fields.
- Add tests that run the six-step segmented probe to a temp directory and assert
  those exact manifest fields.

### A5 - Medium - Imported-PIC context-only policy is not fully encoded

The replacement test says imported-PIC startup payloads are context-only. The
runtime is functionally fail-closed today because the typed startup packet blocks
acceptance. However, `imported_pic_sheath_state` still appears in
`ACCEPTED_STARTUP_MODES`, and a complete reviewed imported-PIC payload can still
be marked `channel_acceptance_eligible=true` at the payload-review layer.

Required correction:

- Either remove `imported_pic_sheath_state` from `ACCEPTED_STARTUP_MODES`, or add
  an explicit `CONTEXT_ONLY_STARTUP_MODES` class that cannot satisfy
  `mode_is_accepted`.
- If imported PIC is ever reintroduced as an acceptance path, it needs a new
  source-reviewed policy packet with unit/hash/source/scope checks and an
  explicit same-scope justification. Do not leave it half-accepted by mode name.
- Add a test that forces the typed startup packet accepting path with an
  imported-PIC payload and proves the imported-PIC mode still cannot promote.

### A6 - Medium - WS9-8 artifact-regeneration claim is not evidenced

The Sprint 9 memo says RTM, source-truth, and module-vetting artifacts were
regenerated. The committed diff shows module-vetting updates, but not
`DPF_REQUIREMENTS_BASELINE.md`, `SRS_TRACEABILITY_MATRIX.*`, or
`FIRST_PRINCIPLES_SOURCE_TRUTH_INDEX.*`.

Required correction:

- Regenerate the RTM/source-truth artifacts and commit them, or update the memo
  and findings docs to say only module-vetting was regenerated.
- Do not claim traceability regeneration without a changed artifact or a command
  transcript proving a no-op check.

### A7 - Medium - Sprint 9 memo/finding transcript is stale

The completion memo says there are two Sprint 9 commits, but HEAD contains four:

- `2b2f290`
- `e8a4818`
- `00d7016`
- `a4c28b9`

The memo also reports an `825 passed` transcript that did not reproduce from the
explicit runnable slices in this audit. The gate still passes, but the transcript
is not exact as written.

Required correction:

- Update the completion memo or add an addendum that lists all four commits.
- Replace stale test-count prose with exact commands and current outputs.
- Fix findings-doc text that says `test_startup_breakdown_audit.py` remains a
  pre-existing failure; the rewritten startup tests passed in this audit.

### A8 - Informational - No acceptance promotion found

Codex and subagents found no accidental promotion of first-principles runtime
acceptance. The runtime remains engineering-candidate/fail-closed.

Required preservation:

- Keep every acceptance flag false until a same-commit packet passes source,
  numerical, comparator, UQ, and certificate gates.

## 5. Super-Sprint 10 Instructions

Super-Sprint 10 is a correction and gate-instrumentation sprint. Its first goal
is to close every Sprint 9 audit finding above. Its second goal is to make the
next physics sprint measurable by running report-only acceptance gates that fail
closed with exact missing inputs.

### SS10-0 - Correct Handoff Narrative and Findings

Required work:

- Add a Sprint 9 audit addendum or revise the Sprint 9 memo to reflect this
  audit verdict.
- Update `CodexFindings.md` and `CortexFindings.md` to say:
  top-level scope/source repair accepted; WS9-6/WS9-7/WS9-8 incomplete; no
  acceptance promoted.
- Remove stale "two commits" and stale test-count wording.

Acceptance tests:

```bash
rg -n "Two commits|825 passed|test_startup_breakdown_audit.py.*pre-existing" \
  docs/SPRINT9_COMPLETION_MEMO_2026_05_20.md CodexFindings.md CortexFindings.md
```

This command must return no stale completion claims unless quoted as historical
input inside this audit document.

### SS10-1 - Split Architecture Evidence from Same-Scope 3-D Evidence

Required work:

- Replace `candidate_evidence.same_scope_3d_validation_packet` for LLNL-like
  architecture evidence with an architecture-only key.
- Reserve `same_scope_3d_validation_packet` for selected-machine same-scope
  evidence only.
- Preserve `can_support_first_principles_acceptance=false`.

Required tests:

```bash
.venv312/bin/python -m pytest tests/test_ws9_runner_scope_source_geometry.py -q
```

Add a new assertion that no PF-1000 full-energy runtime packet with a key
containing `same_scope` carries `llnl_like_180ka_axisymmetric_hybrid_pic`.

### SS10-2 - Complete PF-1000 Geometry-Mask Runtime Integrity

Required work:

- Emit `blocked_geometry_fields` from the deck into conductor-mask telemetry.
- Include the same list in the segmented manifest summary.
- Fix hollow-anode telemetry to match the selected deck.
- Keep the mesh-resolution warning.

Required tests:

```bash
.venv312/bin/python -m pytest tests/test_ws9_runner_scope_source_geometry.py \
  tests/test_source_geometry_packet.py -q
```

The tests must inspect the runtime result from
`pf1000_scholz_2001_24rod_full_energy_deck()`, not only helper dictionaries.

### SS10-3 - Make the Segmented Probe Manifest Audit-Complete

Required work:

- Add `first_principles_scope_summary`, `same_scope_summary`,
  `power_port_summary`, and `geometry_blocker_summary` to the segmented manifest.
- Keep the manifest compact. Do not dump enormous nested arrays or fields.
- Include enough fields for a reviewer to audit the engineering probe without
  rerunning the first-principles runner manually.

Required command:

```bash
.venv312/bin/python -m dpf.cli.main experimental-segmented-whole-shot \
  --deck-preset pf1000_scholz_2001_24rod_full_energy \
  --segment-steps 2 \
  --explicit-total-steps 6 \
  --run-dir /private/tmp/dpf_sprint10_pf1000_full_energy_probe \
  --no-verify-restart-equivalence \
  --output /private/tmp/dpf_sprint10_pf1000_full_energy_probe/manifest.json
```

Required tests:

- Manifest has `can_support_first_principles_acceptance=false`.
- Manifest has `accepted_runtime_claim=false`.
- Manifest selected source scope is PF-1000 Scholz full-energy.
- Manifest architecture scope is LLNL-like hybrid-PIC.
- Manifest same-scope declared scope is `pf1000_full_energy_27_to_40_kv`.
- Manifest power-port terms II/IV/V/VI are blocked on the Sigma-p reviewed face
  set.
- Manifest geometry blocker summary lists all five blocked fields.

### SS10-4 - Encode Imported-PIC Startup as Context-Only Policy

Required work:

- Remove half-accepted imported-PIC mode semantics.
- Imported-PIC payloads may be preserved as engineering context only.
- No imported-PIC payload may satisfy `mode_is_accepted` unless a new explicitly
  reviewed same-scope acceptance policy is created in a future sprint.

Required tests:

```bash
.venv312/bin/python -m pytest tests/test_startup_breakdown_audit.py \
  tests/test_first_principles_startup_bvp.py -q
```

Add a regression test that simulates a future accepting typed startup packet and
proves imported-PIC mode still cannot promote.

### SS10-5 - Traceability and Source-Truth Artifact Truthfulness

Required work:

- Run the source-truth exhaustion check.
- Run module-source vetting.
- Run RTM/SRS traceability export/checks.
- Commit changed artifacts, or explicitly document no-op outputs.

Required commands:

```bash
.venv312/bin/python scripts/verify_first_principles_source_truth_exhaustion.py --strict --check --date 2026_05_20
.venv312/bin/python scripts/verify_first_principles_module_source_vetting.py --strict --check --date 2026_05_20
.venv312/bin/python -m pytest tests/test_srs_traceability_export.py -q
```

If any command must use a newer date slug, update this document in the handoff
with the exact slug and output.

### SS10-6 - Repair Server Readiness Scope Drift

Required work:

- Fix `tests/test_server_readiness.py` expectations or server payload behavior
  so API readiness surfaces match the current fail-closed first-principles
  contract.
- Preserve source-scope separation: API payloads must not reintroduce Akel 16 kV
  source labels for the PF-1000 full-energy demonstrator.

Required test:

```bash
.venv312/bin/python -m pytest tests/test_server_readiness.py -q
```

### SS10-7 - Add Report-Only Acceptance-Gate Dry Run

Required work:

- Add a report-only command or test fixture that runs the numerical fidelity,
  same-scope comparator, UQ, certificate, geometry, startup, power-port, and
  neutron gates against the six-step PF-1000 full-energy engineering probe.
- The output must be a fail-closed ledger. It must not promote acceptance.
- Each failed gate must name the missing source packet, runtime field, or
  numerical check.

Required output:

- A JSON artifact under `results/` only if it passes the first-principles artifact
  linter.
- A markdown summary under `docs/` listing every blocked gate and the next
  concrete source/code action.

### SS10-8 - Periodic Audit and Worktree Policy

Required work:

- Keep the 146-line external-churn exception explicit.
- Do not call the physical worktree clean unless `git status --short` is empty.
- If another dirty line appears, the audit must fail until the team classifies
  or fixes it.

Required command:

```bash
.venv312/bin/python scripts/run_codex_periodic_audit.py --timeout-seconds 900
```

The report must quote the `git_status_clean` note, including exception count.

## 6. Super-Sprint 10 Definition of Done

Super-Sprint 10 is complete only when all of the following are true:

- Findings A1-A7 in this audit are closed by code/docs/tests.
- No new acceptance flag is promoted.
- Runtime PF-1000 full-energy packets no longer expose LLNL-like evidence under
  same-scope keys.
- Runtime conductor-mask telemetry and segmented manifest expose the five blocked
  PF-1000 geometry fields.
- Hollow-anode runtime telemetry matches the selected deck.
- The six-step segmented manifest contains the required scope, same-scope,
  power-port, and geometry summaries.
- Imported-PIC startup is explicitly context-only at the mode policy level.
- `tests/test_server_readiness.py` passes or is documented as a deliberate
  out-of-scope failure with exact baseline reproduction.
- Periodic audit passes with the external-churn exception note.
- `CodexFindings.md` and `CortexFindings.md` are updated with exact command
  outputs and no stale completion claims.

## 7. Work After Super-Sprint 10

Do not begin these until the SS10 correction gate is closed:

- Same-scope PF-1000 full-energy 3-D validation packet design.
- Four absent PF-1000 geometry dimensions and IPPLM facility request.
- Braginskii Table-2 second-reader review cells.
- Sigma-p reviewed face-set construction for power-port terms II/IV/V/VI.
- Startup BVP source packet for D2 breakdown/flashover/liftoff handoff.
- Numerical-fidelity convergence/restart/backend-parity acceptance suite.
- Comparator/UQ/certificate run for PF-1000 full-energy observables.
