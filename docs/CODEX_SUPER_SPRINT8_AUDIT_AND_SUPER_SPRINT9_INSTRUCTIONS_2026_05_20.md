# Codex Super-Sprint 8 Audit and Super-Sprint 9 Instructions

**Date:** 2026-05-20 local / 2026-05-21 UTC  
**Branch:** `codex/corpus`  
**Audited HEAD:** `814ab108e79b4249db926d8585398977cd128161`  
**Verdict:** accept the Sprint 8 candidate packets with corrections. Do not call Sprint 8 internally complete for the PF-1000 full-energy runtime path until the P0 corrections below are landed.

## Classification

This audit is not validation. No first-principles acceptance claim is approved here.

All runtime and source-packet outputs audited in this pass must remain:

- `accepted_runtime_claim=false`
- `can_support_first_principles_acceptance=false`
- engineering-candidate only

## What Passed

The team's broad status is mostly accurate:

- Commits exist as reported: `bd5be3a`, `b270cb5`, `814ab10`.
- Focused Sprint 8 tests passed locally: `202 passed`.
- `ruff check src tests` passed.
- Braginskii Z=1 PlasmaPy cross-check passed with maximum relative difference `0.3618%`; PlasmaPy remains cross-check only, not source authority.
- Full periodic audit passed 9/10 gates at HEAD `814ab10`; the only failed gate was `git_status_clean`, matching the reported PDF symlink type-change caveat.
- The known startup-breakdown audit failure still reproduces and is outside the periodic audit glob.

## Findings

### P0-1: Selected Scope Is Not Propagated Into the Runtime Validation Scope

The Sprint 8 scope lock says the canonical selected scope is:

```text
pf1000_full_energy_27_to_40_kv
```

and explicitly requires that value to be used as the same-scope packet `declared_scope`
(`docs/SPRINT8_WS2_RUNTIME_DEMONSTRATOR_SCOPE_LOCK_2026_05_20.md:18-26`,
`:150-157`).

The runtime does not do that today. The PF-1000 24-rod deck documents the selected scope
(`src/dpf/first_principles/deck.py:976-988`) and sets `startup.source_scope` to the
selected label (`src/dpf/first_principles/deck.py:1161-1170`), but the package-deck
conversion does not carry an explicit `validation_scope`. Instead,
`_validation_scope_from_package_deck()` returns the deck id whenever validation targets
exist (`src/dpf/first_principles/runner.py:3213-3216`).

Observed effect from:

```bash
.venv312/bin/python -m dpf.cli.main first-principles-3d --deck-preset pf1000_scholz_2001_24rod_full_energy
```

The emitted `engineering_current_waveform_comparison.declared_scope` was:

```text
pf1000_scholz_2001_24rod_full_energy_27kv_3p5torr_engineering_candidate
```

not `pf1000_full_energy_27_to_40_kv`.

**Required correction:** add an explicit validation-scope field/path for package-native decks and require the PF-1000 full-energy preset to emit `pf1000_full_energy_27_to_40_kv` into:

- `FirstPrinciples3DDeck.validation_scope`
- `same_scope_source.declared_scope`
- `engineering_current_waveform_comparison.declared_scope`
- segmented whole-shot manifests
- certificate/comparator/numerical-fidelity upstream packets

Add a regression test that fails if any PF-1000 full-energy preset manifest emits the deck id as the declared validation scope.

### P0-2: The PF-1000 Runtime Validation Packet Still Reports LLNL-Like Source Scope

The package-native runner still constructs:

```python
geometry = HybridPICSourceGeometry()
```

unconditionally (`src/dpf/first_principles/runner.py:811`). That object has
`source_scope="llnl_like_180ka_axisymmetric_hybrid_pic"`
(`src/dpf/fields/source_geometry.py:39-46`). The runner then uses that stale geometry
scope in the top-level candidate packet and telemetry (`src/dpf/first_principles/runner.py:1173-1196`).

Observed effect from the same PF-1000 preset run:

```text
validation_packet.source_scope = llnl_like_180ka_axisymmetric_hybrid_pic
```

This is not just wording. It means the selected PF-1000 full-energy source scope and the top-level validation packet source scope disagree at runtime.

**Required correction:** split architecture-source evidence from selected-machine source evidence. The hybrid-PIC paper can remain an architecture source, but the runtime candidate packet for the PF-1000 full-energy preset must carry PF-1000 source scope and deck source references derived from the selected deck. Add tests that the PF-1000 preset never emits `llnl_like_180ka_axisymmetric_hybrid_pic` as `validation_packet.source_scope`.

### P1-1: Same-Scope Akel Helper Still Matches Any PF-1000 Scope

`_looks_like_pf1000_akel_scope()` currently returns true for any `pf1000`, `pf-1000`, or `akel`
string (`src/dpf/first_principles/same_scope.py:464-469`). Therefore the full-energy scope
receives `PF1000_AKEL_TEXT_SUPPORTED_CHANNELS` and an Akel-named acceptance-gate label
(`src/dpf/first_principles/same_scope.py:209-212`, `:266-270`).

Observed direct check:

```python
build_same_scope_source_packet(
    declared_scope="pf1000_full_energy_27_to_40_kv",
    device_name="PF-1000 full energy",
)["text_supported_reference_channels"]
```

returned Akel text-supported channels such as `pinch_geometry_lee_output` and
`timing_uncertainty_text`.

The channels remain `excluded_not_validated`, so this is not a false acceptance. It is still a scope-classification defect.

**Required correction:** make the Akel helper exact: require Akel/16 kV/shot 12581 markers. Add a separate full-energy reference-channel set only after KR target extraction supplies selected-scope records. For now, full-energy text-supported channels should be empty or selected-scope-only.

### P1-2: Bennett Startup Extraction Is Cataloged, Not Runtime-Consumed

Sprint 8 WS4 correctly creates a Bennett target-extraction module and keeps CH03/04/07/08
wrong-scope for the selected full-energy demonstrator. It does not wire Bennett into the
startup runtime registry. The Sprint 8 test suite explicitly enforces that:

- `tests/test_sprint8_ws4_bennett_startup.py:542-570` proves Bennett candidates cannot lift startup BVP acceptance.
- `tests/test_sprint8_ws4_bennett_startup.py:656-671` asserts `startup_bvp.py` source refs are not modified by WS4.

That is acceptable as fail-closed cataloging, but it means any report language saying the runtime startup registry "consumes" Bennett targets is overstated.

**Required correction:** apply a lead-owned startup runtime delta. The startup packet should expose Bennett CH03/04/07/08 as source-backed, wrong-scope candidate context, while still keeping the full-energy startup BVP blocked. Do not let these channels satisfy same-scope startup acceptance.

### P1-3: Periodic Audit Is Not Fully Green While the Worktree Has PDF Type Changes

The current periodic audit result was:

```text
cycle 1: FAIL head=814ab108e79b4249db926d8585398977cd128161 log=/private/tmp/dpf-unified-audit-logs/20260521T030841Z
```

The summary shows 9/10 gates pass and only `git_status_clean` fails. This matches the team's caveat. It still matters: an audit cannot be reported as 10/10 green at this HEAD until the worktree type-change policy is resolved.

**Required correction:** either normalize the PDF storage reorg in a dedicated commit, move those paths outside the repository, or add a documented audit exception that `git_status_clean` can classify as known external storage churn without marking the whole audit green. Do not silently call it 10/10.

### P2-1: Pre-Existing Startup Acceptance Test Failure Is Real Debt

The failure reproduces:

```bash
.venv312/bin/python -m pytest tests/test_startup_breakdown_audit.py::test_reviewed_imported_pic_startup_payload_can_close_packet -q
```

Observed result:

```text
AssertionError: assert 'blocked_startup_packet_not_available' == 'accepted_startup_bvp_packet'
```

`git diff 35bb1a9..HEAD -- src/dpf/first_principles/startup_bvp.py tests/test_startup_breakdown_audit.py` is empty, so the team is correct that this predates Sprint 8. It is still a blocker for any future imported-PIC startup handoff acceptance path.

**Required correction:** decide whether imported reviewed PIC startup payloads are allowed to close a startup BVP packet. If yes, implement the typed payload gate with full source/review/unit/hash requirements. If no, retire or rewrite the failing test so the gate cannot be misread.

## Super-Sprint 9 Instructions

### Sprint Goal

Make the PF-1000 full-energy package-native 3-D runtime path internally coherent as an engineering experiment surface. This sprint is not validation and must not promote any acceptance flag.

### Non-Negotiable Guardrails

1. `KnowledgeReference/` target-extracted files are the only scientific authority.
2. Raw PDFs, web pages, PlasmaPy, and external databases are acquisition or cross-check lanes only until KR-ingested and reviewed.
3. No reduced Lee/snowplow output may drive the package-native first-principles runtime.
4. No workstream may set `accepted_runtime_claim=true` or `can_support_first_principles_acceptance=true`.
5. Every scope transfer must have an explicit transfer-rule packet or remain wrong-scope/context-only.

### WS9-0: Worktree and Audit Gate Cleanup

Deliverables:

- A documented decision on the 145 PDF symlink type changes.
- A periodic audit run that either passes 10/10 or reports 9/10 with an explicit approved exception.
- No unrelated file churn in the sprint commits.

Acceptance tests:

```bash
.venv312/bin/python scripts/run_codex_periodic_audit.py --timeout-seconds 900
git status --short
```

### WS9-1: Runtime Scope Propagation

Deliverables:

- Add an explicit validation-scope field to the package-native deck path.
- PF-1000 full-energy preset must emit `pf1000_full_energy_27_to_40_kv` as the declared validation scope everywhere.
- No packet should substitute the deck id for the selected validation scope.

Required tests:

- `FirstPrinciples3DDeck.from_deck(pf1000_scholz_2001_24rod_full_energy_deck()).validation_scope == SELECTED_SCOPE_LABEL`
- `first-principles-3d --deck-preset pf1000_scholz_2001_24rod_full_energy` emits `same_scope_source.declared_scope == SELECTED_SCOPE_LABEL`
- segmented whole-shot manifest emits `deck.validation_scope == SELECTED_SCOPE_LABEL`

### WS9-2: Runtime Source Evidence Separation

Deliverables:

- Separate `architecture_source` from `selected_machine_source_scope`.
- The hybrid-PIC source remains architecture evidence only.
- PF-1000 preset validation packets must not report `llnl_like_180ka_axisymmetric_hybrid_pic` as the selected source scope.

Required tests:

- PF-1000 preset top-level `validation_packet.source_scope` equals selected PF-1000 source scope.
- Hybrid-PIC architecture source is still present under an explicitly named architecture/equation-method evidence key.
- Source-geometry evidence lists the PF-1000 KR geometry paths when using the PF-1000 preset.

### WS9-3: Same-Scope Full-Energy Packet Repair

Deliverables:

- Rename/refactor `_looks_like_pf1000_akel_scope()` to match only Akel 16 kV / shot 12581.
- Add selected-scope full-energy reference-channel handling only from KR target-extracted sources.
- Remove Akel-named text-supported channels and gate labels from full-energy packets.

Required tests:

- Full-energy scope receives no Akel reference channels.
- Akel 16 kV scope continues to receive Akel text-supported channels as non-acceptance evidence.
- `check_scope_consistency()` still rejects mixed full-energy + Akel packets.

### WS9-4: Bennett Startup Runtime Context

Deliverables:

- Import the Sprint 8 Bennett packet into startup runtime telemetry as wrong-scope candidate context.
- Preserve `blocked_wrong_scope` for the selected full-energy demonstrator.
- Do not modify startup acceptance flags.

Required tests:

- `startup_bvp` exposes Bennett CH03/04/07/08 as wrong-scope candidate context when requested.
- Passing those candidate channels cannot produce `accepted_startup_bvp_packet`.
- Startup BVP remains blocked until a selected-scope startup packet exists.

### WS9-5: Startup Imported-PIC Decision

Deliverables:

- Resolve `test_reviewed_imported_pic_startup_payload_can_close_packet`.
- If accepted imported PIC startup handoff is allowed, implement strict source/review/unit/hash checks.
- If not allowed, rewrite the test and docs to state that imported PIC payloads are context only.

Required tests:

```bash
.venv312/bin/python -m pytest tests/test_startup_breakdown_audit.py tests/test_first_principles_startup_bvp.py -q
```

### WS9-6: PF-1000 Geometry Mask Runtime Integrity

Deliverables:

- Keep missing fields blocked: hollow bore length, insulator wall thickness, backplate radial extent, backplate axial thickness, same-scope reviewed mask.
- Ensure PF-1000 conductor-mask telemetry references the selected deck geometry, not old LLNL geometry.
- Add an explicit mesh-resolution warning when cathode rod diameter is under-resolved.

Required tests:

- PF-1000 24-rod preset emits the five blocked fields.
- PF-1000 conductor mask reports PF-1000 source refs and selected scope.
- Under-resolved rods cannot support geometry acceptance.

### WS9-7: Engineering Runtime Probe

Deliverables:

- Run a short segmented PF-1000 full-energy experiment after WS9-1/WS9-2 fixes.
- Record manifest, duration status, limiter counts, source scope, same-scope scope, and power-port blockers.
- Do not call it validation.

Minimum command:

```bash
.venv312/bin/python -m dpf.cli.main experimental-segmented-whole-shot \
  --deck-preset pf1000_scholz_2001_24rod_full_energy \
  --segment-steps 2 \
  --explicit-total-steps 6 \
  --run-dir results/sprint9_pf1000_full_energy_probe \
  --no-verify-restart-equivalence \
  --output results/sprint9_pf1000_full_energy_probe/manifest.json
```

Required manifest checks:

- `can_support_first_principles_acceptance=false`
- selected scope appears consistently
- old LLNL-like source scope is not used as selected-machine scope
- `duration_request_satisfied` or `horizon_complete` is explicit
- power-port Sigma-p blockers remain explicit

### WS9-8: Handoff and Audit Packet

Deliverables:

- Update `CodexFindings.md` and `CortexFindings.md`.
- Update RTM/source-truth/module-vetting artifacts if code changed.
- Produce a Sprint 9 completion memo with exact commands and outputs.

Required final command set:

```bash
.venv312/bin/python -m pytest tests/test_runtime_demonstrator_scope.py tests/test_first_principles_channel_state_contract.py tests/test_ws7_3d_runtime_ratchet.py -q
.venv312/bin/python -m pytest tests/test_startup_breakdown_audit.py tests/test_first_principles_startup_bvp.py -q
.venv312/bin/python -m ruff check src tests
.venv312/bin/python scripts/run_codex_periodic_audit.py --timeout-seconds 900
```

## Audit Commands Run by Codex

```bash
python3 /Users/anthonyzamora/.codex/skills/dpf-validation/scripts/dpf_skill_preflight.py /Users/anthonyzamora/dpf-unified
git show --stat --oneline --no-renames bd5be3a b270cb5 814ab10
.venv312/bin/python -m pytest tests/test_runtime_demonstrator_scope.py tests/test_sprint8_ws4_bennett_startup.py tests/test_sprint8_braginskii_z1_transport.py tests/test_first_principles_channel_state_contract.py tests/test_ws7_3d_runtime_ratchet.py tests/test_first_principles_power_port.py -q
.venv312/bin/python -m ruff check src tests
.venv312/bin/python -m pytest tests/test_startup_breakdown_audit.py::test_reviewed_imported_pic_startup_payload_can_close_packet -q
.venv312/bin/python scripts/plasmapy_braginskii_z1_crosscheck.py --strict
.venv312/bin/python scripts/run_codex_periodic_audit.py --timeout-seconds 900
```

## Bottom Line

Sprint 8 delivered useful source-backed candidate packets and better fail-closed gates. It did not finish the PF-1000 full-energy runtime scope integration. Super-Sprint 9 must first repair scope/source propagation, then wire candidate startup context and run the short PF-1000 engineering probe with coherent selected-scope telemetry.
