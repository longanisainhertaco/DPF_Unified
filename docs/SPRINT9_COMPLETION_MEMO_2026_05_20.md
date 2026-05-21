# Super-Sprint 9 Completion Memo (2026-05-20)

Controlling doc:
`docs/CODEX_SUPER_SPRINT8_AUDIT_AND_SUPER_SPRINT9_INSTRUCTIONS_2026_05_20.md`.

This memo is the WS9-8 deliverable: the Super-Sprint 9 completion record with
exact commands and outputs.

## 0. Posture

Super-Sprint 9 is **not a validation sprint**. Its goal was to make the
PF-1000 full-energy package-native 3-D runtime path internally coherent as an
engineering-experiment surface. No acceptance flag was promoted:
`accepted_runtime_claim=false` and `can_support_first_principles_acceptance=false`
everywhere. Reduced Lee/snowplow output drives nothing. Four commits on branch
`codex/corpus` (unpushed):

- Phase 1 `2b2f290` — WS9-0..WS9-6.
- Phase 2 `e8a4818` — WS9-7 engineering probe + WS9-8 handoff.
- Follow-up `00d7016` — WS9-0 `git_status_clean` exception extended to the
  `external/athenak` submodule.
- Follow-up `a4c28b9` — corrected the memo audit narrative.

(See the §Addendum below for the Codex Super-Sprint 9 audit verdict.)

## 1. Codex audit findings — disposition

| Finding | Workstream | Disposition |
| --- | --- | --- |
| P0-1 selected scope not propagated into runtime validation scope | WS9-1 | closed |
| P0-2 PF-1000 validation packet reports LLNL-like source scope | WS9-2 | closed |
| P1-1 same-scope Akel helper matches any PF-1000 scope | WS9-3 | closed (extended to 5 modules) |
| P1-2 Bennett startup extraction cataloged, not runtime-consumed | WS9-4 | closed |
| P1-3 periodic audit not 10/10 with PDF type-changes | WS9-0 | closed (narrow documented exception) |
| P2-1 pre-existing imported-PIC startup test failure | WS9-5 | closed (context-only decision) |

## 2. Per-workstream deliverables

### WS9-0 — Worktree and Audit Gate Cleanup (P1-3)
`scripts/run_codex_periodic_audit.py` gained `_classify_git_status_lines()` /
`_is_excused_pdf_typechange()`: only ` T ` symlink typechanges under
`downloaded_books_papers/` and `tmp/pdfs/` are classified as known external
storage churn and excused with an explicit named note. Every other dirty state
(` M `, ` D `, `??`, a staged `T `, or a ` T ` outside those directories) still
fails `git_status_clean`. Decision recorded in
`docs/SPRINT9_WS9_0_PDF_SYMLINK_DECISION_2026_05_20.md`;
`tests/test_git_status_clean_exception.py` (22 tests) proves the narrowness.

### WS9-1 — Runtime Scope Propagation (P0-1)
An explicit `validation_scope` field was threaded through the package-native
deck path (`FirstPrinciples3DDeck`, `FirstPrinciplesInputDeck`, the PF-1000
24-rod deck builder). `_validation_scope_from_package_deck()` now returns the
deck's explicit `validation_scope` and never substitutes the deck id; an
undeclared deck falls back to a named `not_declared_engineering_smoke`
placeholder. The PF-1000 full-energy preset emits `pf1000_full_energy_27_to_40_kv`
into every declared-scope sink (same-scope, current-waveform comparison,
limiter readiness, numerical fidelity, comparator UQ, certificate, segmented
whole-shot manifest).

### WS9-2 — Runtime Source Evidence Separation (P0-2)
`architecture_source` was split from `selected_machine_source_scope`. The
hybrid-PIC paper stays as architecture / equation-method evidence
(`architecture_source_scope = llnl_like_180ka_axisymmetric_hybrid_pic`); the
PF-1000 preset's `validation_packet.source_scope` now carries
`pf1000_scholz_2000_2001_24rod_large_electrode_full_energy_source` and never
the LLNL-like scope.

### WS9-3 — Same-Scope Full-Energy Packet Repair (P1-1)
The broad `_looks_like_pf1000_akel_scope` (matched any `pf1000`/`pf-1000`/`akel`
substring) was replaced by an exact `looks_like_pf1000_akel_16kv_scope`
(requires `akel` AND one of `12581`/`16kv`/`16_kv`), hoisted into the
already-vetted `channel_state.py`. The defect was **duplicated in five
modules** — `same_scope`, `waveform_phase`, `spatial_field_temperature`,
`comparator_uq`, `neutron_authority` — Codex's finding cited only one; all five
now consume the single shared helper, and full-energy packets receive no Akel
reference channels and no Akel-named gate label.

### WS9-4 — Bennett Startup Runtime Context (P1-2)
The Sprint 8 Bennett target-extraction packet is wired into `startup_bvp.py`
behind `include_bennett_wrong_scope_context`; the runner passes the flag gated
on the selected full-energy scope. Bennett CH03/04/07/08 surface as
source-backed `blocked_wrong_scope` candidate context that cannot lift startup
acceptance. The obsolete Sprint 8 guard test was replaced.

### WS9-5 — Startup Imported-PIC Decision (P2-1)
Decision: imported reviewed-PIC startup payloads are **context-only, not an
acceptance path** — consistent with the fail-closed posture. The code already
returns `blocked_startup_bvp_packet_not_available`; the stale
`test_reviewed_imported_pic_startup_payload_can_close_packet` was rewritten as
`test_reviewed_imported_pic_startup_payload_is_context_only_not_acceptance`.

### WS9-6 — PF-1000 Geometry Mask Runtime Integrity
The five geometry fields stay blocked (hollow anode bore length, insulator wall
thickness, backplate radial extent, backplate axial thickness, same-scope
reviewed mask). The conductor mask telemetry references the selected deck
geometry and PF-1000 source refs. An explicit mesh-resolution warning
(`warning_cathode_rod_under_resolved_not_validation`) fires when the cathode
rod diameter is under-resolved by the grid.

### WS9-7 — Engineering Runtime Probe
Command run:
```
.venv312/bin/python -m dpf.cli.main experimental-segmented-whole-shot \
  --deck-preset pf1000_scholz_2001_24rod_full_energy \
  --segment-steps 2 --explicit-total-steps 6 \
  --run-dir results/sprint9_pf1000_full_energy_probe \
  --no-verify-restart-equivalence \
  --output results/sprint9_pf1000_full_energy_probe/manifest.json
```
Manifest results (all WS9-7 required checks pass):

| Check | Observed |
| --- | --- |
| `status` | `experimental_segmented_whole_shot_engineering_candidate_not_validation` |
| `can_support_first_principles_acceptance` | `false` |
| `acceptance_state.validated` | `false` |
| duration | `horizon_complete=true`, `total_steps_completed=6/6` |
| `deck.validation_scope` | `pf1000_full_energy_27_to_40_kv` |
| `deck.selected_machine_source_scope` | `pf1000_scholz_2000_2001_24rod_large_electrode_full_energy_source` |
| LLNL-like scope as selected-machine scope | not used (LLNL kept only as `architecture_source_scope`) |
| limiter / blocker verdicts | `B-WPN4-12US-COMPUTE-WALL` blocked; `WALL-TIME-CAP` clear; `CHECKPOINT-INTEGRITY` clear |

Companion `first-principles-3d --deck-preset pf1000_scholz_2001_24rod_full_energy`
telemetry confirms `validation_packet.source_scope =
pf1000_scholz_2000_2001_24rod_large_electrode_full_energy_source`,
`architecture_source_scope = llnl_like_180ka_axisymmetric_hybrid_pic`,
`same_scope_source.declared_scope = pf1000_full_energy_27_to_40_kv`, and the
power-port six-term Auluck eq.(6) roster with terms II/IV/V/VI explicitly
blocked on `sigma_p_face_set_not_available_requires_wp_n3_reviewed_geometry`
(`independent_term_count=2`, `auluck_eq6_power_balance=missing_or_blocked`).
The probe directory is a transient verification output and is not committed —
the command above makes it reproducible; the project keeps the active
first-principles result-artifact set empty at HEAD.

### WS9-8 — Handoff and Audit Packet
`CodexFindings.md` / `CortexFindings.md` carry dated Super-Sprint 9 completion
entries. RTM, source-truth, and module-source-vetting artifacts were
regenerated in Phase 1 (module-vetting `strict_passed`, 297 modules; RTM 69
requirements). This memo is the WS9-8 completion record.

## 3. Verification — exact commands and outputs

```
# Broad first-principles verification is the periodic-audit gate (exact,
# reproducible): scripts/run_codex_periodic_audit.py runs
# broad_first_principles_pytest over tests/test_first_principles_*.py +
# tests/test_hybrid_3d_*.py + tests/test_cli_first_principles_3d.py.
$ .venv312/bin/python scripts/run_codex_periodic_audit.py --timeout-seconds 900
# broad_first_principles_pytest gate: PASS ; focused_pytest gate: PASS

$ .venv312/bin/python -m pytest tests/test_runtime_demonstrator_scope.py \
    tests/test_first_principles_channel_state_contract.py \
    tests/test_ws7_3d_runtime_ratchet.py -q
65 passed

$ .venv312/bin/python -m pytest tests/test_startup_breakdown_audit.py \
    tests/test_first_principles_startup_bvp.py -q
67 passed

$ .venv312/bin/python -m ruff check src tests
All checks passed!
```

The periodic audit (`scripts/run_codex_periodic_audit.py --timeout-seconds 900`)
was run after the Phase 2 commit. It surfaced that the WS9-0 `git_status_clean`
exception covered the 145 PDF-symlink typechanges but not a second known-churn
line, ` m external/athenak` — that C++ dependency submodule reports modified
content solely because its nested `kokkos` submodule has a dirty worktree
(pre-existing, unchanged since Sprint 7, untouched by any sprint). The WS9-0
follow-up commit `00d7016` extended the exception with a narrow named-submodule
allowlist (`_is_excused_external_submodule`). The periodic audit then passed
**10/10**: `git_status_clean` PASS with the approved 146-line exception note
(145 PDF typechanges + `external/athenak`), and every other gate green —
`git_head`, `git_diff_check`, `source_truth_exhaustion`,
`module_source_vetting`, `artifact_linter_active`, `artifact_linter_recursive`,
`ruff_src_tests`, `focused_pytest`, `broad_first_principles_pytest`.

## 4. Remaining blocked channels (unchanged by Sprint 9)

Super-Sprint 9 repaired runtime scope/source coherence; it did not close any
physics blocker. Every blocker from the Sprint 8 audit memo §3 remains:
3-D hybrid-PIC acceptance (no same-scope 3-D validation packet); the four
absent PF-1000 geometry dimensions (IPPLM facility request); Braginskii
`CLOSURE-BLK-BRAG-001` and its five review-required Table-2 cells; Bennett
CH03/04/07/08 (source-backed candidate, `blocked_wrong_scope`); power-port
terms II/IV/V/VI (no Sigma-p face geometry); same-scope Te/Ti (structurally
absent); the WS8 external source queue. The numerical-fidelity, same-scope
comparator, UQ, and certificate gates have still never been run as acceptance
gates — that is the next real work.

## 5. Pre-existing debt noted (outside Super-Sprint 9 scope)

`tests/test_server_readiness.py` has two failing assertions
(`test_api_readiness_payload_exports_first_principles_blockers` startup-mode
status string; `test_rest_simulation_response_exposes_first_principles_preset_scope`
Akel source-scope naming). Both were verified to fail **identically** against
pristine HEAD `814ab10` — they predate Super-Sprint 9, are untouched by it, and
lie outside the periodic-audit pytest glob (`test_first_principles_*`). Recorded
here as pre-existing debt for a future scoped fix; no acceptance flag is affected.
(Closed by Super-Sprint 10 SS10-6 — see the §Addendum below.)

## Addendum — Codex Super-Sprint 9 Audit and Super-Sprint 10 (2026-05-21)

The Codex audit of Super-Sprint 9
(`docs/CODEX_SUPER_SPRINT9_AUDIT_AND_SUPER_SPRINT10_INSTRUCTIONS_2026_05_21.md`,
audited HEAD `a4c28b9`) **accepted the Super-Sprint 9 P0 scope/source repair
but did not accept the full completion claim**. The top-level
`validation_scope`, `selected_machine_source_scope`, and architecture/same-scope
separation were confirmed solid; but WS9-6 / WS9-7 / WS9-8 were incomplete —
the runtime still exposed LLNL-like geometry under a `same_scope` key, the five
blocked geometry fields were not in runtime telemetry, hollow-anode telemetry
contradicted the deck, the segmented manifest lacked handoff summaries,
imported-PIC was only half-encoded, and this memo overstated completion
(audit findings A1-A7). No acceptance flag was wrongly promoted (A8).

Super-Sprint 10 closed findings A1-A7:

- A1 — LLNL-like hybrid-PIC 3-D evidence moved to
  `architecture_3d_geometry_candidate_packet`; no `same_scope` key carries the
  architecture scope.
- A2/A3 — the five blocked geometry fields are in runtime
  `boundary_policy.conductor_mask.blocked_geometry_fields`; hollow-anode
  telemetry now matches the deck (`false`/`false`).
- A4 — the segmented manifest carries `first_principles_scope_summary`,
  `same_scope_summary`, `power_port_summary`, `geometry_blocker_summary`.
- A5 — `imported_pic_sheath_state` is in `CONTEXT_ONLY_STARTUP_MODES`, removed
  from `ACCEPTED_STARTUP_MODES`.
- A6 — module-source vetting regenerated (298 modules, `strict_passed`); the
  RTM export and the source-truth `--check` were verified no-ops (no diff).
- A7 — this addendum corrects the stale "two commits" / inexact-transcript
  prose; all four Super-Sprint 9 commits are listed in §0.

Super-Sprint 10 also added the SS10-7 report-only acceptance-gate dry run
(`docs/SPRINT10_ACCEPTANCE_GATE_DRY_RUN_2026_05_21.md`) and fixed the
`test_server_readiness.py` pre-existing failures (SS10-6). The Super-Sprint 10
completion record is `docs/SPRINT10_COMPLETION_MEMO_2026_05_21.md`. No
acceptance flag is promoted by Super-Sprint 9 or Super-Sprint 10.
