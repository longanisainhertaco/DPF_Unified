# Codex Super-Sprint 12 Phase P0 Audit and P1 Authorization

Date: 2026-05-21
Branch audited: `codex/corpus`
Audited HEAD: `df84751d7c04d5c200e29788b422e8dc8fcef05f`
P0 base: `0dcf77826ea7df321d57479ac3b4dff213d96811`

## 1. Verdict

Super-Sprint 12 Phase P0 is **accepted for P1 authorization**.

P0 closed the four Codex SS11 audit blockers:

- SS11-A1: direct runtime-deck imported-PIC whole-shot flag clamp.
- SS11-A2: exact-pair server readiness scope resolution.
- SS11-A3: same-scope packet namespace split.
- SS11-A4: structured active-results artifact linter.

No physics channel was promoted. No first-principles acceptance flag was
promoted. The PF-1000 full-energy dry-run ledger remains report-only with all
eight acceptance gates blocked.

P1 may start, with one required P1-0 cleanup noted in Section 4.

## 2. Audit Findings

### P0-A1 - Direct Imported-PIC Runtime-Deck Clamp

Status: Closed

`FirstPrinciples3DDeck.__post_init__` now forces
`startup_can_support_whole_shot_acceptance=False` whenever
`startup_mode` is in `CONTEXT_ONLY_STARTUP_MODES`.

Adversarial direct-constructor probe:

```text
deck_field_startup_can_support_whole_shot_acceptance=False
packet_status=blocked_startup_bvp_packet_not_available
packet_can_support_whole_shot_acceptance=False
packet_can_support_first_principles_acceptance=False
```

This closes the raw direct-construction gap from SS11-A1.

### P0-A2 - Exact-Pair Server Readiness

Status: Closed

`_resolve_runtime_deck_scope()` now resolves only exact ordered
`(validation_scope, source_scope)` pairs:

- Akel: `pf1000_16kv_2021_akel` +
  `pf1000_16kv_2021_akel_shot12581`
- PF-1000 full-energy: `pf1000_full_energy_27_to_40_kv` +
  `pf1000_scholz_2000_2001_24rod_large_electrode_full_energy_source`

Adversarial readiness probes:

```text
full_energy_plus_akel_startup_label -> runtime_deck_id=not_run, scope_match=False
full_energy_plus_not_declared -> runtime_deck_id=not_run, scope_match=False
unknown_validation_plus_full_energy_source -> runtime_deck_id=not_run, scope_match=False
akel_validation_plus_full_energy_source -> runtime_deck_id=not_run, scope_match=False
exact_akel -> Akel deck, scope_match=True
exact_full_energy -> full-energy deck, scope_match=True
```

This closes the partial-pair label-mixing surface from SS11-A2.

### P0-A3 - Same-Scope Namespace Split

Status: Closed

`other_scope_source_groups` and cross-scope policy material are no longer inside
`same_scope_source`. The runner emits the relocated material under sibling
`cross_scope_context_sources` fields.

Independent runtime scan over PF-1000 full-energy telemetry and manifest:

```text
same_scope_source_namespace_issue_count=0
has_cross_scope_context_sources=True True
```

This closes the same-scope namespace blocker from SS11-A3.

### P0-A4 - Structured Active Results Linter

Status: Closed for P0

`scripts/verify_active_results_artifact_hygiene.py` now parses active
non-archive result JSON and checks key chains, including dict key names. It
forbids:

- hybrid-PIC source slugs under any `same_scope` key chain;
- `other_scope` / `wrong_scope` tokens under any `same_scope_source` key chain.

Audit output:

```json
{
  "active_hit_count": 0,
  "clean": true,
  "issues": []
}
```

P0 acceptance note: the linter intentionally permits hybrid-PIC source slugs in
ordinary non-`same_scope` source fields. That behavior matches the current live
artifact surface and preserves the important safety property: no architecture or
cross-scope material is allowed under same-scope evidence keys.

The completion memo wording says architecture or cross-scope evidence is
"permitted only under approved context keys." The implementation is narrower:
it enforces "not under same-scope keys" rather than "only under context keys."
This is not a P0 blocker, but P1 must reconcile the wording/test contract as a
P1-0 cleanup.

## 3. Verification

Commands run from `/Users/anthonyzamora/dpf-unified`.

Focused P0 pytest:

```text
.venv312/bin/python -m pytest \
  tests/test_ss10_imported_pic_context_only_policy.py \
  tests/test_server_readiness.py \
  tests/test_ws9_runner_scope_source_geometry.py \
  tests/test_results_artifact_hygiene.py \
  tests/test_first_principles_acceptance_gate_dry_run.py -q
```

Result: `84 passed in 30.51s`.

Ruff:

```text
ruff check src tests
```

Result: `All checks passed!`.

Source-truth exhaustion:

```text
.venv312/bin/python scripts/verify_first_principles_source_truth_exhaustion.py --strict --check --date 2026_05_21
```

Result: `exhausted=true`, `open_issue_count=0`.

Module-source vetting:

```text
.venv312/bin/python scripts/verify_first_principles_module_source_vetting.py --strict --check --date 2026_05_21
```

Result: `strict_passed=true`, `total_modules=298`.

SRS traceability export:

```text
.venv312/bin/python -m pytest tests/test_srs_traceability_export.py -q
```

Result: `2 passed`.

Active results hygiene:

```text
.venv312/bin/python scripts/verify_active_results_artifact_hygiene.py --strict --check
```

Result: `clean=true`, `active_hit_count=0`.

Acceptance dry run:

```text
.venv312/bin/python -m dpf first-principles-acceptance-dry-run
```

Result: all eight gates blocked; `report_only=True`,
`promotes_acceptance=False`, `accepted_runtime_claim=False`,
`can_support_first_principles_acceptance=False`, `fail_closed=True`.

Periodic audit:

```text
.venv312/bin/python scripts/run_codex_periodic_audit.py --timeout-seconds 900
```

Result:

```text
cycle 1: PASS head=df84751d7c04d5c200e29788b422e8dc8fcef05f
log=/private/tmp/dpf-unified-audit-logs/20260521T160018Z
```

## 4. Required P1-0 Cleanup

Before or at the beginning of P1, the team must reconcile the active artifact
linter policy wording with implementation behavior.

Choose one and document the decision:

1. Strict option: enforce that hybrid-PIC / architecture evidence appears only
   under approved context keys such as `architecture_or_schema_context_sources`,
   `cross_scope_context_sources`, or `source_scope_context`.
2. Current-behavior option: explicitly state that ordinary non-`same_scope`
   source fields may reference architecture/closure sources, while same-scope
   evidence keys remain protected.

If choosing option 1, add a negative fixture proving a hybrid-PIC slug under an
arbitrary non-context key fails.

If choosing option 2, update the script docstring, authority-policy JSON, and
completion memo language so the contract is not overstated.

This cleanup does not block P1 because the P0 safety property is already
verified: no architecture/cross-scope material can live under same-scope source
keys in active artifacts.

## 5. Super-Sprint 12 Phase P1 Instructions

P1 is authorized to begin. It is a physics-source closure sprint, not an
acceptance-promotion sprint.

The controlling scope is:

- validation scope: `pf1000_full_energy_27_to_40_kv`
- selected-machine source scope:
  `pf1000_scholz_2000_2001_24rod_large_electrode_full_energy_source`

P1 workstreams:

1. Same-scope PF-1000 full-energy source packet:
   - extract only from `KnowledgeReference/`;
   - required channels: geometry, bank/circuit, gas, current waveform, startup,
     density history, EM field history, temperature or distribution history,
     neutron scalar yield, neutron timing, spectrum, anisotropy, detector
     response, uncertainty budget, and review certificate;
   - absent channels remain explicitly blocked.

2. Numerical-fidelity acceptance suite:
   - source-backed method map;
   - manufactured or analytic reference cases;
   - mesh and timestep convergence families;
   - backend/precision parity;
   - restart reproducibility;
   - limiter-zero proof;
   - per-observable norms and tolerances.

3. Power-port Sigma-p face set:
   - reviewed interface surface or volume domain;
   - Poynting or `J dot E` power integral;
   - Auluck Eq.6 terms II/IV/V/VI mapping;
   - sign convention;
   - time centering;
   - residual tolerance.

4. Startup BVP:
   - D2 breakdown/preionization;
   - insulator flashover;
   - sheath liftoff;
   - early circuit handoff interval;
   - same-scope geometry and material surface inputs.

5. PF-1000 geometry completion:
   - anode hollow-bore length;
   - insulator wall thickness;
   - backplate radial extent;
   - backplate axial thickness;
   - reviewed conductor-mask packet.

6. Comparator/UQ/certificate skeleton:
   - output-field mapping by observable;
   - measurement/model/numerical uncertainty;
   - pass/fail metrics and tolerances;
   - negative controls;
   - run/evidence hashes;
   - independent-review placeholders that remain blocked until reviewed.

P1 guardrails:

- Do not set `accepted_runtime_claim=True`.
- Do not set top-level `can_support_first_principles_acceptance=True`.
- Do not mark a gate `accepted` unless source packet, code consumption,
  numerical tests, comparator/UQ, and certificate evidence all pass at the same
  commit.
- Reduced models, imported PIC, cross-scope diagnostics, and raw PDFs remain
  context/baseline only unless a reviewed transfer rule explicitly authorizes
  their use.

## 6. Required P1 Handoff Evidence

The next handoff must include:

- exact commit hash and changed-file list;
- P1-0 linter policy decision and tests;
- source packet channel matrix with `KnowledgeReference` paths and line ranges;
- list of every channel still blocked and why;
- numerical-fidelity test matrix and results;
- dry-run ledger output;
- source-truth and module-vetting output;
- periodic audit log path;
- explicit statement that acceptance flags remain false, or a complete
  certificate if that claim changes.
