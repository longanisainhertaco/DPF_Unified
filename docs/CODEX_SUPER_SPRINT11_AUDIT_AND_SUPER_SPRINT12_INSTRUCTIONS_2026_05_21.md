# Codex Super-Sprint 11 Audit and Super-Sprint 12 Instructions

Date: 2026-05-21
Branch audited: `codex/corpus`
Audited HEAD: `0dcf77826ea7df321d57479ac3b4dff213d96811`
Baseline: `fa713a83f53602d20d3eacc1a078202185f9b603`

## 1. Audit Verdict

Super-Sprint 11 is **not accepted as fully complete**.

It is accepted as a no-promotion policy and artifact-integrity improvement:

- no `accepted_runtime_claim` promotion was found;
- no top-level `can_support_first_principles_acceptance` promotion was found;
- the current dry-run ledger remains report-only and all eight gates are blocked;
- the focused SS11 integration gate passes;
- source-truth and module-source vetting pass at the 2026_05_21 baseline;
- the periodic audit passed at current HEAD.

It is not accepted as fully complete because adversarial audit probes found
remaining policy/namespace gaps in the workstreams that SS11 claimed closed:

- S10-A1 is fail-closed at packet output, but not fully clamped at direct
  `FirstPrinciples3DDeck` construction.
- S10-A2 still permits partial scope-pair matching and reports `scope_match=true`
  while echoing a requested source scope that does not match the deck actually run.
- S10-A3 removed the specific hybrid-PIC strings from `same_scope` runtime paths,
  but `same_scope_source` still carries `other_scope` context groups.
- S10-A4's active-results linter is clean for the two explicit stale strings, but
  its coverage is too narrow for future same-scope namespace enforcement.

The next sprint must close these four policy gaps first. Do not begin physics
acceptance work until the P0 closeout tests below pass.

## 2. Audit Findings

### SS11-A1 - Direct Runtime-Deck Imported-PIC Clamp Gap

Severity: Medium

Status: Open

SS11 fixed the canonical `FirstPrinciplesInputDeck -> FirstPrinciples3DDeck`
conversion path: `StartupPolicy.__post_init__` now forces
`can_support_whole_shot_acceptance=False` for
`imported_pic_sheath_state`.

The direct runtime-deck constructor remains a gap:

```text
FirstPrinciples3DDeck(
    startup_mode="imported_pic_sheath_state",
    startup_can_support_whole_shot_acceptance=True,
)
```

can still hold the raw field
`startup_can_support_whole_shot_acceptance=True`. `startup_packet()` blocks the
packet and returns `can_support_first_principles_acceptance=False`, so this is
not an acceptance promotion. It is still a policy integrity gap because raw
runtime-deck state can contradict the context-only startup policy.

Evidence:

- `src/dpf/first_principles/runner.py:149` defines `FirstPrinciples3DDeck`.
- `src/dpf/first_principles/runner.py:216` exposes the raw field.
- `src/dpf/first_principles/runner.py:456` passes the field into startup packet
  construction.
- Direct probe result:
  `deck_field_startup_can_support_whole_shot_acceptance=True`,
  `packet_status=blocked_startup_bvp_packet_not_available`,
  `packet_can_support_whole_shot_acceptance=False`,
  `packet_can_support_first_principles_acceptance=False`.

Required correction:

- Add a runtime-deck level clamp, preferably a `FirstPrinciples3DDeck.__post_init__`
  branch that forces `startup_can_support_whole_shot_acceptance=False` whenever
  `startup_mode` is in the deck/startup context-only taxonomy.
- Add a regression test that constructs `FirstPrinciples3DDeck` directly with
  `startup_mode="imported_pic_sheath_state"` and
  `startup_can_support_whole_shot_acceptance=True`, then asserts the deck field
  and packet output are both false.

### SS11-A2 - Readiness Scope Resolver Still Allows Partial Pair Matches

Severity: High

Status: Open

SS11 fixed the obvious Akel/full-energy contradiction path, but the resolver is
still token based instead of exact-pair based. A full-energy request with an
unknown or Akel-like source scope still runs the full-energy deck, echoes the
requested source scope, and reports `scope_match=true`.

Direct audit probe:

```text
validation_scope=pf1000_full_energy_27_to_40_kv
source_scope=pf1000_akel_candidate_paschen_insulator_seed_layer_not_startup_bvp
```

returned:

```text
runtime_deck_id=pf1000_scholz_2001_24rod_full_energy_27kv_3p5torr_engineering_candidate
source_scope=pf1000_akel_candidate_paschen_insulator_seed_layer_not_startup_bvp
requested_source_scope=pf1000_akel_candidate_paschen_insulator_seed_layer_not_startup_bvp
actual_runtime_source_scope=pf1000_scholz_2000_2001_24rod_large_electrode_full_energy_source
scope_match=True
ready=False
can_support_first_principles_acceptance=False
```

This does not promote readiness, but it preserves the label-mixing surface that
S10-A2 was meant to eliminate.

Required correction:

- Replace token-overlap scope resolution with exact accepted pairs:
  - Akel pair:
    `pf1000_16kv_2021_akel` plus `pf1000_16kv_2021_akel_shot12581`
  - Full-energy pair:
    `pf1000_full_energy_27_to_40_kv` plus
    `pf1000_scholz_2000_2001_24rod_large_electrode_full_energy_source`
- Any partial pair, unknown pair, undeclared pair, startup-source label, or mixed
  pair must return the fail-closed `not_run` payload.
- `scope_match` must be computed from exact requested-vs-actual pair equality,
  not from "resolved family exists".
- Add tests for:
  - full-energy validation plus Akel startup source label -> `not_run`;
  - full-energy validation plus `not_declared` source -> `not_run`;
  - unknown validation plus full-energy source -> `not_run`;
  - Akel validation plus full-energy source -> `not_run`;
  - exact Akel pair -> Akel deck;
  - exact full-energy pair -> full-energy deck.

### SS11-A3 - Same-Scope Packet Still Carries Other-Scope Context

Severity: Medium

Status: Open

SS11 correctly moved the LLNL-like hybrid-PIC architecture reference out of
`SAME_SCOPE_SOURCE_REFS` and out of forbidden `same_scope` runtime subtrees.
However, `build_same_scope_source_packet()` still returns
`other_scope_source_groups` inside the `same_scope_source` packet.

Evidence:

- `src/dpf/first_principles/same_scope.py:21` defines same-scope source refs.
- `src/dpf/first_principles/same_scope.py:157` defines other-scope groups.
- `src/dpf/first_principles/same_scope.py:301` returns
  `other_scope_source_groups` inside the same-scope packet.
- Runtime structured scan found zero instances of the three SS11 forbidden
  hybrid-PIC strings under `same_scope` keys, but still found `other_scope`
  tokens in `telemetry.same_scope_source` and manifest
  `same_scope_source_packet` subtrees.

Required correction:

- Split the packet into strict same-scope evidence and cross-scope context:
  - `same_scope_source` must contain only reviewed source references and channels
    whose scope exactly matches the declared runtime scope.
  - Move all `other_scope` groups, wrong-scope diagnostic context, and transfer
    guidance into a sibling key such as `cross_scope_context_sources` or
    `source_scope_context`.
- Add recursive tests that fail if any path containing `same_scope_source`
  contains a value or key with `other_scope`, `wrong_scope`, `llnl_like`, or the
  hybrid-PIC source slug.
- Preserve the context data; do not delete it. It is useful for rejection tests
  and schema design, but it must not live under the same-scope evidence packet.

### SS11-A4 - Active Results Linter Coverage Is Too Narrow

Severity: Medium

Status: Open

SS11 archived the six stale 2026_05_16 top-level result files and the new active
results linter passes. That closes the two explicit stale strings from S10-A4.

The linter currently checks only:

- `same_scope_3d_validation_packet`
- `llnl_like_180ka_axisymmetric_hybrid_pic`

Active non-archive result artifacts still contain the hybrid-PIC source path
outside same-scope subtrees. That is not automatically wrong, but the linter
does not yet enforce the intended structure: architecture evidence may appear
only under architecture/context fields, never under same-scope source fields.

Required correction:

- Extend `scripts/verify_active_results_artifact_hygiene.py` from flat string
  scanning to structured JSON path scanning.
- For active non-archive result JSON:
  - forbid the SS11 hybrid-PIC source slug under any path containing
    `same_scope`;
  - forbid `other_scope`/`wrong_scope` tokens under `same_scope_source`;
  - permit architecture evidence only under explicitly named
    `architecture_or_schema_context_sources`, `cross_scope_context_sources`, or
    equivalent non-acceptance context keys.
- Add negative tests proving an active JSON with architecture evidence under
  `same_scope_source` fails, and an active JSON with the same evidence under the
  approved architecture/context key passes.

### SS11-A5 - Dry-Run Predicate

Severity: None

Status: Closed

The dry-run predicate now requires:

- accepted packet status,
- empty missing list,
- `can_support_first_principles_acceptance is True`.

Adversarial probes with missing, `None`, and `False` flags all remained blocked.
The positive control with explicit `True` can pass one gate while the ledger
remains report-only and non-promoting.

### SS11-A6 - Traceability Evidence

Severity: None

Status: Closed

The completion memo includes source-truth, module-vetting, SRS/RTM, periodic
audit, and no-promotion evidence. Current-head periodic audit also passed:

```text
cycle 1: PASS head=0dcf77826ea7df321d57479ac3b4dff213d96811
log=/private/tmp/dpf-unified-audit-logs/20260521T140234Z
```

### SS11-A7 - No Acceptance Promotion

Severity: None

Status: Closed

No incorrect `accepted_runtime_claim`, `can_support_first_principles_acceptance`,
or `promotes_acceptance` promotion was found in the SS11 audit. The current
PF-1000 full-energy acceptance dry run still reports all eight gates blocked:

- numerical fidelity;
- same-scope comparator;
- UQ;
- certificate;
- geometry;
- startup;
- power port;
- neutron.

## 3. Verification Run By Codex

Commands run from `/Users/anthonyzamora/dpf-unified`:

```text
.venv312/bin/python -m pytest \
  tests/test_ss10_runner_deck_segmented_coherence.py \
  tests/test_ss10_imported_pic_context_only_policy.py \
  tests/test_first_principles_acceptance_gate_dry_run.py \
  tests/test_server_readiness.py \
  tests/test_git_status_clean_exception.py \
  tests/test_ws9_runner_scope_source_geometry.py \
  tests/test_startup_breakdown_audit.py \
  tests/test_first_principles_startup_bvp.py \
  tests/test_source_geometry_packet.py \
  tests/test_results_artifact_hygiene.py -q
```

Result: `241 passed in 37.34s`.

```text
ruff check src tests
```

Result: `All checks passed!`.

```text
.venv312/bin/python scripts/verify_first_principles_source_truth_exhaustion.py \
  --strict --check --date 2026_05_21
```

Result: `exhausted=true`, `open_issue_count=0`.

```text
.venv312/bin/python scripts/verify_first_principles_module_source_vetting.py \
  --strict --check --date 2026_05_21
```

Result: `strict_passed=true`, `total_modules=298`.

```text
.venv312/bin/python -m pytest tests/test_srs_traceability_export.py -q
```

Result: `2 passed`.

```text
.venv312/bin/python scripts/verify_active_results_artifact_hygiene.py \
  --strict --check
```

Result: `clean=true`, `active_hit_count=0`.

```text
.venv312/bin/python -m dpf first-principles-acceptance-dry-run
```

Result: all eight gates blocked; `report_only=True`,
`promotes_acceptance=False`, `accepted_runtime_claim=False`,
`can_support_first_principles_acceptance=False`, `fail_closed=True`.

```text
.venv312/bin/python scripts/run_codex_periodic_audit.py --timeout-seconds 900
```

Result: `PASS`, log
`/private/tmp/dpf-unified-audit-logs/20260521T140234Z`.

## 4. Super-Sprint 12 Work Order

Super-Sprint 12 is a two-phase sprint. Phase P0 must close SS11 audit gaps.
Phase P1 may start physics-source closure only after P0 passes.

### P0 - Finish SS11 Policy Integrity

P0 is mandatory before any new first-principles physics acceptance work.

Deliverables:

1. Runtime-deck imported-PIC clamp:
   - add direct `FirstPrinciples3DDeck` clamping for context-only startup modes;
   - add direct-constructor regression tests;
   - prove both raw deck field and startup packet output are false.

2. Exact-pair readiness scope resolver:
   - replace token-overlap matching with exact scope/source pair matching;
   - block all partial, unknown, undeclared, mixed, and startup-label source
     scopes with `runtime_deck_id=not_run`;
   - report `scope_match=false` unless requested and actual pairs are identical;
   - add the six tests listed in SS11-A2.

3. Same-scope namespace split:
   - remove `other_scope_source_groups` from `same_scope_source`;
   - move wrong-scope and architecture context into a sibling non-acceptance key;
   - add recursive tests banning `other_scope`, `wrong_scope`, `llnl_like`, and
     the hybrid-PIC source slug under `same_scope_source`.

4. Active artifact structured linter:
   - enforce the same namespace rule over active non-archive result JSON;
   - allow architecture evidence only under explicitly named context keys;
   - add positive and negative fixture tests.

5. Hygiene:
   - `git diff --check HEAD~...HEAD` or an equivalent commit-range hygiene gate
     must pass;
   - no new source-truth, module-vetting, or RTM drift.

P0 acceptance command set:

```text
.venv312/bin/python -m pytest \
  tests/test_ss10_imported_pic_context_only_policy.py \
  tests/test_server_readiness.py \
  tests/test_ws9_runner_scope_source_geometry.py \
  tests/test_results_artifact_hygiene.py \
  tests/test_first_principles_acceptance_gate_dry_run.py -q
ruff check src tests
.venv312/bin/python scripts/verify_active_results_artifact_hygiene.py --strict --check
.venv312/bin/python scripts/verify_first_principles_source_truth_exhaustion.py --strict --check --date 2026_05_21
.venv312/bin/python scripts/verify_first_principles_module_source_vetting.py --strict --check --date 2026_05_21
.venv312/bin/python -m dpf first-principles-acceptance-dry-run
```

Expected P0 result:

- all focused tests pass;
- all eight dry-run gates remain blocked;
- no acceptance flag is promoted;
- no same-scope namespace leak remains;
- all partial or mismatched readiness scope pairs fail closed with no deck run.

### P1 - Start Physics-Source Closure For PF-1000 Full-Energy

P1 is the real physics sprint, but it remains fail-closed until accepted by
source packet, code consumption, numerical fidelity, comparator/UQ, and
certificate gates at the same commit.

Workstreams:

1. Same-scope PF-1000 full-energy source packet:
   - target scope: `pf1000_full_energy_27_to_40_kv`;
   - selected source scope:
     `pf1000_scholz_2000_2001_24rod_large_electrode_full_energy_source`;
   - extract only from `KnowledgeReference/`;
   - required channels: geometry, bank/circuit, gas, current waveform, startup,
     density history, EM field history, temperature or distribution history,
     neutron scalar yield, neutron timing, spectrum, anisotropy, detector
     response, uncertainty budget, and review certificate;
   - any absent channel stays explicitly blocked.

2. Numerical-fidelity acceptance suite:
   - source-backed method map;
   - analytic/manufactured reference cases;
   - mesh and timestep convergence families;
   - backend/precision parity;
   - restart reproducibility;
   - limiter-zero proof;
   - per-observable norms and tolerances.

3. Power-port Sigma-p face set:
   - reviewed interface surface or volume domain;
   - Poynting or `J dot E` integral;
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
   - independent review placeholders that remain blocked until reviewed.

P1 acceptance boundary:

- Do not set `accepted_runtime_claim=True`.
- Do not set top-level `can_support_first_principles_acceptance=True`.
- Do not mark a gate `accepted` unless source packet, code consumption,
  numerical tests, comparator/UQ, and certificate evidence all pass at the same
  commit.
- Reduced models, imported PIC, cross-scope diagnostics, and raw PDFs remain
  context/baseline only unless a reviewed transfer rule explicitly authorizes
  their use.

## 5. Required Audit After Super-Sprint 12

The next handoff must include:

- exact commit hash and changed-file list;
- P0 adversarial probe output for direct imported-PIC deck construction;
- P0 adversarial probe output for all partial readiness scope pairs;
- recursive same-scope namespace scan output;
- active artifact structured linter output;
- dry-run ledger output;
- source-truth and module-vetting output;
- periodic audit log path;
- explicit list of any physics channels promoted to candidate or accepted, with
  `KnowledgeReference` source paths and line ranges;
- explicit statement that acceptance flags remain false, or a full certificate
  if that claim changes.
