# Super-Sprint 12 Phase P0 Completion Memo

Date: 2026-05-21
Branch: codex/corpus
Code commit: `5aefb6df707aa25f6c4d7393fd17e0dcae60fc2b` (`5aefb6d`)
Docs commit: assigned at commit time (this file lands in the docs commit)

---

## 1. Codex SS11 Audit Verdict

Super-Sprint 11 was **not accepted as fully complete**.

It was accepted as a no-promotion policy and artifact-integrity improvement:
no `accepted_runtime_claim` promotion was found; no top-level
`can_support_first_principles_acceptance` promotion was found; the dry-run
ledger remained report-only with all eight gates blocked; the focused SS11
integration gate passed (`241 passed`); source-truth exhaustion and
module-source vetting passed at the 2026_05_21 baseline; the periodic audit
passed at HEAD `0dcf77826ea7df321d57479ac3b4dff213d96811`.

The audit found four residual policy/namespace gaps in the workstreams that
SS11 claimed closed (findings SS11-A1 through SS11-A4). SS11-A5 (dry-run
predicate), SS11-A6 (traceability evidence), and SS11-A7 (no acceptance
promotion) were already Closed.

Phase P0 is the mandatory closeout of those four gaps before any P1 physics
acceptance work may begin.

---

## 2. P0 Workstream Outcomes

### P0-1 — closes SS11-A1 (Direct Runtime-Deck Imported-PIC Clamp Gap)

File: `src/dpf/first_principles/runner.py:247–269`

`FirstPrinciples3DDeck` is a frozen dataclass. A `__post_init__` was added that
uses `object.__setattr__` to force
`startup_can_support_whole_shot_acceptance=False` whenever `startup_mode` is in
`CONTEXT_ONLY_STARTUP_MODES` (imported from `deck.py`). This clamp fires on
every construction path — direct constructor, `from_deck`, and
`dataclasses.replace` — so no path through the runtime-deck layer can hold the
raw field `True` for imported-PIC startup modes.

Adversarial direct-constructor probe result (post-fix):

```
deck field startup_can_support_whole_shot_acceptance=False
packet_status=blocked_startup_bvp_packet_not_available
packet can_support_whole_shot_acceptance=False
packet can_support_first_principles_acceptance=False
```

Tests: 3 new tests in `tests/test_ss10_imported_pic_context_only_policy.py`.

### P0-2 — closes SS11-A2 (Readiness Scope Resolver Still Allows Partial Pair Matches)

File: `src/dpf/server/readiness.py`

`_resolve_runtime_deck_scope` replaced token-overlap matching with exact
ordered `(validation_scope, source_scope)` pair matching. Two accepted pairs
are defined: `_AKEL_SCOPE_PAIR` and `_FULL_ENERGY_SCOPE_PAIR`. Any
partial/unknown/undeclared/mixed/startup-label pair returns the fail-closed
payload with `runtime_deck_id=not_run` and no deck execution. `scope_match` is
computed from exact pair equality (previously hard-coded `True`).

Adversarial probe — the SS11-A2 mixed pair
(`validation_scope=pf1000_full_energy_27_to_40_kv`,
`source_scope=pf1000_akel_candidate_paschen_insulator_seed_layer_not_startup_bvp`)
now returns:

```
runtime_deck_id=not_run
scope_match=False
(no deck executed)
```

Tests: 6 new tests in `tests/test_server_readiness.py` covering the six
SS11-A2 cases.

### P0-3 — closes SS11-A3 (Same-Scope Packet Still Carries Other-Scope Context)

File: `src/dpf/first_principles/same_scope.py`

`other_scope_source_groups` and the cross-scope policy were removed from the
`same_scope_source` packet built by `build_same_scope_source_packet`. A new
`build_cross_scope_context_sources()` helper and `CROSS_SCOPE_*` constants
carry the relocated material. `src/dpf/first_principles/runner.py` emits a
sibling `cross_scope_context_sources` key in both telemetry and the segmented
manifest.

Recursive runtime scan of the PF-1000 full-energy preset: **0 leaks** — no
path containing `same_scope_source` carries `other_scope`, `wrong_scope`,
`llnl_like`, or the hybrid-PIC source slug.

Tests: recursive ban tests added to
`tests/test_ws9_runner_scope_source_geometry.py`.

### P0-4 — closes SS11-A4 (Active Results Linter Coverage Too Narrow)

File: `scripts/verify_active_results_artifact_hygiene.py`

Rewritten from flat-string scanning to structured JSON key-chain scanning over
both scalar VALUES and dict KEY NAMES. The enforced safety property is that the
hybrid-PIC slugs are forbidden under any `same_scope` key chain and
`other_scope`/`wrong_scope` under any `same_scope_source` key chain;
architecture evidence may otherwise appear in ordinary non-`same_scope` source
fields, with the approved `*_context_sources` / `source_scope_context` keys the
recommended home for relocated cross-scope context. The linter excludes
`archive_*` directories and reports malformed files.

Opus code review found one HIGH: the linter inspected only values, not key
names. This was FIXED — key-name scanning was added along with 2 additional
fixture tests, and the linter remains `clean=true`, `active_hit_count=0`.

Tests: 15 tests in `tests/test_results_artifact_hygiene.py`.

---

## 3. Required Audit After Super-Sprint 12 — Handoff Evidence (§5)

### Commit hashes and changed-file list

Code commit: `5aefb6df707aa25f6c4d7393fd17e0dcae60fc2b` (`5aefb6d`)
Docs commit: assigned at commit time (this memo, `CodexFindings.md`,
`CortexFindings.md`, `docs/external_team_submissions/.../CHANGELOG.md`)

Changed files in code commit `5aefb6d`:
- `src/dpf/first_principles/runner.py` (P0-1 `__post_init__` clamp; P0-3
  cross_scope_context_sources sibling key; test-maintenance namespace update)
- `src/dpf/server/readiness.py` (P0-2 exact-pair resolver)
- `src/dpf/first_principles/same_scope.py` (P0-3 namespace split;
  build_cross_scope_context_sources helper; CROSS_SCOPE_* constants)
- `scripts/verify_active_results_artifact_hygiene.py` (P0-4 structured
  key-chain linter with key-name scanning)
- `tests/test_ss10_imported_pic_context_only_policy.py` (P0-1: 3 new tests)
- `tests/test_server_readiness.py` (P0-2: 6 new tests)
- `tests/test_ws9_runner_scope_source_geometry.py` (P0-3: recursive ban tests)
- `tests/test_results_artifact_hygiene.py` (P0-4: 15 tests incl. 2 key-name
  fixture tests added post-Opus review)
- `tests/test_first_principles_runner.py` (test maintenance: namespace update
  for cross_scope_policy relocation)
- `tests/test_first_principles_same_scope_full_energy_packet.py` (test
  maintenance: other_scope_diagnostics -> cross_scope_diagnostics rename)

### P0 acceptance test suite

```
.venv312/bin/python -m pytest \
  tests/test_ss10_imported_pic_context_only_policy.py \
  tests/test_server_readiness.py \
  tests/test_ws9_runner_scope_source_geometry.py \
  tests/test_results_artifact_hygiene.py \
  tests/test_first_principles_acceptance_gate_dry_run.py -q
```

Lead-verified result: **84 passed**.

Pre-commit quality gate at code commit `5aefb6d`: **123 passed**.

Ruff: `ruff check src tests` → `All checks passed.`

### Direct imported-PIC deck-construction adversarial probe

Post-fix probe (constructing `FirstPrinciples3DDeck` directly with
`startup_mode="imported_pic_sheath_state"` and
`startup_can_support_whole_shot_acceptance=True`):

```
deck field startup_can_support_whole_shot_acceptance=False
packet_status=blocked_startup_bvp_packet_not_available
packet can_support_whole_shot_acceptance=False
packet can_support_first_principles_acceptance=False
```

Both raw deck field and packet output are False. P0-1 closes SS11-A1.

### Readiness scope-pair adversarial probe

Post-fix probe (mixed pair: `validation_scope=pf1000_full_energy_27_to_40_kv`,
`source_scope=pf1000_akel_candidate_paschen_insulator_seed_layer_not_startup_bvp`):

```
runtime_deck_id=not_run
scope_match=False
(no deck executed)
```

All six SS11-A2 adversarial cases (full-energy + Akel source label,
full-energy + not_declared source, unknown validation + full-energy source,
Akel validation + full-energy source, exact Akel pair → Akel deck, exact
full-energy pair → full-energy deck) pass. P0-2 closes SS11-A2.

### Recursive same-scope namespace scan result

Recursive runtime scan of PF-1000 full-energy preset over all paths containing
`same_scope_source` for tokens `other_scope`, `wrong_scope`, `llnl_like`, or
the hybrid-PIC source slug:

```
0 leaks
```

All relocated material is preserved under `cross_scope_context_sources`. P0-3
closes SS11-A3.

### Active artifact structured linter output

Command: `.venv312/bin/python scripts/verify_active_results_artifact_hygiene.py --strict --check`

Verbatim output:

```json
{
  "active_hit_count": 0,
  "approved_context_keys": {
    "exact": [
      "source_scope_context"
    ],
    "suffix": [
      "_context_sources"
    ]
  },
  "authority_policy": "results/ JSON artifacts outside archive_* directories are walked by key chain; the SS11 hybrid-PIC source slugs are forbidden under any 'same_scope' key chain, 'other_scope'/'wrong_scope' tokens are forbidden under any 'same_scope_source' key chain (over both scalar values and dict key names); architecture or cross-scope evidence may otherwise appear in ordinary non-same_scope source fields, with the approved context keys (a key ending in '_context_sources' or named 'source_scope_context') the recommended home for relocated cross-scope context; stale artifacts are relocated (not rewritten) to archive_* dirs",
  "clean": true,
  "hybrid_pic_slugs": [
    "llnl_like_180ka_axisymmetric_hybrid_pic",
    "fully-electromagnetic-hybrid-pic-fluid-dpf-neutron-yield",
    "hybrid_pic_architecture_order_of_magnitude_other_scope"
  ],
  "issues": [],
  "ordinary_non_same_scope_source_fields": "allowed",
  "protected_key_chains": [
    "same_scope",
    "same_scope_source"
  ],
  "scope": "active_results_artifact_hygiene",
  "wrong_scope_tokens": [
    "other_scope",
    "wrong_scope"
  ]
}
```

`clean=true`, `active_hit_count=0`. P0-4 closes SS11-A4.

### P1-0 — Active-Results Linter Policy Reconciliation (2026-05-21)

The Codex SS12-P0 audit (finding P0-A4) noted that the linter docstring,
`authority_policy` JSON, and this memo's wording overstated the contract — they
said architecture/cross-scope evidence is "permitted only under approved
context keys", while the implementation enforces the narrower, intended safety
property: such evidence is FORBIDDEN under `same_scope` key chains and may
otherwise appear in ordinary non-`same_scope` source fields.

P1-0 decision — **option 2 (current behavior)**. "Forbidden under same-scope
keys" is the correct safety property and matches the live artifact surface: the
hybrid-PIC slug legitimately appears in ordinary closure / power-port `source`
attribution fields, which is not a namespace violation. Option 1 (forbid the
slug under every non-context key) would over-restrict and falsely flag those
legitimate citations. The linter docstring, the `authority_policy` JSON, and
this memo were corrected to state the enforced property accurately; the linter
behavior is unchanged and `active_hit_count` remains 0. No negative fixture for
a slug "under an arbitrary non-context key" is added, because under option 2
that case is permitted, not a violation.

The machine-readable linter report now makes the same decision explicit with
`ordinary_non_same_scope_source_fields="allowed"` and
`protected_key_chains=["same_scope", "same_scope_source"]`.

### Dry-run ledger output

Command: `.venv312/bin/python -m dpf first-principles-acceptance-dry-run`

Verbatim output:

```
Report-only acceptance-gate dry run (SS10-7)
  deck_preset: built_in:pf1000_scholz_2001_24rod_full_energy
  runtime_status: engineering_candidate_not_validation
  report_only: True
  promotes_acceptance: False
  accepted_runtime_claim: False
  fail_closed: True
  numerical_fidelity: blocked (25 missing)
  same_scope_comparator: blocked (17 missing)
  uq: blocked (19 missing)
  certificate: blocked (31 missing)
  geometry: blocked (5 missing)
  startup: blocked (24 missing)
  power_port: blocked (8 missing)
  neutron: blocked (19 missing)
  ledger_json: /var/folders/dk/7vdd4krn6nq2_f25bsfph_dr0000gn/T/dpf_first_principles_acceptance_gate_dry_run.json
```

All eight gates blocked. `report_only=True`, `promotes_acceptance=False`,
`accepted_runtime_claim=False`, `can_support_first_principles_acceptance=False`,
`fail_closed=True`.

### Source-truth exhaustion output

Command: `.venv312/bin/python scripts/verify_first_principles_source_truth_exhaustion.py --strict --check --date 2026_05_21`

Verbatim output:

```json
{
  "exhausted": true,
  "json": "docs/FIRST_PRINCIPLES_SOURCE_TRUTH_EXHAUSTION_2026_05_21.json",
  "markdown": "docs/FIRST_PRINCIPLES_SOURCE_TRUTH_EXHAUSTION_2026_05_21.md",
  "open_issue_count": 0
}
```

`exhausted=true`, `open_issue_count=0`.

### Module-source vetting output

Command: `.venv312/bin/python scripts/verify_first_principles_module_source_vetting.py --strict --check --date 2026_05_21`

Verbatim output:

```json
{
  "active_physics_unvetted_count": 0,
  "json": "docs/FIRST_PRINCIPLES_MODULE_SOURCE_VETTING_2026_05_21.json",
  "markdown": "docs/FIRST_PRINCIPLES_MODULE_SOURCE_VETTING_2026_05_21.md",
  "missing_source_reference_count": 0,
  "strict_passed": true,
  "total_modules": 298
}
```

`strict_passed=true`, `total_modules=298`.

### Periodic audit log path

The periodic audit was run as a six-cycle campaign at the SS12-P0
documentation commit. Five of the six cycles passed all ten gates
(cycles 2-6); cycle 1 hit the pre-existing non-deterministic native
batch crash (SWIG / Athena++ test layer) on one pytest gate — not an
SS12-P0 regression, and that gate passes in every clean cycle. Clean
cycle `20260521T153243Z` — `Passed: True`, all ten gates:

| Gate | Result |
| --- | --- |
| `git_status_clean` | PASS (146-line external-churn exception) |
| `git_head` | PASS |
| `git_diff_check` | PASS |
| `source_truth_exhaustion` | PASS |
| `module_source_vetting` | PASS |
| `artifact_linter_active` | PASS |
| `artifact_linter_recursive` | PASS |
| `ruff_src_tests` | PASS |
| `focused_pytest` | PASS (99 passed) |
| `broad_first_principles_pytest` | PASS (559 passed) |

`git_status_clean` PASS carries the verbatim exception note:

> APPROVED EXCEPTION: 146 known external-churn line(s) excused (Sprint 9
> WS9-0 decision) -- PDF-symlink typechanges in `downloaded_books_papers/`,
> `tmp/pdfs/` and the `external/athenak` dependency submodule.

Clean-cycle log directories under `/private/tmp/dpf-unified-audit-logs/`:
`20260521T152602Z`, `20260521T152745Z`, `20260521T152928Z`,
`20260521T153106Z`, `20260521T153243Z`. The periodic audit passes
**10/10** for Super-Sprint 12 Phase P0.

### Physics channels promoted to candidate or accepted

ZERO physics channels were promoted to candidate or accepted. P0 is
policy/artifact integrity work only. No `KnowledgeReference` source paths were
consumed for acceptance promotion. No physics blocker was closed.

### Acceptance flag status

`accepted_runtime_claim=False`, `can_support_first_principles_acceptance=False`,
and `promotes_acceptance=False` everywhere in the codebase at this commit.
These flags were not set `True` anywhere in P0 and remain unchanged from the
SS11 baseline.

---

## 4. Known Scope Boundary

### MEDIUM — Other packet builders retain OTHER_SCOPE_SOURCE_GROUPS (deliberate)

Three other packet builders — `src/dpf/first_principles/comparator_uq.py`,
`neutron_authority.py`, and `spatial_field_temperature.py` — still define
`OTHER_SCOPE_SOURCE_GROUPS` and emit `other_scope_source_groups` keys.

These packets are emitted under their OWN non-same_scope telemetry and manifest
keys (`comparator_uq`, `neutron_authority`, `spatial_field_temperature`), never
under a `same_scope_source` key chain. This is NOT a namespace leak and is
compliant with the literal scope of SS11-A3 and P0-3, which targeted the
`same_scope_source` namespace specifically.

The SS12 key-name-aware linter (`scripts/verify_active_results_artifact_hygiene.py`)
would catch any future misfiling of these groups under a `same_scope` key
chain. The remaining inconsistency — that these builders use the same
`OTHER_SCOPE_SOURCE_GROUPS` constant name — is a candidate for a future
consistency pass to rename them to `CROSS_SCOPE_SOURCE_GROUPS` and align with
the P0-3 naming convention, but it is not a policy violation at this commit.

### LOW — Readiness scope constants duplicate canonical labels

The accepted scope pair constants (`_AKEL_SCOPE_PAIR`, `_FULL_ENERGY_SCOPE_PAIR`)
in `src/dpf/server/readiness.py` duplicate the canonical label strings that are
defined elsewhere in the codebase. This creates a maintenance risk if canonical
labels change without updating the readiness pairs. Follow-up: centralize the
canonical pair definitions or add a test that asserts the constants equal the
canonical values.

---

## 5. P1 Authorization

P1 physics work (PF-1000 full-energy same-scope source packet, numerical
fidelity, startup BVP, power-port Sigma-p, geometry, comparator/UQ, and
certificate gates) may begin only after P0 is accepted by the Codex audit.

P0 has closed SS11-A1 through SS11-A4, preserved the no-promotion boundary,
and passed all acceptance-gate commands. The lead must run the periodic audit
and formally accept P0 before P1 begins.
