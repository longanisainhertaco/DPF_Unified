# Package-Native 3D Runner Acceptance Contract — Sprint 6 WS5 Design (2026-05-20)

## 1. Problem statement

### 1.1 Where the 3D runner emits its labels and outputs

The package-native 3D hybrid EM/PIC-fluid runner is implemented in
`src/dpf/first_principles/runner.py`. Its backend identity is set at
`runner.py:2890`:

```python
backend="package_native",
```

The module-level constant at `runner.py:94` defines the run mode:

```python
RUN_MODE = "first_principles_3d_hybrid_em_pic_fluid"
```

The `HybridEMPicFluidRunResult.to_dict()` method at `runner.py:491-504`
serialises the return object. The top-level keys it emits are:

```
status, run_mode, scientific_status, reduced_models_used,
can_support_first_principles_acceptance, manifest,
conservation_telemetry, validation_packet, telemetry
```

Critically, the acceptance-relevant evidence packets — limiter ledger,
hybrid_pic_3d_evidence, physics_fidelity_evidence, mhd_numerical_fidelity,
energy accounting, same_scope_source — are **not at top level**. They are
nested under `telemetry.*` (assembled at `runner.py:1184-1223`) and under
`candidate_evidence.*` (assembled at `runner.py:2916-2948`). The
`validation_packet` is a sub-key, not a top-level mapping the gate can index
directly.

The geometry's `source_scope` field (deck.py:279, defaulting to
`"end_of_rundown_or_engineering_startup"`) is carried through at
`runner.py:1168` and embedded inside `telemetry["source_scope"]` and
`validation_packet` rather than raised to the root of the result dict.

### 1.2 Why `first_principles_mhd_readiness_report` cannot accept this run

`first_principles_mhd_readiness_report` is defined at
`src/dpf/validation/first_principles_mhd.py:674`. Its backend gate at
`mhd.py:698` calls `first_principles_backend_scope_status(result)`, which
is defined at `mhd.py:395-462`.

The backend classification at `mhd.py:416-429` accepts only a backend whose
label starts with `"python"` and does not contain `"fallback"`:

```python
python_instrumented = normalized.startswith("python") and "fallback" not in normalized
if python_instrumented:
    return { "status": "python_cylindrical_instrumented",
             "can_support_first_principles_acceptance": True, ... }
```

The 3D runner's backend label is `"package_native"` (`runner.py:2890`), which
does not start with `"python"`. The fallthrough path at `mhd.py:431-462`
matches it against `("athenak", "athena", "metal", "mlx", "hybrid")`. Since
`"package_native"` matches none of these, `blocked_token` is set to
`normalized` itself (`"package_native"`) and the function returns
`can_support_first_principles_acceptance: False` with status
`"backend_scope_blocked"`.

Consequently, the readiness report at `mhd.py:753-760` always appends a
blocker for `"instrumented_backend_scope"`. There is no dispatch path in the
report that recognises `"package_native"` as a separate contract requiring its
own evidence schema.

Additionally, `first_principles_mhd_readiness_report` interrogates the result
dict using top-level key lookups (e.g., `result.get("physics_fidelity_evidence")`
at `mhd.py:822`, `result.get("mhd_numerical_fidelity")` at `mhd.py:832`). The
3D runner buries these under `telemetry.*`, so even if the backend gate were
widened, the evidence lookups would return `None` and every downstream check
would block.

### 1.3 Audit findings that have called this out

The gate-ledger row at
`docs/FIRST_PRINCIPLES_PHYSICS_ACCEPTANCE_GATE_LEDGER_2026_05_20.csv` row 12:

```
package_native_3d_acceptance_contract,control_plane,
PACKAGE-NATIVE-3D-ACCEPTANCE-CONTRACT;FIRST-PRINCIPLES-MHD-GATE-PARITY,
...accepted_physics_allowed=false,...
```

explicitly classifies the 3D runner's acceptance gate as blocked and requires
two blocker IDs to be resolved: `PACKAGE-NATIVE-3D-ACCEPTANCE-CONTRACT` and
`FIRST-PRINCIPLES-MHD-GATE-PARITY`. The Codex Sprint 5 WS2 "Structural Blockers
Remain" section identified that the gate has no acceptance branch for the 3D
runner and that the cylindrical gate's scope must not be silently widened.

---

## 2. Design goals

1. **Preserve fail-closed posture.** `accepted_runtime_claim=false` and
   `can_support_first_principles_acceptance=false` remain the default for every
   3D run until all upstream gates in the named contract pass. A missing or
   malformed required key produces an immediate named blocker, not a silent pass.

2. **Do NOT apply a `python_*` backend label to the 3D runner.** The 3D runner's
   identity is `backend="package_native"` (`runner.py:2890`). Relabelling it as
   `python_*` would misrepresent the execution stack and conflate the cylindrical
   Python MHD path with the hybrid PIC-fluid 3D path. This is explicitly
   prohibited.

3. **Do NOT silently widen the cylindrical gate.** The `python_cylindrical_instrumented`
   branch at `mhd.py:417-429` and the scope check at `mhd.py:714-721` are scoped
   to the PF-1000/Akel 16 kV cylindrical Python path. They must not be modified to
   absorb the 3D runner.

4. **Introduce a named 3D acceptance contract with its own required-output schema.**
   The contract is a distinct function with explicit key expectations. Any key
   required by the cylindrical gate that the 3D runner does not emit at top-level
   must be either raised to top-level by the runner (preferred) or explicitly
   mapped within the 3D contract (if raising is a larger refactor deferred to
   Sprint 8+).

---

## 3. Proposed contract — schema a 3D run must satisfy

### 3.1 Backend label values accepted by this contract

```
package_native_3d_hybrid_pic_fluid
```

The existing `"package_native"` label (`runner.py:2890`) must be updated to this
value before the contract can be invoked. The contract function must reject any
result whose backend label is not exactly `"package_native_3d_hybrid_pic_fluid"`,
preventing it from accidentally absorbing unlabelled or future backend variants.

### 3.2 Source-scope values accepted — explicit allow-list

```
pf1000_akel_16kv_2021_shot_12581
pf1000_full_energy_27_to_40_kv_gribkov_scholz_era
```

No other source-scope value may be used to satisfy the contract. The 3D runner
currently receives `geometry.source_scope` from the deck (`runner.py:1168`,
`deck.py:279`). The deck's default `"end_of_rundown_or_engineering_startup"` is
not on this allow-list and must not be accepted.

### 3.3 Required top-level result keys

The 3D runner must emit these keys at the top level of its result dict (i.e.,
accessible via `result.get(key)`) before the contract can be checked. Currently
they are nested under `telemetry.*` or `candidate_evidence.*`. The contract
design records the required keys; the implementation sprint is responsible for
raising them.

```
backend                              # "package_native_3d_hybrid_pic_fluid"
source_scope                         # one of the allowed values above
run_mode                             # "first_principles_3d_hybrid_em_pic_fluid"
first_principles_limiter_ledger      # pass-through from telemetry["limiter_readiness"]
hybrid_pic_3d_evidence               # from telemetry["candidate_evidence"]
physics_fidelity_evidence            # from validation_packet["physics_fidelity_evidence"]
mhd_numerical_fidelity               # from telemetry["numerical_fidelity"]
first_principles_energy_accounting   # from conservation_telemetry
same_scope_source                    # from telemetry["same_scope_source"]
certificate_gate                     # from telemetry["certificate_gate"]
```

### 3.4 Required attached packets — structural requirements

Each packet must be a non-empty mapping with at least a `"status"` key.
The contract checks these at the schema level only; physics correctness
of each packet is the responsibility of the upstream gate that produces it.

| Packet key | Required sub-key | Accepted value or shape |
|---|---|---|
| `first_principles_limiter_ledger` | `can_support_first_principles_acceptance` | `True` (bool) |
| `hybrid_pic_3d_evidence` | any non-empty mapping | present |
| `physics_fidelity_evidence` | `passed` | `True` (bool) |
| `mhd_numerical_fidelity` | `passed` | `True` (bool) |
| `first_principles_energy_accounting` | `can_support_first_principles_acceptance` | `True` (bool) |
| `same_scope_source` | `status` | not `"blocked_same_scope_source_packet_not_available"` |
| `certificate_gate` | `status` | not blocked |

### 3.5 Explicit non-acceptance preconditions

If any of the following conditions is true, the contract immediately returns
`can_support_first_principles_acceptance: False` with a named blocker string.
No partial acceptance is possible.

| Condition | Blocker ID |
|---|---|
| `result.get("backend")` is not `"package_native_3d_hybrid_pic_fluid"` | `BACKEND_LABEL_MISMATCH` |
| `result.get("source_scope")` not in the allow-list | `SOURCE_SCOPE_NOT_ACCEPTED` |
| Any required top-level key is missing | `REQUIRED_KEY_MISSING:<key_name>` |
| `first_principles_limiter_ledger["can_support_first_principles_acceptance"]` is not `True` | `LIMITER_LEDGER_NOT_ACCEPTED` |
| `physics_fidelity_evidence["passed"]` is not `True` | `PHYSICS_FIDELITY_NOT_ACCEPTED` |
| `mhd_numerical_fidelity["passed"]` is not `True` | `NUMERICAL_FIDELITY_NOT_ACCEPTED` |
| `first_principles_energy_accounting["can_support_first_principles_acceptance"]` is not `True` | `ENERGY_ACCOUNTING_NOT_ACCEPTED` |
| `same_scope_source` is blocked | `SAME_SCOPE_BLOCKED` |
| `certificate_gate` is blocked | `CERTIFICATE_GATE_BLOCKED` |

---

## 4. Proposed code surfaces

### 4.1 New function — `package_native_3d_acceptance_status`

Location: `src/dpf/validation/first_principles_mhd.py`

Signature:

```python
def package_native_3d_acceptance_status(
    result: Mapping[str, object],
) -> dict[str, object]:
    """Check package-native 3D hybrid PIC-fluid result against the named
    acceptance contract.

    Returns a status mapping with:
      - ``can_support_first_principles_acceptance`` (bool): True only if all
        required keys are present and all packet sub-checks pass.
      - ``missing_key_blockers`` (list[str]): REQUIRED_KEY_MISSING:<key> entries
        for every absent top-level key.
      - ``failed_packet_blockers`` (list[str]): named blocker IDs for failed
        packet sub-checks.
      - ``source_scope`` (str): the declared source scope from the result.
      - ``backend`` (str): the backend label from the result.
      - ``status`` (str): one of
          "package_native_3d_contract_passed" |
          "package_native_3d_contract_blocked"
    """
```

This function must NOT call `first_principles_mhd_readiness_report` internally
and must NOT share state with the cylindrical gate. It is a parallel, named
acceptance path.

### 4.2 Dispatch in `first_principles_mhd_readiness_report`

Location: `src/dpf/validation/first_principles_mhd.py:698` (the call to
`first_principles_backend_scope_status`).

A dispatch guard must be added **before** the existing backend gate:

```python
backend_label = str(result.get("backend") or "").strip().lower()
if backend_label == "package_native_3d_hybrid_pic_fluid":
    return _first_principles_mhd_readiness_from_3d_contract(result)
```

`_first_principles_mhd_readiness_from_3d_contract` is a private helper that
calls `package_native_3d_acceptance_status` and assembles a
`FirstPrinciplesMHDReadiness` from its output. It must return
`ready=False` with the 3D contract's blocker list until the contract passes.
It must not inherit the cylindrical gate's PF-1000/Akel scope check at
`mhd.py:714-721` — the 3D contract has its own source-scope allow-list
(§3.2 above).

### 4.3 New test module

Location: `tests/test_first_principles_package_native_3d_acceptance.py`

Required test cases (full specifications in §6 of this memo):

- `test_missing_backend_label_blocks` — result with no `backend` key returns
  `can_support_first_principles_acceptance=False`.
- `test_wrong_backend_label_blocks` — result with `backend="package_native"`
  (old label) returns `can_support_first_principles_acceptance=False` with
  blocker `BACKEND_LABEL_MISMATCH`.
- `test_source_scope_not_in_allowlist_blocks` — `source_scope` set to
  `"end_of_rundown_or_engineering_startup"` blocks.
- `test_required_key_missing_blocks` — each required top-level key missing
  individually produces a `REQUIRED_KEY_MISSING:<key>` blocker.
- `test_failed_limiter_ledger_blocks` — correct backend + scope but
  `first_principles_limiter_ledger["can_support_first_principles_acceptance"]=False`
  blocks.
- `test_failed_physics_fidelity_blocks` — `physics_fidelity_evidence["passed"]=False`
  blocks.
- `test_regression_cylindrical_gate_unchanged` — a result with
  `backend="python_mhd_cylindrical"` and correct Akel scope still routes
  through the cylindrical gate and returns the same status as before this
  dispatch guard was added.

---

## 5. What this design does NOT do

- Does not implement the contract — that is Sprint 7 work.
- Does not modify `first_principles_backend_scope_status` or the
  `python_cylindrical_instrumented` branch at `mhd.py:416-429`.
- Does not promote the 3D runner to accepted physics — the 3D runner sets
  `can_support_first_principles_acceptance: False` at `runner.py:1222` and
  `runner.py:2915`; those values remain unchanged.
- Does not pre-commit to a particular implementation language for the required
  packets — the contract specifies the schema, not the code that produces it.
- Does not change the gate ledger — row 12 remains `accepted_physics_allowed=false`
  until the implementation sprint provides evidence that all contract preconditions
  pass.

---

## 6. Pre-conditions before implementation

1. **Codex audit must approve the contract schema** defined in §3 of this memo.
   Specifically, the allow-listed source-scope values and the required top-level
   key list must be reviewed against the 3D runner's actual output.

2. **The 3D runner's source_scope must be reviewed** — the current deck default
   `"end_of_rundown_or_engineering_startup"` (`deck.py:279`) is not on the
   allow-list. The implementation sprint must either update the default or require
   callers to set an accepted scope explicitly.

3. **The required top-level keys must be verified to be achievable** from the 3D
   runner's internal state. As of `runner.py:491-504`, `HybridEMPicFluidRunResult.to_dict()`
   does not emit `first_principles_limiter_ledger`, `hybrid_pic_3d_evidence`,
   `physics_fidelity_evidence`, or `same_scope_source` at the top level. The
   implementation sprint must raise these or define an explicit mapping strategy
   before the contract function can be written.

4. **Backend label migration** — the existing `backend="package_native"` at
   `runner.py:2890` must be updated to `"package_native_3d_hybrid_pic_fluid"`.
   The migration must be accompanied by a grep-and-verify pass across all call
   sites and tests that check for `"package_native"` to prevent silent regressions.

5. **Gate parity review** — the blocker ID `FIRST-PRINCIPLES-MHD-GATE-PARITY`
   in the ledger row requires that the 3D contract prove parity with the
   cylindrical gate for the evidence categories it shares (limiter ledger, energy
   accounting, numerical fidelity). The implementation sprint must produce a
   parity matrix before closing that blocker.

---

## 7. Ledger traceability

Gate ledger row:
`docs/FIRST_PRINCIPLES_PHYSICS_ACCEPTANCE_GATE_LEDGER_2026_05_20.csv` row 12

Blocker IDs resolved by this design (not yet closed — design only):
- `PACKAGE-NATIVE-3D-ACCEPTANCE-CONTRACT` (design memo committed here)
- `FIRST-PRINCIPLES-MHD-GATE-PARITY` (parity matrix deferred to implementation sprint)

Affected source files (read-only in this sprint):
- `src/dpf/validation/first_principles_mhd.py` (gate function, lines 395-462 and 674-876)
- `src/dpf/first_principles/runner.py` (backend label line 2890, output structure lines 491-504 and 2880-2951)
- `src/dpf/first_principles/deck.py` (source_scope default line 279)
