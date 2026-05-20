# Claim-Limited Te/Ti Observable Exclusion — Sprint 6 WS5 Design (2026-05-20)

## 1. Problem statement

### 1.1 Structural location of the blocking

`src/dpf/first_principles/same_scope.py` defines
`BLOCKING_SAME_SCOPE_CHANNELS` at lines 72-86:

```python
BLOCKING_SAME_SCOPE_CHANNELS = (
    "accepted_digitized_current_waveform",
    "startup_breakdown_preionization",
    "density_spatial_history",
    "em_field_history",
    "electron_temperature_history",
    "ion_temperature_or_distribution_history",
    "neutron_timing_history",
    "neutron_spectrum",
    "neutron_anisotropy",
    "detector_response_and_calibration",
    "uncertainty_budget",
    "source_review_certificate",
    "cross_scope_transfer_rule_or_rejection_tests",
)
```

In `build_same_scope_source_packet` at `same_scope.py:117-174`, the missing-set
is computed at line 143:

```python
missing.update(BLOCKING_SAME_SCOPE_CHANNELS)
```

This unconditional `update` means `electron_temperature_history` and
`ion_temperature_or_distribution_history` are permanently in the missing set
regardless of what the calling code declares. The gate returns
`status="blocked_same_scope_source_packet_not_available"` and
`decision="do_not_promote_whole_shot_first_principles_claim"` at `same_scope.py:148-151`
for every invocation.

### 1.2 Literature state for same-scope PF-1000 bulk-pinch Te and Ti

No DPF paper in the local `KnowledgeReference/` corpus publishes an accepted
same-scope PF-1000 bulk-pinch Te or Ti history for the Akel 16 kV shot set.

The closest candidates are:
- **Bernard 1977** — wrong-scope filament-phase soft-X-ray derived
  temperatures; not bulk-pinch; not PF-1000; not the Akel shot set.
- **Plasma Focus Update 2021 / Gribkov** — PF-1000U local-hotspot
  filter-ratio temperatures; local hotspot, not whole-column bulk-pinch
  temperature history; PF-1000U device, not PF-1000 Akel 16 kV campaign.

Neither source satisfies the same-scope requirement. The absence of accepted
same-scope bulk-pinch Te/Ti history is a field-wide measurement gap, not a
local data-availability issue.

### 1.3 Certificate is permanently unissuable while Te/Ti is hardcoded blocking

Because `BLOCKING_SAME_SCOPE_CHANNELS` is unconditionally unioned into the
missing set at `same_scope.py:143`, and because no accepted same-scope PF-1000
bulk-pinch Te/Ti history exists, the same-scope packet can never exit blocked
status. This makes the first-principles certificate permanently unissuable
while this code structure stands.

---

## 2. What this design does NOT propose

- **No generic `caveat_accepted` state.** The Codex Sprint 5 WS2 audit
  explicitly directed: "do not add a generic `caveat_accepted` Te/Ti state."
  This design does not add such a state, does not create a lane that accepts
  Te/Ti via a blanket caveat, and does not widen the same-scope gate to
  accept any channel that lacks a per-channel exclusion record plus
  certificate-text.

- **No lifting of the same-scope comparator gate.** The gate structure in
  `build_same_scope_source_packet` at `same_scope.py:117-174` is not widened.
  The function's fail-closed posture and its policy mapping at `same_scope.py:167-174`
  remain unchanged.

- **No Te/Ti counted as same-scope comparator evidence.** A channel in
  `observable_excluded_not_validated` state is removed from the blocking set
  only for the purpose of computing whether the certificate can be issued. It
  is NOT added to `accepted_same_scope_channels` and NOT treated as
  comparator evidence.

---

## 3. What this design DOES propose

A new explicit per-channel state `observable_excluded_not_validated` that:

(a) records the channel name, the scope to which the exclusion applies, the
    certificate text section that documents the exclusion, and the reason
    (field-wide measurement absence);

(b) requires the calling code to supply a matching exclusion record for each
    excluded channel — there is no implicit exclusion;

(c) requires the certificate manifest to carry matching exclusion text (a
    specific `certificate_section_id`), not just an exclusion record in the
    packet;

(d) requires a reviewer sign-off flag on each exclusion record
    (`reviewer_signoff_required=True`) — the sign-off must be present;

(e) removes the excluded channel from the blocking set **only if** all of
    (b), (c), and (d) are satisfied simultaneously; if any is absent the
    channel reverts to blocking;

(f) leaves `accepted_runtime_claim=false` and
    `can_support_first_principles_acceptance=false` — the same-scope gate
    advancing past this blocker is a necessary but not sufficient condition
    for first-principles acceptance.

---

## 4. Proposed schema

### 4.1 `ObservableExclusion` dataclass

New dataclass in `src/dpf/first_principles/same_scope.py`:

```python
@dataclass(frozen=True)
class ObservableExclusion:
    channel_name: str
    scope: str
    certificate_section_id: str
    exclusion_reason: str
    reviewer_signoff_required: bool = True
```

All fields are mandatory. `reviewer_signoff_required` must be `True`; a
value of `False` is a schema error and the gate must reject it.

### 4.2 `OBSERVABLE_EXCLUSION_ELIGIBLE_CHANNELS` constant

New constant in `src/dpf/first_principles/same_scope.py`:

```python
OBSERVABLE_EXCLUSION_ELIGIBLE_CHANNELS: frozenset[str] = frozenset({
    "electron_temperature_history",
    "ion_temperature_or_distribution_history",
})
```

This is a strict allow-list. No channel outside this set may ever be placed
in `observable_excluded_not_validated` state via this mechanism. The gate
must reject any exclusion record whose `channel_name` is not in this set with
an explicit error: `channel_not_eligible_for_exclusion:<channel_name>`.

The allow-list is currently scoped to the PF-1000 full-energy shot set. If a
future scope requires a different exclusion list, a separate allow-list
constant must be defined for that scope — the existing constant must not be
widened without a new audit review.

### 4.3 New packet field: `excluded_observables`

New field on the same-scope packet returned by `build_same_scope_source_packet`:

```python
"excluded_observables": List[ObservableExclusion]
```

This field is populated only when the calling code supplies validated exclusion
records. Its presence in the packet output is informational; it does not itself
change the gate decision — the gate decision is determined by whether the records
satisfy all four conditions in §3.

### 4.4 Gate mutation logic (proposed replacement for `same_scope.py:143`)

The unconditional `missing.update(BLOCKING_SAME_SCOPE_CHANNELS)` at
`same_scope.py:143` is replaced by a guarded update:

```python
effective_blocking = set(BLOCKING_SAME_SCOPE_CHANNELS)
validated_exclusions: list[ObservableExclusion] = []

for excl in supplied_exclusions:  # caller-supplied list
    if excl.channel_name not in OBSERVABLE_EXCLUSION_ELIGIBLE_CHANNELS:
        # Non-allowlisted channel: add explicit error blocker, do not exclude
        missing.add(f"non_eligible_exclusion_attempt:{excl.channel_name}")
        continue
    if not excl.reviewer_signoff_required:
        missing.add(f"exclusion_signoff_missing:{excl.channel_name}")
        continue
    if not _certificate_carries_exclusion_text(certificate_manifest, excl):
        # Certificate text not found: channel stays blocking
        continue
    # All conditions met: remove from effective blocking
    effective_blocking.discard(excl.channel_name)
    validated_exclusions.append(excl)

missing.update(effective_blocking)
```

`_certificate_carries_exclusion_text` is a new private helper that checks
whether the certificate manifest mapping contains a key matching
`excl.certificate_section_id` with a non-empty value. If the manifest is
absent or empty, the check fails for all exclusions.

The function signature of `build_same_scope_source_packet` gains two new
keyword-only parameters:

```python
def build_same_scope_source_packet(
    *,
    declared_scope: str,
    device_name: str | None = None,
    validation_targets: ...,
    accepted_same_scope_channels: ...,
    observable_exclusions: tuple[ObservableExclusion, ...] | list[ObservableExclusion] = (),
    certificate_manifest: Mapping[str, object] | None = None,
) -> dict[str, Any]:
```

The default values of `()` and `None` preserve backward compatibility: all
existing call sites that do not pass exclusions continue to receive the same
fully-blocking output.

---

## 5. Per-channel exclusion records needed

These are the two exclusion records that must be supplied by the calling code
for the PF-1000 full-energy shot set scope:

### 5.1 `electron_temperature_history`

```python
ObservableExclusion(
    channel_name="electron_temperature_history",
    scope="pf1000_full_energy_27_to_40_kv_gribkov_scholz_era",
    certificate_section_id="te_observable_excluded_validation_section",
    exclusion_reason=(
        "no_accepted_same_scope_pf1000_bulk_pinch_te_history_field_wide_absence"
    ),
    reviewer_signoff_required=True,
)
```

The `exclusion_reason` value encodes the field-wide measurement gap diagnosis:
no published DPF paper provides an accepted same-scope PF-1000 bulk-pinch
electron temperature history for the Akel 16 kV / Gribkov full-energy shot
sets. Bernard 1977 is filament-phase wrong-scope; Plasma Focus Update 2021
is local-hotspot PF-1000U, not bulk-pinch PF-1000 Akel.

### 5.2 `ion_temperature_or_distribution_history`

```python
ObservableExclusion(
    channel_name="ion_temperature_or_distribution_history",
    scope="pf1000_full_energy_27_to_40_kv_gribkov_scholz_era",
    certificate_section_id="ti_observable_excluded_validation_section",
    exclusion_reason=(
        "no_accepted_same_scope_pf1000_bulk_pinch_ti_history_field_wide_absence"
    ),
    reviewer_signoff_required=True,
)
```

Same reasoning as Te: no accepted same-scope bulk-pinch Ti or ion distribution
function history exists for the PF-1000 Akel or Gribkov full-energy shot sets.

---

## 6. Test pre-conditions

The following test cases must be defined before implementation. They serve as
regression anchors to prevent accidental gate widening.

### 6.1 Regression: missing exclusion records still block

A `build_same_scope_source_packet` call with no `observable_exclusions` and
no `certificate_manifest` must produce a packet where:
- `"electron_temperature_history"` is in `missing_acceptance_channels`
- `"ion_temperature_or_distribution_history"` is in `missing_acceptance_channels`
- `status == "blocked_same_scope_source_packet_not_available"`

This proves the unconditional blocking of Te/Ti is not accidentally removed.

### 6.2 Regression: exclusion records without certificate text still block

A call with correct `ObservableExclusion` records for Te and Ti but with
`certificate_manifest=None` or `certificate_manifest={}` must still produce
a packet where both channels are in `missing_acceptance_channels`.

### 6.3 Regression: exclusion records + certificate text may advance the gate

A call with correct `ObservableExclusion` records for Te and Ti AND a
`certificate_manifest` that contains non-empty values for both
`"te_observable_excluded_validation_section"` and
`"ti_observable_excluded_validation_section"`, with `reviewer_signoff_required=True`
on both records, MAY produce a packet where Te and Ti are NOT in
`missing_acceptance_channels`.

This test asserts only that those two channels are no longer blocking — it must
NOT assert that `can_support_first_principles_acceptance=True`. The overall
same-scope gate result remains blocked if any other channel in
`BLOCKING_SAME_SCOPE_CHANNELS` is still missing.

### 6.4 Regression: non-allowlisted channel cannot be excluded

A call with an `ObservableExclusion` record for
`channel_name="accepted_digitized_current_waveform"` must produce a packet
that includes a blocker of the form
`"non_eligible_exclusion_attempt:accepted_digitized_current_waveform"` and
must NOT remove that channel from the blocking set.

### 6.5 Regression: exclusion mechanism cannot set `can_support_first_principles_acceptance=True`

No combination of valid exclusion records and certificate manifest may cause
`build_same_scope_source_packet` to return a packet with
`can_support_first_principles_acceptance: True`. That flag is set only by the
upstream first-principles readiness gate after all evidence is assembled.

### 6.6 Regression: reviewer_signoff_required=False is rejected

An `ObservableExclusion` record with `reviewer_signoff_required=False` must
produce a `"exclusion_signoff_missing:<channel>"` blocker and must NOT remove
the channel from the blocking set.

---

## 7. Implementation deferral

- This design memo is committed in Sprint 6 WS5 (2026-05-20).
- The implementation — `ObservableExclusion` dataclass, `OBSERVABLE_EXCLUSION_ELIGIBLE_CHANNELS`,
  modified `build_same_scope_source_packet` signature, `_certificate_carries_exclusion_text`,
  and the test module — is **Sprint 7+ work**.
- No acceptance flag changes in this sprint.
- No certificate gate widening in this sprint.
- The `BLOCKING_SAME_SCOPE_CHANNELS` tuple at `same_scope.py:72-86` is not
  modified in this sprint. The implementation sprint modifies only the
  runtime gate logic that consumes it.
- The Codex audit of this memo and the companion Package-Native 3D contract
  design (Sprint 6 WS5) must occur at the same commit before either
  implementation sprint begins.

---

## 8. Ledger traceability

No gate ledger row currently exists specifically for the Te/Ti observable
exclusion mechanism. The Codex audit of this memo should create one with:

```
channel_te_ti_observable_exclusion,same_scope_gate,
TE-TI-OBSERVABLE-EXCLUSION-DESIGN,
explicit per-channel exclusion with certificate text and reviewer signoff,
src/dpf/first_principles/same_scope.py,
regression tests §6.1-§6.6,
no_generic_caveat_accepted_lane;no_te_ti_as_comparator_evidence,
exclusion records must carry certificate_section_id,
true,true,true,blocked_not_accepted,false,
design memo SPRINT6_WS5_CLAIM_LIMITED_TE_TI_OBSERVABLE_EXCLUSION_DESIGN_2026_05_20.md
```

Affected source file (read-only in this sprint):
- `src/dpf/first_principles/same_scope.py` (BLOCKING_SAME_SCOPE_CHANNELS lines 72-86, gate logic lines 117-174)
