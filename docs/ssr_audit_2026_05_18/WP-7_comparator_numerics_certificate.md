# WP-7 Audit: Comparator, Numerical Fidelity, Certificate, Generalization

Date: 2026-05-18
Auditor scope: SSR-010, SSR-011, SSR-012, SSR-013 (WP-7).
Repo: `/Users/anthonyzamora/dpf-unified` — branch `codex/corpus` @ `1acc886`.
Runtime: `.venv312/bin/python` 3.12.13.

Audit basis: read-only inspection of 13 in-scope modules plus
`runner.py` wiring (lines 1000-1210, 2040-2141). Every cited
`KnowledgeReference/` line range below was opened and confirmed.

Files audited (read-only, none modified):
`same_scope.py` (302), `comparator_uq.py` (351), `waveform_phase.py` (297),
`spatial_field_temperature.py` (302), `current_waveform_comparator.py` (328),
`source_targets.py` (1265), `numerical_fidelity.py` (441),
`experimental_numerics.py` (905), `limiter_readiness.py` (241),
`limiter_proof.py` (283), `certificate_gate.py` (242), `manifest.py` (260),
`generalization.py` (415).

---

## (a) Verdict

`request_changes`.

The WP-7 modules are **honest and structurally fail-closed**. No overclaim
was found: every packet hardcodes `can_support_first_principles_acceptance:
False`, the certificate gate cannot return an accepted state, no experimental
waveform is used as a drive or fit, and cross-scope material cannot pass
PF-1000/Akel acceptance. The runtime is correctly an
`engineering_candidate_not_validation` artifact.

The verdict is `request_changes` (not `accept_engineering_progress`) for one
reason in WP-7 scope: **the spec-mandated negative-control test file
`tests/test_first_principles_certificate_negative_controls.py` does not
exist.** WP-7 deliverable explicitly requires "Negative controls for draft,
blocked, cross-scope, missing-UQ, missing-review, hidden-limiter,
reduced-model fallback, and app-only evidence." The Rejection Criteria state
"Tests do not include negative controls" is a rejection trigger. The packet
logic is correct; it is simply not pinned by tests. Proposed patch text is in
section (e).

This is not `reject_overclaim`: nothing claims readiness. It is not
`accept_certificate_candidate`: most upstream packets are blocked by design,
so the certificate payload is incomplete.

---

## (b) Source Evidence Table

All `KnowledgeReference/` paths confirmed present on disk; cited line ranges
opened and confirmed to support the stated role.

| Local source path:lines | Claim / role used | Implemented in | Verify result |
| --- | --- | --- | --- |
| `KnowledgeReference/radiation-physics-and-chemistry-188-2021-109633.md:108-142` | PF-1000/Akel device geometry, bank `C0=1332 uF`, 16 kV, 1.05-1.2 Torr, diagnostics | `same_scope.py`, `waveform_phase.py`, `comparator_uq.py` | CONFIRMED — lines name 480 mm electrodes, 231 mm Cu anode, twelve 8-cm cathode tubes, `C0=1332 uF`, 16 kV, 170.5 kJ, 1.05/1.2 Torr. |
| `KnowledgeReference/radiation-physics-and-chemistry-188-2021-109633.md:256-333` (same_scope cites `256-333`) | shot-12581 deck values `L0=25 nH, C0=1332 uF, r0=6.1 mOhm, b=16, a=11.55, z0=48, V0=16 kV, p0=1.2 Torr` | `same_scope.py` SAME_SCOPE_SOURCE_REFS | CONFIRMED — lines 262-270 give exactly those values for shot 12581. **Material finding:** lines 262-264 also state "The measured wave shape of the current is then used as the basis to fit the computed current" — the Akel paper itself fits a reduced (Lee) model to the measured waveform. This is the precise practice the spec forbids in first-principles paths; the codebase correctly does NOT do this (see SSR-010 below). |
| `KnowledgeReference/radiation-physics-and-chemistry-188-2021-109633.md:862-889` | scalar neutron yield + shot-to-shot uncertainty text: `(1.75 +/- 0.2)e9`, range `(3+/-0.2)e8` to `(6.1+/-0.2)e9` | `same_scope.py`, `comparator_uq.py` (cites `120-139,862-889`) | CONFIRMED — lines 866-871 give the measured-yield averages and uncertainty ranges. Role label "scalar yield / series uncertainty context" is accurate. |
| `KnowledgeReference/fully-electromagnetic-hybrid-pic-fluid-dpf-neutron-yield-acb71fa9.md:20` | "source ingestion does not make a validation target" | `certificate_gate.py` CERTIFICATE_GATE_SOURCE_REFS | CONFIRMED — line 20 verbatim: ingestion "does not convert any paper claim... into an accepted validation target until a separate typed target packet is extracted and reviewed." Strongly supports the fail-closed gate design. |
| `KnowledgeReference/anisotropy-of-the-emission-of-dd-fusion-neutrons-caused-by-the-plasma-focus-vessel-527cc533.md:121-137` | PF-1000 full-energy anisotropy is 450-500 kJ at 3.5 Torr — NOT Akel 16 kV | `same_scope.py`, `comparator_uq.py`, `generalization.py` | CONFIRMED — lines 132-137 state "discharging energy of 450-500 kJ... pressure of 3.5 Torr." The "other_scope" classification is correct: this is not the Akel 16 kV shot-12581 scope. |
| `KnowledgeReference/digitization/akel-2021-fig1-current-waveform-shot-12581-draft-packet.json` (referenced 710-787) | draft waveform packet, `review_status=draft`, not acceptance evidence | `waveform_phase.py` DRAFT_AKEL_FIG1_PACKET_STATUS | CONFIRMED — file exists (788 lines). Packet is correctly marked `review_status: "draft"`, `independent_review_count: 0`, `accepted_for_validation: False`. |
| `KnowledgeReference/...constrained-transport-embedded-boundary...md`, `particle-simulation-of-plasmas-review-and-advances-6d7355ba.md`, `2019nrlplasma-formulary-037290d4.md` | numerical-method context for shock / Yee-Courant / resistive timestep / charge conservation | `numerical_fidelity.py`, `experimental_numerics.py`, `limiter_readiness.py` | NOT line-by-line re-opened this pass — paths confirmed present; role labels are method-context only and the packets do not assert physics from them. Recommend the spatial-numerics auditor confirm line ranges. Marked `[partial-verify]`. |
| `source_targets.py` GV / May15 / May16 registries | cross-scope candidate decks (GV workbooks, theses) | `source_targets.py` | CONFIRMED behavior: every entry carries `accepted_for_validation: False` / `accepted_for_whole_shot_first_principles: False`. KR record paths are referenced but not re-hashed this pass `[partial-verify]`. No fabricated KR citation found; the GV root is a local Downloads path explicitly flagged `user_verified_local_download_not_knowledge_reference_promoted`. |

No fabricated, wrong, or unsupported `KnowledgeReference/` citation was found
in the WP-7 modules. Citations that were not re-opened line-by-line this pass
are flagged `[partial-verify]` and are method-context labels only (they do not
carry physics claims).

---

## (c) Per-SSR Status

### SSR-010 — Same-Scope Comparator and UQ

| Check | Status | Evidence |
| --- | --- | --- |
| Every accepted observable has a same-scope source target | PASS (vacuously) | No observable is accepted. `same_scope.py:142-146` force-adds all `BLOCKING_SAME_SCOPE_CHANNELS` to `missing`; `comparator_uq.py:158-159` force-adds all `REQUIRED_COMPARATOR_UQ_CHANNELS` to `missing`. Acceptance is structurally unreachable. |
| Units / coord system / time origin / interpolation / metric / tolerance recorded per observable | PARTIAL | `current_waveform_comparator.py:136-147` records unit conversion (`A*1e-3 -> kA`), time-alignment policy ("absolute workbook time... no phase shift"), output mapping. But metric/tolerance/coordinate-system are listed only as REQUIRED channels in `comparator_uq.py` — they are declared as gates, not yet bound. Acceptable for a blocked packet. |
| Measurement + model + numerical uncertainty present | PASS (as required channels) | `comparator_uq.py:78-81` lists `measurement_uncertainty_by_observable`, `model_uncertainty_by_observable`, `numerical_uncertainty_by_observable`, `closure_sensitivity_uncertainty`, `detector_response_uncertainty` — all in `REQUIRED_COMPARATOR_UQ_CHANNELS`, all forced into `missing`. |
| Pass/fail rule per observable | PASS (as required channel) | `comparator_uq.py:86` `pass_fail_rule_by_observable` required + missing. |
| **CRITICAL: no experimental waveform used as drive or fit while claiming prediction** | **PASS** | `current_waveform_comparator.py:185-190` and `:255-260` both hardcode `experimental_waveform_used_as_drive: False`, `experimental_waveform_used_as_fit: False`, `reduced_model_used: False`, `comparison_changes_solver_state: False`. The comparator only `np.interp`s the target onto sim times and computes residuals (`:109-184`); it returns telemetry and never mutates solver state. The GV target is loaded read-only via `extract_gv_current_waveform_packet`. |
| **CRITICAL: cross-scope material cannot pass PF-1000/Akel acceptance without a transfer rule** | **PASS** | `same_scope.py:265-290` `_target_scope_matches` requires the target's `declared_scope`/`validation_scope`/`scope` to normalize-equal the run scope, OR a `source_reference` whose text contains `akel` AND (`12581`/`16kv`/`16_kv`). Mismatched targets get `decision: rejected_missing_or_mismatched_scope_metadata` (`:230-232`). `OTHER_SCOPE_SOURCE_GROUPS` (`:98-114`) carries explicit `scope_mismatch` strings; `cross_scope_policy` (`:175-180`) sets `can_use_other_scope_for_acceptance: False` and demands `TRANSFER_RULE_REQUIRED_CHANNELS`. Same pattern in `comparator_uq.py`, `waveform_phase.py`, `spatial_field_temperature.py`. |

SSR-010 result: **honest and fail-closed.** The single most important
Rejection-Criteria check — experimental waveform as drive/fit — passes
unambiguously. Note the irony confirmed against KR: the Akel 2021 paper
(`rad-phys...109633.md:262-264`) itself fits the Lee model to the measured
waveform; the WP-7 comparator is explicitly built to NOT replicate that, and
succeeds.

Gap: SSR-010 acceptance is unreachable by construction, which is correct for
the current blocker state, but means there is no positive-path test of the
`_target_scope_matches` accept branch. A negative control that supplies a
cross-scope target and asserts rejection is required (section e).

### SSR-011 — Numerical Fidelity

| Required item | Status | Evidence |
| --- | --- | --- |
| Timestep convergence family | PRESENT as gate + probe | `numerical_fidelity.py:71` `timestep_family_definitions` required+missing. `experimental_numerics.py:221-274` `build_experimental_numerical_family_packet` accepts `family_kind` + `case_payloads`, summarizes per-case observables and `pairwise_comparisons` with `tolerance_claim: False`, `convergence_decision.status: "not_assessed_no_accepted_tolerances"`. Family is a probe, not an accepted convergence claim. |
| Mesh convergence family | PRESENT as gate + probe | Same `build_experimental_numerical_family_packet`; `mesh_timestep_convergence` block in runtime audit (`:182-191`) is `available: False`, `status: missing_mesh_timestep_convergence_family`. |
| Restart reproducibility | PRESENT as gate + probe | `experimental_numerics.py:277-392` `build_experimental_reproducibility_packet` hashes a stable payload over terminal scalars/history/last-step/conservation, sets `all_state_observable_hashes_identical`, but `checkpoint_restart.available: False`. Decision: `deterministic_rerun_matched_not_restart_acceptance` — explicitly distinguishes a deterministic rerun from accepted checkpoint/restart. |
| Split-continuation equivalence | PRESENT as gate | `experimental_numerics.py:343-351` `continued_run_equivalence` `available: False`, `status: missing_split_run_continuation_packet`. |
| Checkpoint/restart equivalence | PRESENT as gate | `:333-342` `checkpoint_restart` `available: False`. |
| Backend / precision declaration | PRESENT as gate | `numerical_fidelity.py:81` `backend_precision_parity_packet`; `:58` `backend_and_precision_parity` test surface. `manifest.py:48` records `backend` field. |
| Divergence budgets | PRESENT as gate + runtime obs | `numerical_fidelity.py:73-74` `divergence_b_packet`, `gauss_law_or_continuity_packet`. `experimental_numerics.py:685-698` `_divergence_snapshot` reports `final_max_abs_div_B_T_per_m` with `tolerance_claim: False`, `requires_gauss_or_divergence_nondominance: True`. |
| Limiter-zero OR physically-bounded limiter proof | PRESENT, fail-closed | `limiter_proof.py:45-134` `build_experimental_limiter_zero_probe_packet` classifies `ACCEPTANCE_BLOCKING_LIMITER_COUNTS` vs `METHOD_REVIEW_LIMITER_COUNTS`, computes `zero_acceptance_blockers_observed` only when inventory complete + zero blocking + finite + target satisfied — and STILL returns `can_support_limiter_zero_acceptance: False`. `limiter_readiness.py` requires a 19-channel inventory. |
| Artifact hashes | PRESENT | `manifest.py:171-185` `stable_manifest_hash` (deterministic SHA-256 over sorted JSON); `experimental_numerics.py:556-564` `_stable_payload_hash`. |
| **Convergence claims have FAMILIES, not single runs** | **PASS** | `build_experimental_numerical_family_packet` takes a `Sequence` of `case_payloads`; `convergence_decision` stays `not_assessed` until accepted tolerances + >= 3 ordered levels exist (`next_required_actions:259-265`). No single-run convergence claim exists. |
| **No unbounded memory growth** | PASS (within WP-7 scope) | The WP-7 packets retain no growing buffers; `_history_summary` consumes a list and emits scalar min/max. `runner.py` exposes `history_stride` / `max_step_results` (capping retained step payloads). |
| **No history truncation hiding cumulative ledgers** | PASS (within WP-7 scope) | `experimental_numerics.py:736-772` `_history_summary` computes `cumulative_neutrons_max`, `cumulative_neutrons_non_decreasing`, `field_energy_J` min/max from retained samples; `_runtime_horizon` (`:602-628`) separately reports `retained_step_result_count` vs `n_steps_completed` so a capped payload is visible, not hidden. Cumulative quantities are summarized as max/monotonicity rather than dropped. NOTE: full verification that the upstream simulator preserves cumulative ledgers when step payloads are capped is a WP-4 concern, outside this scope; flagged for the engine auditor. |

SSR-011 result: **fail-closed and structurally complete as a gate set.** All
nine required items are present either as required channels in
`numerical_fidelity.py` or as runtime probes in `experimental_numerics.py`.
`numerical_fidelity.py:212-213` force-adds all `REQUIRED_NUMERICAL_FIDELITY_CHANNELS`
to `missing`; `can_support_numerical_acceptance: False` is hardcoded.
Convergence is correctly family-based and unassessed.

### SSR-012 — Certificate Gate

| Check | Status | Evidence |
| --- | --- | --- |
| Lists every upstream packet + acceptance state | PASS | `certificate_gate.py:89-101` `REQUIRED_UPSTREAM_PACKET_CHANNELS` maps 11 certificate channels to packet names; `_upstream_packet_acceptance_matrix` (`:194-207`) emits per-channel `{packet, upstream_status, accepted_for_certificate, decision}`. `runner.py:1074-1092` passes 12 upstream packets in. |
| Acceptance requires ALL upstream accepted | PASS | `_status_is_accepted_for_certificate` (`:234-242`) accepts only `accepted*` / `ready` / `passed` / `reviewed_accepted`. `_status_blocks_certificate` (`:221-231`) treats `None`, anything in `BLOCKING_UPSTREAM_STATUSES`, or any `blocked*`/`candidate*`/`rejected*` prefix as blocking. `acceptance_policy` (`:147-153`) sets `all_upstream_packets_must_be_accepted: True`. |
| Required negative tests present | PASS as gate | `REQUIRED_NEGATIVE_TEST_CHANNELS` (`:103-111`) enumerates draft/blocked/cross-scope/missing-uq/missing-review/hidden-limiter/app-only-or-reduced-model. `_negative_test_matrix` (`:210-218`) marks each `missing_required_negative_test` unless its channel is in `accepted`. |
| Manifest has command provenance, versions, artifact hashes, source-packet hashes, review metadata | PARTIAL | `manifest.py` records `manifest_version`, `run_id`, `created_utc`, `backend`, `runtime` (python/platform/machine via `_runtime_profile`), `source_index_references`, `inputs`/`outputs` artifacts with optional `sha256`, and `manifest_sha256`. **Gap:** there is no explicit `command`/`argv` provenance field and no `source_packet_hash` field distinct from `source_index_references`. `certificate_gate.py` requires `commands_and_versions`, `evidence_packet_hashes`, `source_packet_hashes` as channels — but `manifest.py` does not populate command-line or per-source-packet hashes. Acceptable while blocked, but flagged: SSR-012 expects the manifest to carry command provenance. |
| **CRITICAL: certificate cannot return `accepted_first_principles` while ANY upstream blocked** | **PASS** | `certificate_gate.py:114-164` has NO accepting branch. `status` is the constant string `blocked_first_principles_certificate_not_available`. `can_write_accepted_certificate`, `can_release_first_principles_claim`, `can_support_first_principles_acceptance` are all hardcoded `False`. `release_label` is constant `engineering_candidate_not_releasable_for_first_principles_claim`. There is no code path, parameter, or upstream combination that flips any of these. Verified by `grep`: the only repo occurrence of `accepted_first_principles` in `certificate_gate.py` is line 137 inside the decision string `do_not_write_accepted_first_principles_certificate`. |
| **CRITICAL: no doc/README/UI claims readiness while gate blocked** | **PASS within WP-7 scope** | `runner.py:2054` `_first_principles_candidate_packet` returns `status: "not_validation"`, `scientific_status: ENGINEERING_CANDIDATE_STATUS`, `can_support_first_principles_acceptance: False`, and an 11-item `blocking_reasons` list. `manifest.py:66-74` `__post_init__` RAISES `ValueError` if `run_status`, `validation_status`, or `artifact_status` are anything other than the engineering-candidate constants, or if `can_support_first_principles_acceptance` is True — the manifest physically cannot be constructed in an accepted state. Status/blockers docs (`STATUS_BLOCKERS_2026_05_18.md`) describe everything as blocked. (A full repo-wide README/UI sweep is outside WP-7 file scope; Audit Phase 3 `rg` should confirm.) |

SSR-012 result: **the certificate gate is honest and cannot be made to lie.**
The strongest evidence is structural: `certificate_gate.py` has no accepting
return path at all, and `manifest.py.__post_init__` raises on any non-candidate
status. This satisfies the Audit-Phase-7 hard requirement.

### SSR-013 — Generalization

| Check | Status | Evidence |
| --- | --- | --- |
| General-DPF claim rejected when only PF-1000/Akel exercised | PASS | `generalization.py:302-335` `status` is constant `blocked_generalized_dpf_machine_path_not_available`; `can_claim_generalized_dpf_machine: False`, `can_support_first_principles_acceptance: False` hardcoded. `claim_policy` (`:315-321`) sets `single_scope_is_not_generalized: True`. `:293-294` force-adds all `REQUIRED_GENERALIZATION_CHANNELS` to `missing`. |
| GV / May15 / May16 / Soto / IPFS stay candidate | PASS | `CANDIDATE_SECOND_SCOPES` (`:118-251`) lists Soto2010/CCHEN, PF-1000 full-energy, FAETON-I, IR-MPF-100, compact Chinese DPF, LLNL-180kA, MJOLNIR, PF-1000/Akel other-shot — every entry `source_status` is `candidate_requirement_material_not_acceptance` or `user_validated_source_target_material_not_acceptance`. `_candidate_second_scope_decision` (`:406-415`) stamps `decision: candidate_requirement_material_not_acceptance`, `must_write_independent_certificate: True`, and requires `SECOND_SCOPE_REQUIRED_GATE_IDS` FP-1..FP-14. `source_targets.py` GV/May registries independently carry `accepted_for_validation: False`. |
| Second scope needs independent full evidence path | PASS | `REQUIRED_GENERALIZATION_CHANNELS` (`:95-116`) enumerates `second_scope_*` for geometry/drive/startup/power-port/dimensionality/closure/density-field-temp/neutron/detector-UQ/comparator-UQ/numerical-fidelity/certificate plus `no_hidden_pf1000_akel_assumptions` and `cross_scope_negative_tests`. `_upstream_acceptance_gate` (`:365-380`) blocks generalization until the primary-scope certificate is accepted. |

SSR-013 result: **fail-closed and correct.** A general-DPF claim is
unreachable; PF-1000/Akel-only exercise cannot promote it; all named
second-scope devices are candidate-only.

---

## (d) Certificate Upstream-Acceptance-Matrix Check

`certificate_gate.py` builds `upstream_packet_acceptance_matrix` from the 11
entries in `REQUIRED_UPSTREAM_PACKET_CHANNELS`. The runner passes 12 upstream
packets (`runner.py:1078-1091`). Cross-check:

| Certificate channel | Required packet | Runner supplies? | Matrix decision when blocked |
| --- | --- | --- | --- |
| same_scope_source_packet_accepted | `same_scope_source` | yes | `missing_or_blocking_upstream_packet` |
| waveform_phase_packet_accepted | `waveform_phase` | yes | `missing_or_blocking_upstream_packet` |
| spatial_field_temperature_packet_accepted | `spatial_field_temperature` | yes | `missing_or_blocking_upstream_packet` |
| neutron_authority_packet_accepted | `neutron_authority` | yes | `missing_or_blocking_upstream_packet` |
| comparator_uq_packet_accepted | `comparator_uq` | yes | `missing_or_blocking_upstream_packet` |
| numerical_fidelity_packet_accepted | `numerical_fidelity` | yes | `missing_or_blocking_upstream_packet` |
| physics_closure_packet_accepted | `physics_closure` | yes | `missing_or_blocking_upstream_packet` |
| limiter_zero_or_physical_bounds_packet | `limiter_readiness` | yes | `missing_or_blocking_upstream_packet` |
| power_port_packet_accepted | `power_port` | yes | `missing_or_blocking_upstream_packet` |
| startup_packet_accepted | `startup_bvp` | yes | `missing_or_blocking_upstream_packet` |
| dimensionality_handoff_packet_accepted | `dimensionality_handoff` | yes | `missing_or_blocking_upstream_packet` |

Findings:

1. **Coverage gap (minor, honest-direction):** the runner also passes
   `experimental_limiter_zero_probe` as an upstream packet, but it is NOT a
   key in `REQUIRED_UPSTREAM_PACKET_CHANNELS`, so it does not appear in the
   acceptance matrix. The limiter-zero probe is instead consumed indirectly
   via `limiter_readiness`. This is acceptable (the probe is a runtime
   observation, not an acceptance packet) but worth noting: the matrix keys
   on `limiter_readiness` only. No overclaim — the omitted packet would only
   ever ADD blockers, never remove them.

2. **`engineering_current_waveform_comparison` is NOT in the certificate
   upstream set.** It is passed to `comparator_uq` (`runner.py:1069-1071`)
   but the certificate keys on `comparator_uq`, which transitively depends on
   it. Acceptable: the comparator-UQ packet is the certificate's single
   waveform-comparison dependency.

3. **The matrix cannot produce a false `accepted`.**
   `_status_is_accepted_for_certificate` requires an `accepted*`/`ready`/
   `passed`/`reviewed_accepted` status. Every WP-7 upstream packet returns a
   hardcoded `blocked_*` or `candidate_*` status string, so every matrix row
   resolves to `accepted_for_certificate: False`. Confirmed: there is no
   parameter to `build_first_principles_certificate_gate_packet` that can set
   an upstream status to accepted — statuses come only from the upstream
   packets themselves, none of which can be accepted.

4. **Negative-test matrix is keyed on `accepted_channels`** (deck-supplied).
   `_negative_test_matrix` marks all 7 negative-test channels
   `missing_required_negative_test` unless the deck lists them in
   `certificate_accepted_channels`. Since no deck does, all 7 are reported
   missing — correct fail-closed behavior, and it is the certificate itself
   declaring the negative-test gap.

Conclusion for (d): the acceptance matrix is correctly wired, fail-closed, and
cannot return `accepted_first_principles` while any upstream packet is blocked.
The certificate-gate honesty requirement is **satisfied**.

---

## (e) Proposed Patch Text — Negative Controls (DO NOT APPLY)

The spec requires `tests/test_first_principles_certificate_negative_controls.py`
with concrete pytest functions for: draft, blocked, cross-scope, missing-UQ,
missing-review, hidden-limiter, reduced-model-fallback, app-only evidence.
The file does not exist (`ls tests/` confirms). Proposed content below — author
only; not applied per audit rules. Tolerances/targets are NOT invented: every
assertion checks an existing hardcoded fail-closed field, not a numeric bound.

```python
# tests/test_first_principles_certificate_negative_controls.py
"""WP-7 negative controls: the first-principles certificate and its WP-7
upstream packets must fail closed for draft, blocked, cross-scope, missing-UQ,
missing-review, hidden-limiter, reduced-model-fallback, and app-only evidence.

These tests assert ONLY existing fail-closed contract fields. They invent no
tolerance and no acceptance threshold. If any test starts failing, a packet
has gained an accepting path and must be re-audited against SSR-010/011/012.
"""

from __future__ import annotations

from dpf.first_principles.certificate_gate import (
    build_first_principles_certificate_gate_packet,
)
from dpf.first_principles.comparator_uq import build_comparator_uq_packet
from dpf.first_principles.numerical_fidelity import build_numerical_fidelity_packet
from dpf.first_principles.same_scope import build_same_scope_source_packet
from dpf.first_principles.generalization import build_generalized_dpf_machine_packet
from dpf.first_principles.waveform_phase import build_waveform_phase_packet

PF1000_SCOPE = "pf1000_akel_16kv_1p2torr_shot_12581"
PF1000_DEVICE = "PF-1000/Akel"


def _accepted_upstream() -> dict[str, dict[str, str]]:
    """An all-accepted upstream set used to prove the gate STILL fails closed
    even if every dependency were (hypothetically) accepted, because the
    certificate channels themselves remain unfilled."""
    names = (
        "startup_bvp", "limiter_readiness", "power_port",
        "dimensionality_handoff", "physics_closure", "same_scope_source",
        "waveform_phase", "spatial_field_temperature", "neutron_authority",
        "comparator_uq", "numerical_fidelity",
    )
    return {name: {"status": "accepted_engineering_review"} for name in names}


def test_certificate_blocked_when_any_upstream_blocked() -> None:
    """blocked control: one blocked upstream packet must block the certificate
    and must appear as a certificate blocker."""
    upstream = _accepted_upstream()
    upstream["power_port"] = {"status": "blocked_power_port_not_available"}
    packet = build_first_principles_certificate_gate_packet(
        declared_scope=PF1000_SCOPE,
        device_name=PF1000_DEVICE,
        upstream_packets=upstream,
    )
    assert packet["status"] == "blocked_first_principles_certificate_not_available"
    assert packet["can_write_accepted_certificate"] is False
    assert packet["can_release_first_principles_claim"] is False
    assert packet["can_support_first_principles_acceptance"] is False
    assert "power_port" in packet["upstream_certificate_blockers"]
    matrix = packet["upstream_packet_acceptance_matrix"]
    assert matrix["power_port_packet_accepted"]["accepted_for_certificate"] is False


def test_certificate_blocked_with_no_upstream_packets() -> None:
    """draft/empty control: with no upstream packets the certificate must not
    release and the release label must remain engineering-candidate."""
    packet = build_first_principles_certificate_gate_packet(
        declared_scope=PF1000_SCOPE, device_name=PF1000_DEVICE,
    )
    assert packet["release_decision"] == "do_not_release_first_principles_claim"
    assert packet["release_label"].startswith("engineering_candidate")
    assert packet["can_support_first_principles_acceptance"] is False


def test_certificate_cannot_accept_even_with_all_upstream_accepted() -> None:
    """Structural control: even an all-accepted upstream set cannot flip the
    certificate, because the required certificate channels are unfilled."""
    packet = build_first_principles_certificate_gate_packet(
        declared_scope=PF1000_SCOPE, device_name=PF1000_DEVICE,
        upstream_packets=_accepted_upstream(),
    )
    assert packet["status"] == "blocked_first_principles_certificate_not_available"
    assert packet["can_write_accepted_certificate"] is False
    assert packet["missing_acceptance_channels"], "channels must remain missing"


def test_certificate_negative_test_channels_required_and_missing() -> None:
    """missing-negative-test control: all 7 required negative-test channels
    must report missing when the deck declares none."""
    packet = build_first_principles_certificate_gate_packet(
        declared_scope=PF1000_SCOPE, device_name=PF1000_DEVICE,
        upstream_packets=_accepted_upstream(),
    )
    matrix = packet["negative_test_matrix"]
    for channel in (
        "negative_test_draft_evidence",
        "negative_test_blocked_evidence",
        "negative_test_cross_scope_evidence",
        "negative_test_missing_uq",
        "negative_test_missing_review",
        "negative_test_hidden_limiter",
        "negative_test_app_only_or_reduced_model_fallback",
    ):
        assert matrix[channel]["present"] is False
        assert matrix[channel]["decision"] == "missing_required_negative_test"


def test_same_scope_rejects_cross_scope_target() -> None:
    """cross-scope control: an accepted-status target from a different scope
    must be rejected with mismatched-scope metadata and must not add a
    same-scope channel."""
    cross = {
        "name": "pf1000_full_energy_anisotropy",
        "observable": "neutron_anisotropy",
        "status": "accepted_same_scope_source",
        "declared_scope": "pf1000_full_energy_450kj_3p5torr",
    }
    packet = build_same_scope_source_packet(
        declared_scope=PF1000_SCOPE, device_name=PF1000_DEVICE,
        validation_targets=(cross,),
    )
    decisions = packet["validation_target_scope_decisions"]
    assert any(
        d["decision"] == "rejected_missing_or_mismatched_scope_metadata"
        for d in decisions
    )
    assert "neutron_anisotropy" not in packet["accepted_same_scope_channels"]
    assert packet["can_support_first_principles_acceptance"] is False


def test_comparator_uq_blocked_when_missing_uq_channels() -> None:
    """missing-UQ control: with no accepted UQ channels every UQ channel must
    be missing and comparator acceptance must be blocked."""
    packet = build_comparator_uq_packet(
        declared_scope=PF1000_SCOPE, device_name=PF1000_DEVICE,
    )
    missing = set(packet["missing_acceptance_channels"])
    for channel in (
        "measurement_uncertainty_by_observable",
        "model_uncertainty_by_observable",
        "numerical_uncertainty_by_observable",
        "uq_propagation_method",
        "independent_review_certificate",
    ):
        assert channel in missing
    assert packet["can_support_comparator_acceptance"] is False
    assert packet["can_support_first_principles_acceptance"] is False


def test_waveform_phase_blocked_when_missing_review() -> None:
    """missing-review control: the draft Akel Fig.1 packet must stay
    review_status=draft and the waveform-phase packet must not accept."""
    packet = build_waveform_phase_packet(
        declared_scope=PF1000_SCOPE, device_name=PF1000_DEVICE,
    )
    draft = packet["draft_digitization_packet_status"]
    assert draft["review_status"] == "draft"
    assert draft["independent_review_count"] == 0
    assert draft["accepted_for_validation"] is False
    assert "independent_review_accepted" in packet["missing_acceptance_channels"]
    assert packet["can_support_first_principles_acceptance"] is False


def test_numerical_fidelity_hidden_limiter_does_not_pass() -> None:
    """hidden-limiter control: a runtime limiter-zero probe that observed zero
    blockers must NOT promote the numerical limiter-zero surface to accepted."""
    upstream = {
        "experimental_limiter_zero_probe": {
            "status": "experimental_limiter_zero_probe_not_validation",
            "zero_acceptance_blockers_observed": True,
        },
    }
    packet = build_numerical_fidelity_packet(
        declared_scope=PF1000_SCOPE, device_name=PF1000_DEVICE,
        upstream_packets=upstream,
    )
    surface = packet["test_surface_status"]["limiter_zero_acceptance"]
    assert surface["can_support_numerical_acceptance"] is False
    assert "accepted" not in surface["status"] or "not_acceptance" in surface["status"]
    assert packet["can_support_numerical_acceptance"] is False
    assert packet["can_support_first_principles_acceptance"] is False


def test_generalization_rejects_reduced_model_and_app_only_scope() -> None:
    """reduced-model-fallback / app-only control: a general-DPF claim must be
    blocked, and candidate second scopes must stay candidate-only."""
    packet = build_generalized_dpf_machine_packet(
        declared_scope=PF1000_SCOPE, device_name=PF1000_DEVICE,
        upstream_packets={
            "certificate_gate": {
                "status": "blocked_first_principles_certificate_not_available",
            },
        },
    )
    assert packet["can_claim_generalized_dpf_machine"] is False
    assert packet["can_support_first_principles_acceptance"] is False
    for decision in packet["candidate_second_scope_decisions"]:
        assert decision["decision"] == "candidate_requirement_material_not_acceptance"
        assert decision["must_write_independent_certificate"] is True


def test_certificate_app_only_evidence_channel_missing() -> None:
    """app-only control: the certificate's package-native execution proof
    channel must be missing when no package-native proof is declared."""
    packet = build_first_principles_certificate_gate_packet(
        declared_scope=PF1000_SCOPE, device_name=PF1000_DEVICE,
        upstream_packets=_accepted_upstream(),
    )
    assert "package_native_execution_proof" in packet["missing_acceptance_channels"]
    assert "reduced_model_rejection_proof" in packet["missing_acceptance_channels"]
```

Patch note: the helper `_accepted_upstream()` is used to prove the *strongest*
property — even a hypothetically all-accepted upstream set cannot flip the
certificate, because the certificate's own required channels (negative tests,
hashes, reviewer metadata, package-native proof) remain unfilled. This is the
exact Audit-Phase-7 guarantee. If a future change makes
`test_certificate_cannot_accept_even_with_all_upstream_accepted` fail, the gate
has gained an accepting path and must be re-audited.

---

## (f) Negative Tests: Present vs Missing

Existing negative coverage found in `tests/test_first_principles_runner.py`:

| Test | Covers | WP-7 relevance |
| --- | --- | --- |
| `test_first_principles_3d_runner_rejects_invalid_step_count` | deck input validation | not WP-7 |
| `test_first_principles_3d_runner_rejects_invalid_boundary_policy` | deck input validation | not WP-7 |
| `test_first_principles_runner_rejects_seeded_startup_for_acceptance` | SSR-004 startup | upstream of WP-7, not WP-7 |
| `test_first_principles_runner_marks_pf1000_akel_same_scope_as_blocked` | SSR-010 same-scope blocked status | **partial WP-7** — asserts `same_scope` is blocked, but does not exercise cross-scope rejection, missing-UQ, or the certificate matrix. |

Missing (required by WP-7 deliverable + Rejection Criteria):

| Required negative control | Present? | Where it should live |
| --- | --- | --- |
| draft evidence rejected | NO | `test_first_principles_certificate_negative_controls.py` |
| blocked upstream blocks certificate | NO | same |
| cross-scope target rejected by comparator | NO | same |
| missing-UQ blocks comparator-UQ | NO | same |
| missing-review blocks waveform-phase | NO | same |
| hidden-limiter does not promote numerical surface | NO | same |
| reduced-model fallback blocks generalization | NO | same |
| app-only evidence channel missing in certificate | NO | same |
| certificate cannot accept with all-upstream-accepted (structural) | NO | same |

`manifest.py.__post_init__` raising `ValueError` on non-candidate status is an
*implicit* negative control (a manifest cannot be built in an accepted state),
but it is not exercised by a dedicated test in WP-7 scope. Recommend a test
`test_first_principles_manifest_rejects_accepted_status` asserting the
`ValueError` (`test_first_principles_manifest.py` exists and is the right home).

---

## (g) Remaining Blockers

WP-7-scope blockers (preventing `accept_engineering_progress` -> stronger):

1. **B-1 (test gap, SSR-012/Rejection Criteria):**
   `tests/test_first_principles_certificate_negative_controls.py` does not
   exist. Eight spec-named negative controls plus the structural
   "cannot-accept" test are absent. Patch text supplied in section (e).
   Severity: high — this alone forces `request_changes`.

2. **B-2 (manifest provenance, SSR-012):** `manifest.py` does not record
   command-line / `argv` provenance and has no `source_packet_hashes` field
   distinct from `source_index_references`. `certificate_gate.py` lists
   `commands_and_versions`, `evidence_packet_hashes`, `source_packet_hashes`
   as required channels but nothing populates the command-line portion.
   Severity: medium — acceptable while blocked, but must be closed before any
   certificate-candidate verdict.

3. **B-3 (untested accept path):** because every WP-7 packet is hardcoded
   fail-closed, the *accept* branches of `_target_scope_matches` and
   `_accepted_channels_from_targets` are never exercised. There is no
   regression protection on the logic that WOULD accept a same-scope target.
   The cross-scope negative control in (e) exercises the reject branch only.
   Severity: low — but note for whoever later wires real accepted targets.

4. **B-4 (`[partial-verify]` citations):** numerical-method KR line ranges in
   `numerical_fidelity.py` / `experimental_numerics.py` / `limiter_readiness.py`
   (constrained-transport, particle-simulation-review, NRL formulary) were
   confirmed present-on-disk but not re-opened line-by-line this pass. They
   carry method-context role labels only (no physics claim). Recommend the
   spatial-numerics auditor confirm. Severity: low.

Upstream blockers (NOT WP-7 scope, but they keep the certificate correctly
blocked — listed for completeness): startup BVP, power port, reviewed
geometry, physics closures, neutron authority all remain `blocked` /
`candidate` per `STATUS_BLOCKERS_2026_05_18.md`. The certificate gate
correctly reflects every one of these as a blocker in its acceptance matrix.

---

## Audit Bottom Line

WP-7 (SSR-010/011/012/013) is **honest engineering scaffolding that fails
closed correctly**. The two Rejection-Criteria danger checks pass cleanly:
(1) no experimental waveform is used as a drive or fit, and (2) the
certificate cannot return `accepted_first_principles` while any upstream
packet is blocked — `certificate_gate.py` has no accepting code path and
`manifest.py` raises on any non-candidate status. Cross-scope material cannot
pass PF-1000/Akel acceptance; general-DPF claims are unreachable.

The only WP-7 finding that blocks a clean pass is the **missing negative-
control test file**, which the spec explicitly requires and which the
Rejection Criteria explicitly name. Verdict: `request_changes` — author
`tests/test_first_principles_certificate_negative_controls.py` (patch text in
section e) and add manifest command provenance (B-2).
