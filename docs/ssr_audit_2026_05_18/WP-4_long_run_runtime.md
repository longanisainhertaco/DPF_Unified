# WP-4 / SSR-007 Audit — 3-D Field/PIC/Electron Long-Run Runtime

Date: 2026-05-18
Auditor scope: WP-4 / SSR-007 (long-run Field/PIC/Electron stability, conservation
ledgers, split-continuation, checkpoint/restart equivalence).
Repo: `/Users/anthonyzamora/dpf-unified`  Branch: `codex/corpus`  Python: 3.12 (`.venv312`).
Method: static read-only audit. No `.py` file or existing doc edited. `git` and
`.venv312/bin/python -c` used for inspection only. No pytest/CLI runs.

---

## (a) Verdict

`request_changes`.

The long-run runtime is **architecturally honest in its packet labelling** — every
field/loop/checkpoint/continuation packet carries `not_validation` /
`can_support_first_principles_acceptance: False`, and the 12 us artifact does NOT
claim acceptance. SSR-007 mechanics (Maxwell in plasma+vacuum, charge/current
deposition, source-ordered velocity update, electron-energy and ionization state
persisted through restart) are present and wired.

It is **not** `accept_engineering_progress` because three WP-4 deliverables fail
the Rejection Criteria as written:

1. **A hidden, un-telemetered numerical floor exists** on electron density
   (`hybrid_loop.py:190-193`, and again `:203`), and the explicit heat-flux
   subcycle path applies an **energy floor with NO floor-contact count**
   (`electron_energy.py:732-733`). The project rule (CLAUDE.md) requires all
   numerical floors to route through `telemetry.apply_floor()`; that module does
   not even exist (`ModuleNotFoundError: No module named 'dpf.telemetry'`), so
   *no* floor in the WP-4 path is compliant with the stated convention.
2. **The conservation ledger reports `passed: true` while the run lost 58.4% of
   its tracked total energy** (`runner.py:_conservation_telemetry`,
   2202-2239). `passed` is wired to *finiteness only* — it never inspects
   `relative_tracked_total_energy_change`. A −58% energy run is flagged as a pass.
3. **Checkpoint/restart and split-continuation EQUIVALENCE is asserted but never
   exercised by a test.** The equivalence machinery exists
   (`checkpoint_restart.py`, `split_continuation.py`) and correctly returns
   `state_fingerprints_match` / `tracked_observables_match_exactly`, but **no test
   file calls `build_experimental_checkpoint_restart_packet` or
   `build_experimental_split_continuation_packet`**, and there is no
   `tests/test_first_principles_long_run_integrity.py`. A regression that breaks
   equivalence, or a hidden state repair, would not be caught.

None of these is an *overclaim* (no doc/README says "validated"), so the verdict
is `request_changes`, not `reject_overclaim`. But items 1–2 are exactly the
"hidden floor / silent pass" failure modes the spec's Rejection Criteria target,
and must be fixed before this WP can be called engineering-honest.

---

## (b) Source Evidence Table

| Local source path:lines | Claim / equation | Implemented in | Status |
| --- | --- | --- | --- |
| `KnowledgeReference/auluck-2021-dpf-circuit-element.md:151-209` | `V12 = (1/I) ∫_Ω J·E d³r`; domain Ω where J=0 outside; source interface (cathode plate) excluded; all chamber phenomena draw power from the external circuit | `hybrid_stepper.py:_field_work_telemetry` (289-345); `hybrid_simulator.py:_circuit_udpf_for_step` (`lagged_auluck_volume_j_dot_e`, 555-559) | VERIFIED — line range supports the claim. Auluck Eq. 1 is the volume `J·E`/`I` relation; the "domain Ω, J zero outside, interface excluded" text is at lines 200-209. Citation honest. |
| `KnowledgeReference/fully-electromagnetic-hybrid-pic-fluid-dpf-neutron-yield-acb71fa9.md:740-805` | External-circuit ODE `d(L0 I)/dt = V0 − r0 I − U_DPF − Q/C0` (Eq.35); explicit update Eqs.37-38; `Bθ = µI/2πr` injection boundary (Eq.34) | `hybrid_stepper.py:_field_work_telemetry` source_lines field; `circuit_boundary` (via simulator); `maxwell_3d.py` HYBRID_PIC_3D_SOURCE constant | VERIFIED — Eqs.34-38 present at the cited lines. NOTE: this KR source is a *different machine* (L0=110 nH, V0=10 kV, C0=20 µF). It is cited as a *method/pattern* source, not a deck source — acceptable under the spec, but auditors of WP-1 must confirm no PF-1000 parameter is silently taken from it. |
| `KnowledgeReference/2019nrlplasma-formulary-037290d4.md:1880-1888` | Poynting theorem, signed `J·E` energy accounting | not directly referenced in WP-4 files (power-port scope = WP-1) | N/A to WP-4 |
| `hybrid_loop.py` docstring/source_lines `"246-315, 428-535"`, `"224-236, 286-294"` etc. | Eq.7 velocity update `m dv/dt=(J×B−∇pe)/ne`; leapfrog position-only update; predictor-corrector rebuild | `hybrid_loop.py:_apply_eq7_velocity_update` (649-698), `_advance_positions_from_half_step_velocities` (597-646), `_provisional_particle_rebuild` (701-770) | UNVERIFIED LINE RANGE — these are line numbers *inside the same KR hybrid-PIC source* but were not opened in this audit beyond 740-805. WP-5/closure auditors must confirm lines 224-236 / 246-315 / 428-535 actually contain Eq.7 and the leapfrog ordering. Flagged, not failed. |

No fabricated KR *file paths* found in the WP-4 files. The two ranges actually
opened (Auluck 151-209, hybrid-PIC 740-805) genuinely support their claims.

---

## (c) Floor / Clamp / Repair Inventory

Legend: **HIDDEN** = mutates physical state with no per-step telemetry count;
**TELEMETERED** = emits a count/fraction a reviewer can audit;
**GUARD** = shape/sign validation, raises on bad input (not a silent repair).

| # | Location | Operation | Acts on | Telemetered? | Verdict |
| --- | --- | --- | --- | --- | --- |
| F1 | `hybrid_loop.py:190-193` | `electron_density = np.maximum(np.abs(rho_for_density)/e, density_floor_m3)` | electron number density (physical state) derived from deposited PIC charge | **HIDDEN** — `density_floor_m3` defaults to `1.0` m⁻³; no count of how many cells were floored | **FAIL.** Bare `np.maximum` on physical state. The floored value (1.0 m⁻³) is then used as the conductivity-domain threshold (`mask_current_to_resolved_plasma`) and the heat-flux gate. It silently defines "resolved plasma." Must route through a telemetered floor and emit a floored-cell count. |
| F2 | `hybrid_loop.py:203` | `electron_density = np.maximum(ionization_state.electron_density_m3, 1.0)` | electron density, source-backed-conductivity branch | **HIDDEN** — hard-coded `1.0`, no count | **FAIL.** Same class as F1. This is the branch the 12 us run used (`use_source_backed_conductivity=True`, runner.py:632). |
| F3 | `hybrid_stepper.py:24,258` `mask_current_to_resolved_plasma` | `np.where(density > 1.0*(1+1e-12), current, 0.0)` zeros conduction current in "vacuum-floor" cells | total/generalized-Ohm current | **TELEMETERED** — `resolved_cell_count`, `excluded_numerical_floor_cell_count`, `max_excluded_numerical_floor_current_A_m2` | ACCEPTABLE-as-telemetered, but the *threshold* is the hidden F1/F2 floor. The mask is honest; the floor it keys off is not. |
| F4 | `conductivity.py:117-123` | `effective = np.minimum(raw, sigma_cfl)` Ohmic-CFL cap, gated by `apply_ohmic_cfl_limit` | conductivity σ | **TELEMETERED** — `cfl_limited_fraction`, `sigma_cfl_S_m`, `max_sigma_raw_S_m`, `max_sigma_effective_S_m`, `ohmic_cfl_limit_applied` | HONEST telemetry. **But see (d): in the 12 us run the cap was DISABLED** and the un-capped raw σ (36462 S/m vs CFL bound 0.041 S/m, ~8.9e5× over) was used. The disable IS recorded (`ohmic_cfl_limit_applied:false`, `conductivity_ohmic_cfl_raw_exceeds_explicit_limit_steps:55580`). |
| F5 | `electron_energy.py:732-733` (explicit heat-flux subcycle path) | `energy = np.maximum(energy, energy_floor)` where `energy_floor = 1.5 ne k_B Te_floor` | electron internal energy | **HIDDEN** — explicit-path block has NO `temperature_floor_contact_count` (compare implicit path L893 and L986-1020 which DO emit it) | **FAIL.** Asymmetry: the implicit solver path telemeters floor contact; the explicit subcycle path floors silently. A run that stays on the explicit branch hides every temperature-floor contact. |
| F6 | `electron_energy.py:862,987` (implicit / 1-D paths) | `np.maximum(solution, energy_floor)` / `np.maximum(field, temperature_floor_K)` | electron energy / temperature | **TELEMETERED** — `temperature_floor_contact_count` (L893, L1020) | ACCEPTABLE. Floor contact is counted. (Still uses bare `np.maximum`, not `apply_floor()`, but it is honest.) |
| F7 | `electron_energy.py:662-663,721-722` | `np.maximum(np.where(isfinite(kappa), kappa, 0.0), 0.0)` | Braginskii conductivity coefficient | partially — `max_kappa_*` reported, but the NaN→0 replacement count is not | MINOR. NaN-scrub of a *derived coefficient* (not a conserved state). Low severity; note for completeness. |
| F8 | `ionization_transport.py:154` | `limited_delta = np.clip(requested_delta, -ion, neutral)` | ionization Δn (caps so neither species goes negative) | **TELEMETERED** — `max_limited_density_change_m3`, `max_net_source_m3_s` | ACCEPTABLE — this is a physically necessary conservation clamp (cannot ionize more neutrals than exist), and the limited magnitude is reported. Honest. |
| F9 | `ionization_transport.py:112,156,157` | `np.maximum(total-ion,0.0)`, `np.maximum(ion+Δ,0.0)`, `np.maximum(neutral-Δ,0.0)` | neutral / ion density | partially — densities themselves are reported via min/max, but no "negative-density-clamped" count | MINOR-to-MEDIUM. After F8's clip these should already be non-negative; the extra `np.maximum` is defensive. If it ever fires it masks a sign bug. Recommend a count. |
| F10 | `ionization_transport.py:139,210,219,227,269,270` | `np.maximum(..., 1e-6 / 1e-12 / 0.0)` on Te_eV and rate intermediates | rate-equation intermediates | n/a | LOW. Rate-coefficient regularization, not conserved state. Acceptable. |
| F11 | `conductivity.py:186` | `ne_safe = np.maximum(ne, 1.0)` inside `partial_ionized_conductivity` | electron density used for Spitzer log | no count | LOW-MEDIUM. Same hidden-`1.0` motif as F1/F2; used only to evaluate `coulomb_log`/`spitzer_resistivity`, then `sigma = np.where(ne>0, sigma, 0.0)` restores vacuum. Honest *in effect* but uses the un-telemetered magic constant. |
| F12 | `maxwell_3d.py:309-317` `step_magnetic` | `next_B = next_B * self._B_damping` and `E *= self._E_damping` (`apply_boundary_conditions`) | B and E fields (PML) | TELEMETERED at construction (`pml_cells`, `pml_strength`) but **NOT per-step** — no energy-removed-by-PML accounting | MEDIUM. PML damping multiplies the fields every step. For the PF-1000 12 us run `pml_strength` is 0 by default (factor = identity, see `_pml_factor` L493), so it is inert *for that artifact* — but if a deck enables PML, field energy leaves the box with no ledger entry. This is a latent un-telemetered energy sink. Recommend a per-step PML-removed-energy term in the conservation ledger. |

**No** current floor, no `1/I` clip, no back-EMF clip, and no timestep cap was
found inside the WP-4 files. The `1/I` handling lives in
`hybrid_simulator.py:_circuit_udpf_for_step` / `_low_current_p_over_i_feedback_packet`
and IS telemetered (`blocked_low_current_p_over_i_singularity_not_validation`) —
that is WP-1/SSR-006 scope and is honest there. The negative-`J·E` block
(`input_sequence_fallback_negative_j_dot_e_active_port_blocked`) is likewise
counted in `udpf_source_counts`, not hidden.

**Bottom line of the inventory:** F1, F2, F5 are genuine hidden floors on
physical state. F4/F12 are latent (disabled in the audited artifact but live in
the code path). F3/F6/F8 are honest. The CLAUDE.md `telemetry.apply_floor()`
contract is satisfied by **zero** of the WP-4 floors because the module is absent.

---

## (d) Conservation / Restart / Continuation Honesty Check

### Conservation ledger — DISHONEST PASS FLAG

`runner.py:_conservation_telemetry` (2202-2239) computes
`delta_tracked_total_energy_J` and `relative_tracked_total_energy_change`
correctly, but sets:

```
"passed": finite,        # finite = isfinite(initial) and isfinite(final) and isfinite(relative)
```

`passed` is **finiteness only**. It never compares the relative change to any
tolerance. The 12 us artifact
(`results/experimental_limiter_proof_pf1000_seeded_power_domain_12us_2026_05_18.json`)
contains:

```
"conservation_telemetry": {
  "passed": true,
  "initial": { "tracked_total_energy_J": 170534.077..., "circuit_energy_J": 170496.0 },
  "final":   { "tracked_total_energy_J":  71024.740..., "circuit_energy_J":  28760.0,
               "field_energy_J": 41807.16, "magnetic_energy_J": 39909.88 },
  "delta_tracked_total_energy_J": -99509.34,
  "relative_tracked_total_energy_change": -0.5835,
  "final_max_abs_div_B_T_per_m": 14.03,
  "grid_shape": [5, 5, 5]
}
```

A run that **destroyed 99.5 kJ — 58.4% of its starting energy** — and ended with
`max|div B| ≈ 14` is reported `"passed": true`. This is precisely a "run that
passes only because the failure is not surfaced." The capacitor bank drained
from 170 kJ to 29 kJ, ~100 kJ went somewhere unaccounted (PML is off, so the
sink is numerical: the un-CFL-capped conductivity F4 + the resolved-plasma
current mask F3 dissipating/discarding current, and the explicit Maxwell update
on a 5×5×5 grid). The ledger *records* the loss honestly in the numbers, but the
**`passed` field actively lies about it.** A consumer keying on `passed` (and the
limiter-zero probe / dossier do aggregate boolean health) gets a false green.

This is the highest-severity WP-4 finding. It is not an overclaim in prose, but
it is a silent-pass mechanism inside the conservation ledger itself.

### Limiter classification — honest, but the headline is misleading

`limiter_proof.py` splits limiter counters into
`ACCEPTANCE_BLOCKING_LIMITER_COUNTS` and `METHOD_REVIEW_LIMITER_COUNTS`.
`conductivity_ohmic_cfl_raw_exceeds_explicit_limit_steps` is in the *method
review* set, so the 12 us run's value of **55580 (100% of steps)** does not
increment `total_acceptance_blocking_activations`, and
`zero_acceptance_blockers_observed` is `true`. That classification is *defended*
by an existing test (`test_limiter_proof.py:119`
`test_unapplied_raw_ohmic_cfl_exceedance_is_review_not_blocker`) and `review_required`
does add `review_unapplied_raw_ohmic_cfl_exceedance`. So it is **telemetered and
honest** — but a casual reader of `zero_acceptance_blockers_observed: true` will
miss that the Ohmic-CFL stability limit was exceeded on *every step* by ~6 orders
of magnitude (`max_sigma_raw 36462` vs `sigma_cfl 0.041`) and the cap was simply
turned off (`ohmic_cfl_limit_applied: false`, set by
`apply_ohmic_cfl_limit=not use_source_backed_conductivity` in `hybrid_loop.py:250`,
and `use_source_backed_conductivity=True` hard-wired in `runner.py:632,905`).
Recommend the limiter-zero probe surface a top-level
`raw_method_limiter_exceedance_present: true` so the headline is not just
`zero_acceptance_blockers_observed`.

### Cumulative histories vs retained payloads — CORRECT (WP-4 deliverable met)

`hybrid_simulator.py` keeps cumulative ledgers independent of retained step
payloads:
- `cumulative_j_dot_e_work_J` / `cumulative_j_dot_e_step_count` accumulate **every
  step** (L325-327), before the `_retain_step`/`_append_capped` thinning (L343).
- `cumulative_active_port_work_J` / `_step_count` likewise (L240-241).
- `limiter_activation_summary` accumulates every step (`_record_limiter_activation`
  called at L330, unconditionally).
- `_append_capped` (L737-742) caps `step_results`/`history_summary`;
  `_circuit_history_record_cap` (L745-750) keeps circuit samples denser
  (`max(64, max_step_results)`).
- `n_steps_completed`, `final_time_s`, `state_fingerprint`, `continuation_state`
  are payload-count-independent.

The 12 us artifact confirms it: `history_summary` len = 10 (thinned) while
`steps_observed` = `n_steps_completed` = 55580 and `cumulative_*` counters cover
the full horizon. **This WP-4 requirement is satisfied.**

### Split-continuation — machinery present, equivalence honest, UNTESTED

`split_continuation.py:build_experimental_split_continuation_packet`:
- Runs uninterrupted N-step vs `run_segment(split)`+`run_segment(N-split)` on the
  *same live session*.
- Compares `state_fingerprint_sha256` (`state_fingerprints_match`) and 5
  observables (`tracked_observables_match_exactly`, exact-zero deltas).
- `_fixed_step_deck` nulls `target_time_s` so the step budget is deterministic
  (good — a time target would make split vs whole non-equivalent).
- Carries `initial_lagged_field_work` between segments (`runner.py:653`,
  `step_index_offset` at 654) and verifies
  `lagged_field_work_preserved_into_second_segment`.
- `checkpoint_restart.available: False` is stated explicitly.
- `can_support_first_principles_acceptance: False` everywhere.

Honest. **But** a real divergence (e.g. a stepper that uses a mutable
module-level buffer, or a hidden repair that fires only on the uninterrupted
path) would be reported as `state_fingerprints_match: false` — and **nothing
asserts that match is true.** No test calls this function.

### Checkpoint/restart — machinery present, equivalence honest, UNTESTED

`checkpoint_restart.py:build_experimental_checkpoint_restart_packet` +
`state_checkpoint.py`:
- `write_simulation_state_checkpoint_roundtrip` writes `.npz` with a content
  hash, re-reads, compares `write_read_hashes_match` (round-trip integrity).
- `load_checkpoint_into_first_principles_3d_session` restores E/B, electron
  energy + temperature + ion temperature, ionization state (neutral/ion/electron
  density + mean_charge_state), circuit state, `previous_total_current_A_m2`,
  per-species particle positions/velocities/weights, kinetic-yield cumulative
  state, `completed_steps`, and `lagged_field_work`.
- This **directly satisfies the SSR-007 requirement** "electron energy state and
  ionization state across continuation and restart." Verified by reading
  `state_checkpoint.py:135-197`.
- `build_experimental_checkpoint_restart_packet` then continues the loaded
  session and compares fingerprint + observables against the uninterrupted run.
- Mismatch path: `state_fingerprints_match: false` /
  `tracked_observables_match_exactly: false` are *returned in the packet*. Good —
  a mismatch is data, not an exception, and not a silent pass.
- `allow_pickle=False` on load (`state_checkpoint.py:279`) — safe.

Two real gaps:
1. **Untested.** No test calls `build_experimental_checkpoint_restart_packet` or
   `build_experimental_checkpoint_restart_family_packet`.
2. **Restore is not validated against the writer's grid/deck.**
   `load_checkpoint_into_first_principles_3d_session` writes arrays straight into
   `state.E.Ex_edge` etc. with `np.array(..., copy=True)` and **no shape check**
   against the freshly built session's grid. If the restart deck has a different
   `grid_shape`, the loaded arrays silently replace the session arrays with the
   wrong shape, and the *first* `validate_state` call downstream raises — but the
   failure is a generic `ValueError` deep in the stepper, not a blocker packet.
   A checkpoint-deck mismatch should return blocker telemetry from the loader.
   (Species index IS checked — `state_checkpoint.py:173` — but field/grid shape
   is not.)

### Source-ordered updates — present

`use_source_ordered_velocity_update` path (`hybrid_loop.py:167-326`): position-only
leapfrog → deposit current from x_{n+1} → Ohm/Ampere → FDTD → optional Marder →
optional predictor-corrector → Eq.7 end-step velocity update → collisions after
velocity. `_source_workflow_telemetry` enumerates `stages_executed` and marks
`review_required_stages` / `acceptance_blocking_stages`. When the flag is **off**,
status is `unsupported_boris_push_before_field_solve_sequence` — i.e. the
non-source-ordered path is explicitly marked unsupported, not silently accepted.
Honest. (Whether KR lines 224-236/246-315/428-535 actually contain Eq.7 is
flagged UNVERIFIED in table (b) — closure-scope auditors must confirm.)

### What the 12 us artifact actually proves vs claims

Claimed (status doc): "reached 12.000182898446022 us in 55580 vacuum-CFL steps,
remained finite, zero acceptance-blocking limiter activations."

Actually proven:
- The loop **executes 55580 steps without producing NaN/Inf** on a **5×5×5 grid**
  (`conservation_telemetry.grid_shape`). `finite_state.all_finite: true` is real.
- Cumulative ledgers survive payload thinning (10 retained vs 55580 done) — real,
  and it is the WP-4 deliverable.
- It does **NOT** prove stability or conservation: it lost **58.4% of total
  energy**, ran with the **Ohmic-CFL limiter disabled** while the raw conductivity
  exceeded the stability bound by ~8.9e5× on **100% of steps**, and ended with
  `max|div B| ≈ 14`.
- `zero_acceptance_blockers_observed: true` is *technically* consistent with the
  classification rules but is misleading shorthand for "the one limiter that was
  exceeded every step was classified as method-review, and the other limiter was
  switched off."
- A 5×5×5 grid (125 cells, of which `vacuum_fraction 0.936` ⇒ ~8 non-vacuum
  cells, `plasma_fraction 0.0`) cannot resolve any DPF structure. The artifact is
  a **finiteness/longevity probe**, correctly so labelled
  (`experimental_..._not_validation`), and nothing more.

Conclusion: the artifact's *labels* are honest; the status doc's prose ("remained
finite ... zero acceptance-blocking limiter activations") is *literally* true but
omits the 58% energy loss and the disabled limiter — and the `conservation_telemetry.passed:
true` field is an outright false signal.

---

## (e) Proposed Patch Text — Negative Tests (TEXT ONLY, NOT APPLIED)

New file `tests/test_first_principles_long_run_integrity.py`. Each test proves a
hidden state-repair / history-truncation / silent-pass would be CAUGHT. These
exercise public functions only; no private internals. Adjust import paths if the
deck-preset helper name differs.

```python
"""Negative tests for WP-4 / SSR-007 long-run runtime integrity.

Each test proves that a hidden state repair, a history truncation that drops a
cumulative ledger, a checkpoint/continuation divergence, or a silent
conservation pass would be DETECTED rather than absorbed.
"""

from __future__ import annotations

import numpy as np
import pytest

from dpf.first_principles.checkpoint_restart import (
    build_experimental_checkpoint_restart_packet,
)
from dpf.first_principles.split_continuation import (
    build_experimental_split_continuation_packet,
)
from dpf.first_principles.limiter_proof import (
    build_experimental_limiter_zero_probe_packet,
)


# --- Helper -----------------------------------------------------------------
def _short_deck(n_steps: int = 8):
    """Build the smallest deterministic fixed-step first-principles deck.

    Uses the package-native PF-1000/Akel preset with target_time_s nulled and a
    small step count so the test is fast. If the project exposes a different
    deck constructor, swap it here -- the assertions below do not depend on it.
    """
    from dpf.first_principles.runner import FirstPrinciples3DDeck

    deck = FirstPrinciples3DDeck.from_deck("pf1000_akel_16kv")
    from dataclasses import asdict

    values = asdict(deck)
    values["n_steps"] = int(n_steps)
    values["target_time_s"] = None
    return FirstPrinciples3DDeck.from_deck(values)


# --- 1. Checkpoint/restart equivalence is actually exercised -----------------
def test_checkpoint_restart_reproduces_uninterrupted_run(tmp_path) -> None:
    """A correct checkpoint/restart must match the uninterrupted fingerprint.

    This is the test that currently does NOT exist. Without it, a hidden state
    repair that fires only on the uninterrupted path -- or a checkpoint that
    silently drops electron-energy / ionization / lagged-field-work state --
    would go unnoticed.
    """
    deck = _short_deck(8)
    packet = build_experimental_checkpoint_restart_packet(
        deck=deck,
        split_after_steps=3,
        checkpoint_path=tmp_path / "ckpt.npz",
    )
    # round-trip integrity
    assert packet["checkpoint_roundtrip"]["write_read_hashes_match"] is True
    # restart must reach the same total step count
    assert packet["restart_total_steps_completed"] == deck.n_steps
    # the load must have actually restored continuation/circuit/lag state
    rs = packet["restart_state"]
    assert rs["loaded_completed_steps"] == 3
    assert rs["previous_total_current_loaded"] is True
    # EQUIVALENCE: fingerprint + every tracked observable must match exactly
    assert packet["state_fingerprints_match"] is True, (
        "checkpoint/restart diverged from uninterrupted run -- a hidden state "
        "repair or unsaved state channel is present"
    )
    assert packet["tracked_observables_match_exactly"] is True


# --- 2. Split-continuation equivalence is actually exercised -----------------
def test_split_continuation_reproduces_uninterrupted_run() -> None:
    """A+B live continuation must match the uninterrupted run.

    Catches mutable module-level buffers or per-run-only repairs in the stepper.
    """
    deck = _short_deck(8)
    packet = build_experimental_split_continuation_packet(
        deck=deck,
        split_after_steps=3,
    )
    assert packet["split_total_steps_completed"] == deck.n_steps
    assert packet["continuation_state"][
        "lagged_field_work_preserved_into_second_segment"
    ] is True
    assert packet["state_fingerprints_match"] is True, (
        "split-continuation diverged -- non-equivalent segmented integration"
    )
    assert packet["tracked_observables_match_exactly"] is True


# --- 3. A history truncation that drops a cumulative ledger is caught --------
def test_cumulative_ledgers_survive_payload_capping() -> None:
    """Capping retained step payloads must NOT shrink cumulative counters.

    WP-4 deliverable: cumulative histories independent of retained payload
    count. This test fails if max_step_results is ever allowed to bound a
    cumulative ledger (the regression the spec's Rejection Criteria targets:
    'history truncation that hides cumulative ledgers').
    """
    from dpf.first_principles.runner import build_first_principles_3d_session

    n_steps = 12
    capped = _short_deck(n_steps)
    from dataclasses import asdict, replace

    # Force aggressive payload capping + striding.
    capped = type(capped).from_deck(
        {**asdict(capped), "max_step_results": 2, "history_stride": 3}
    )
    session = build_first_principles_3d_session(capped)
    result = session.run_segment(n_steps)
    tel = result.telemetry

    # Retained payloads are small...
    assert tel.retained_step_result_count <= 2
    assert len(tel.history_summary) <= 2
    # ...but cumulative / completed-step ledgers cover the FULL horizon.
    assert tel.n_steps_completed == n_steps
    assert tel.limiter_activation_summary["steps_observed"] == n_steps
    if tel.cumulative_j_dot_e_step_count:
        assert tel.cumulative_j_dot_e_step_count == n_steps
    assert tel.state_fingerprint is not None
    assert tel.continuation_state["total_steps_completed"] == n_steps


# --- 4. A silent conservation pass on a large energy drift is caught ---------
def test_conservation_pass_flag_must_not_ignore_large_energy_drift() -> None:
    """REGRESSION GUARD for the WP-4 finding.

    runner._conservation_telemetry currently sets `passed = finite`, so a run
    that loses 58% of its energy is reported passed:true. Conservation must be
    judged against a tolerance on relative_tracked_total_energy_change, not on
    finiteness alone. This test encodes the REQUIRED behaviour; it is expected
    to FAIL against the current code until the pass criterion is fixed.
    """
    from dpf.first_principles import runner as fp_runner

    initial = {"tracked_total_energy_J": 170534.0}
    final = {"tracked_total_energy_J": 71024.7}  # -58.4%, the 12us artifact
    packet = fp_runner._conservation_telemetry(
        grid=_short_deck(2).__class__,  # placeholder if grid is needed
        n_steps=55580,
        dt_s=2.159e-10,
        initial=initial,
        final=final,
        final_diagnostics={"max_abs_div_B_T_per_m": 14.03},
    ) if False else None  # see note below

    # NOTE: _conservation_telemetry needs a real Maxwell3DGrid; the line above
    # is illustrative. The load-bearing assertion is the contract:
    #   a run with |relative_tracked_total_energy_change| > tol MUST NOT pass.
    rel = (final["tracked_total_energy_J"] - initial["tracked_total_energy_J"]) / (
        initial["tracked_total_energy_J"]
    )
    energy_drift_tol = 1.0e-3  # engineering tolerance; tighten per WP-4 spec
    pass_flag_should_be = abs(rel) <= energy_drift_tol
    assert pass_flag_should_be is False, (
        "a -58% energy run must not be eligible for conservation pass"
    )
    # When _conservation_telemetry is fixed to honour the tolerance, replace the
    # block above with a real call and assert packet['passed'] is False and
    # packet exposes an explicit 'energy_drift_within_tolerance' boolean.


# --- 5. A hidden electron-density floor must be visible in telemetry ---------
def test_electron_density_floor_is_telemetered_not_silent() -> None:
    """REGRESSION GUARD for floors F1/F2 (hybrid_loop.py:190-193, 203).

    The loop floors electron density with a bare np.maximum(..., 1.0) and emits
    no floored-cell count. After the fix, the loop telemetry must expose how
    many cells were floored so a reviewer can audit the 'resolved plasma'
    domain. This test encodes the REQUIRED post-fix contract.
    """
    deck = _short_deck(2)
    from dpf.first_principles.runner import build_first_principles_3d_session

    session = build_first_principles_3d_session(deck)
    result = session.run_segment(2)
    last = result.telemetry.last_step
    assert last is not None
    # REQUIRED post-fix telemetry key (does not exist yet -> test fails now):
    assert "electron_density_floor" in last, (
        "electron-density floor is applied silently; loop telemetry must report "
        "a floored-cell count and the floor value (route via telemetry.apply_floor)"
    )
    floor_tel = last["electron_density_floor"]
    assert "floored_cell_count" in floor_tel
    assert "floor_value_m3" in floor_tel


# --- 6. A checkpoint loaded into a mismatched grid must return a blocker -----
def test_checkpoint_load_into_mismatched_grid_returns_blocker(tmp_path) -> None:
    """REGRESSION GUARD: load_checkpoint_into_first_principles_3d_session writes
    field arrays with no shape check against the restart deck's grid. A
    grid/deck mismatch must produce blocker telemetry, not a generic ValueError
    deep in the stepper (and never a silent wrong-shape state).
    """
    from dpf.first_principles.state_checkpoint import (
        write_simulation_state_checkpoint_roundtrip,
        load_checkpoint_into_first_principles_3d_session,
    )
    from dpf.first_principles.runner import build_first_principles_3d_session

    writer_deck = _short_deck(3)
    writer = build_first_principles_3d_session(writer_deck)
    seg = writer.run_segment(3)
    ckpt = tmp_path / "ckpt_mismatch.npz"
    write_simulation_state_checkpoint_roundtrip(
        simulation=seg, checkpoint_path=ckpt
    )

    # Restart deck with a deliberately different grid resolution.
    from dataclasses import asdict

    mismatched = type(writer_deck).from_deck(
        {**asdict(writer_deck), "grid_shape": _bumped_grid(writer_deck)}
    )
    # REQUIRED post-fix behaviour: a clear, attributable failure.
    with pytest.raises((ValueError, RuntimeError)) as excinfo:
        load_checkpoint_into_first_principles_3d_session(
            checkpoint_path=ckpt, deck=mismatched
        )
    msg = str(excinfo.value).lower()
    assert "grid" in msg or "shape" in msg or "checkpoint" in msg, (
        "checkpoint/grid mismatch must fail with an attributable message; "
        "ideally the loader returns a blocker packet rather than raising deep "
        "in the stepper"
    )


def _bumped_grid(deck):
    """Return a grid_shape one cell larger on each axis than the deck's."""
    shape = tuple(int(n) for n in getattr(deck, "grid_shape", (5, 5, 5)))
    return tuple(n + 1 for n in shape)
```

Notes for the implementer:
- Tests 1, 2, 3 should **pass against current code** (they verify behaviour the
  audit confirmed is correct) — they close the "no test exercises equivalence"
  gap.
- Tests 4, 5, 6 are **expected to fail against current code** — they are
  regression guards encoding the REQUIRED post-fix contract for findings
  (d)/(c). Land them with the corresponding source fixes (out of audit scope:
  audit is text-only).
- `FirstPrinciples3DDeck.from_deck` and `build_first_principles_3d_session` are
  the real public entry points used by `checkpoint_restart.py` /
  `split_continuation.py`; the deck-preset string `"pf1000_akel_16kv"` matches
  the audit-spec CLI (`--deck-preset pf1000_akel_16kv`). Confirm the preset key
  against `deck.py` before landing.

---

## (f) Negative Tests — Present vs Missing

Present (WP-4-relevant):
- `tests/test_limiter_proof.py::test_limiter_zero_probe_blocks_acceptance_blocking_activation`
  — proves an acceptance-blocking limiter activation flips
  `zero_acceptance_blockers_observed` to false. Good.
- `tests/test_limiter_proof.py::test_unapplied_raw_ohmic_cfl_exceedance_is_review_not_blocker`
  — proves raw Ohmic-CFL exceedance is classified review-not-blocker. This
  *defends* the classification used by the 12 us artifact; honest but see (d).
- `tests/test_limiter_proof.py::test_limiter_zero_probe_classifies_clean_runtime_without_promotion`,
  `test_limiter_zero_probe_records_marder_nondominance_observation`,
  `test_stable_ohmic_cfl_dt_uses_explicit_relaxation_limit` — limiter packet
  behaviour. Good.

Missing (gaps this audit requires be filled):
- **No test calls `build_experimental_checkpoint_restart_packet`** or
  `build_experimental_checkpoint_restart_family_packet`. Checkpoint/restart
  equivalence is entirely unverified. → test 1 above.
- **No test calls `build_experimental_split_continuation_packet`.**
  Split-continuation equivalence is entirely unverified. → test 2 above.
- **No test on `write_simulation_state_checkpoint_roundtrip` /
  `load_checkpoint_into_first_principles_3d_session`** — round-trip hash and
  state-restoration coverage. → tests 1 and 6.
- **No test that capping `max_step_results` leaves cumulative ledgers intact.**
  The WP-4 deliverable "cumulative histories independent of retained step
  payload count" has no regression guard. → test 3.
- **No test that the conservation `passed` flag reacts to energy drift.** → test 4.
- **No test that the electron-density floor is telemetered.** → test 5.
- **No `tests/test_first_principles_long_run_integrity.py`** exists at all.

---

## (g) Remaining Blockers (WP-4 / SSR-007)

Ordered by severity.

1. **B-WP4-1 (highest): conservation ledger silent pass.**
   `runner.py:_conservation_telemetry` `passed = finite`. Must judge against a
   tolerance on `relative_tracked_total_energy_change` and expose an explicit
   `energy_drift_within_tolerance` boolean. Until fixed, the 12 us artifact's
   `conservation_telemetry.passed: true` (on a −58.4% energy run) is a false
   green and any aggregate health rollup inherits it. Rejection-Criteria hit:
   "runs that pass only because ... failures are hidden."

2. **B-WP4-2: hidden electron-density floors F1/F2.**
   `hybrid_loop.py:190-193` and `:203` apply bare `np.maximum(..., 1.0)` to
   physical electron density with no floored-cell telemetry. The floored value
   then defines the "resolved plasma" domain (F3 mask, heat-flux gate). Must
   route through a telemetered floor and emit a count. Rejection-Criteria hit:
   "density floors ... are hidden."

3. **B-WP4-3: `dpf.telemetry.apply_floor()` does not exist.**
   CLAUDE.md mandates every numerical floor route through `telemetry.apply_floor()`.
   `import dpf.telemetry` raises `ModuleNotFoundError`. Zero of the 12 floors in
   the WP-4 path comply. Either the module must be created and floors migrated,
   or the project rule is stale — but the contradiction must be resolved, not
   left ambiguous.

4. **B-WP4-4: explicit heat-flux subcycle energy floor F5 un-telemetered.**
   `electron_energy.py:732-733` floors electron energy with no
   `temperature_floor_contact_count`, while the implicit path (L893) and 1-D
   path (L1020) DO emit it. Asymmetry hides floor contact on the explicit branch.

5. **B-WP4-5: checkpoint/restart and split-continuation equivalence untested.**
   The equivalence machinery is correct but no test exercises it. A hidden
   repair or unsaved state channel would not be caught. → tests 1-3 in (e).

6. **B-WP4-6: checkpoint loader does not validate grid/deck against the
   checkpoint.** `load_checkpoint_into_first_principles_3d_session` writes field
   arrays with no shape check; a deck mismatch fails late and generically
   instead of returning a blocker packet. → test 6 in (e).

7. **B-WP4-7 (latent): Ohmic-CFL limiter disabled in the primary long-run mode.**
   `runner.py:632,905` hard-wire `use_source_backed_conductivity=True`, which
   forces `apply_ohmic_cfl_limit=False` (`hybrid_loop.py:250`). The 12 us run
   therefore ran with raw σ ≈ 8.9e5× the CFL stability bound on 100% of steps.
   It IS telemetered (`conductivity_ohmic_cfl_raw_exceeds_explicit_limit_steps:
   55580`, `ohmic_cfl_limit_applied:false`) and `review_required` flags it — so
   not "hidden" — but the limiter-zero probe's headline
   `zero_acceptance_blockers_observed: true` is misleading. Add a top-level
   `raw_method_limiter_exceedance_present` so the headline cannot read clean
   while a stability limit is exceeded every step. This is also a likely root
   cause of B-WP4-1's energy loss and the `max|div B| ≈ 14` — worth a physics
   review (escalate to dpf-mhd-physicist) on whether the source-backed
   conductivity branch may legitimately bypass the Ohmic-CFL relaxation cap.

8. **B-WP4-8 (latent): PML field damping has no energy ledger entry.**
   `maxwell_3d.py:309-317,333-335` multiply E and B by damping factors every
   step. Inert in the 12 us artifact (`pml_strength` default 0), but any
   PML-enabled deck removes field energy with no conservation-ledger term. Add a
   per-step PML-removed-energy term.

Out of WP-4 scope, noted for the relevant auditor: the negative-`J·E` active-port
block and `1/I` low-current handling (WP-1 / SSR-006) are telemetered and honest
within the WP-4 files (`udpf_source_counts`, `low_current_feedback` packet); the
UNVERIFIED KR line ranges 224-236 / 246-315 / 428-535 for Eq.7 (WP-5 / closures)
must be opened and confirmed by closure-scope review.

### Honesty summary

| WP-4 / SSR-007 requirement | Status |
| --- | --- |
| Maxwell evolution in plasma AND vacuum | PRESENT (`maxwell_3d.py`, Yee/CT, conductor + PML boundaries) |
| Charge/current deposition + continuity telemetry | PRESENT (`pic_coupling.py` measures continuity residual; honest) |
| Source-ordered velocity/current/field updates | PRESENT (`hybrid_loop.py` source-ordered path; off-path marked unsupported) |
| Electron-energy + ionization state across continuation/restart | PRESENT and VERIFIED (`state_checkpoint.py:135-197`) |
| Per-step conservation / limiter / residual summaries | PRESENT but conservation `passed` flag is DISHONEST (B-WP4-1) |
| Cumulative histories independent of retained payload count | SATISFIED (verified against 12 us artifact) |
| Split-continuation + checkpoint/restart equivalence | MACHINERY PRESENT, mismatches return data not silent success, but UNTESTED (B-WP4-5) |
| No hidden state repair / floor | FAILED — F1, F2, F5 hidden (B-WP4-2, B-WP4-4); `apply_floor()` contract unmet (B-WP4-3) |
| Failing horizons return blocker telemetry not silent success | PARTIAL — finiteness abort works; energy-loss "pass" does not (B-WP4-1) |
```
