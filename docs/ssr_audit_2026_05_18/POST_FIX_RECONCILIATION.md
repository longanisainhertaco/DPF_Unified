# POST-FIX RECONCILIATION — WP Audit Docs vs Applied Fixes

Date: 2026-05-18
Closes: Codex finding **A-10** ("The WP Audit Docs Are Stale Relative To The Fixes")
and **Immediate Blocker 8** ("Reconcile the WP audit docs after fixes").
Repo: `/Users/anthonyzamora/dpf-unified`  Branch: `codex/corpus`  HEAD: `c847a0d`
Runtime: `.venv312/bin/python` (Python 3.12)

## Purpose

The eight detailed WP audit docs (`WP-1` … `WP-7`, `claim_safety_sweep`) were
written by the external review team as **static, read-only hypotheses** during
the SSR audit pass. Several of them describe the codebase as it stood **before**
the integrity fixes in commits `e7bedea` / `cd64b8e` were applied — and a
further set of fixes (WP-N3 geometry patches, WP-N4 checkpoint guard, conservation
re-wording, manifest provenance, the negative-control test suites) landed
**after** the WP docs were authored. The Codex external audit
(`c847a0d`) confirmed the WP docs are no longer self-consistent with the code.

This document reconciles every WP finding against the **current code at git
HEAD** — the code is truth, the WP docs are hypotheses. Each finding is verified
by reading the actual source line and/or running the test that exercises it.

### Method and verification notes

- Every `fixed` row was checked by opening the cited current-code line and, where
  a test exists, by running it. Doc-cited line numbers were re-derived against HEAD.
- Test verification command (run clean, `__pycache__` and `.pytest_cache` cleared):
  `.venv312/bin/python -B -m pytest tests/test_first_principles_geometry.py
  tests/test_first_principles_long_run_integrity.py
  tests/test_first_principles_startup_bvp.py tests/test_first_principles_closures.py
  tests/test_first_principles_neutron_authority.py
  tests/test_first_principles_certificate_negative_controls.py -v` → **all pass**.
- **Stale-cache caveat:** a first run produced `NameError` (`_optional_str_tuple`,
  `_deck_fingerprint`) and stale `XPASS` lines. These were `.pyc` / `.pytest_cache`
  artifacts from pre-patch bytecode. After clearing caches, all tests pass and
  both helpers resolve. Both names are genuinely defined at HEAD
  (`manifest.py:403`, and `_deck_fingerprint` is reachable in `state_checkpoint.py`).
  No live `NameError` exists at HEAD — verified by fresh runs.

### Codex four xfails — STATUS AT HEAD

The Codex doc lists four xfails as "real blockers". **At git HEAD, all four are
resolved and the tests PASS** (the WP-N3/WP-N4 patches landed after the Codex
audit ran; the xfail markers were removed):

| Codex-listed xfail | Status at HEAD | Evidence |
| --- | --- | --- |
| `test_first_principles_geometry.py::test_conductor_mask_packet_emits_mask_hash` | PASS | `runner.py:2604` `_conductor_mask_sha256`, `:2674` `mask_sha256` emitted in `projection_error` block |
| `test_first_principles_geometry.py::test_conductor_mask_packet_emits_projection_error` | PASS | `runner.py:2621` `_conductor_mask_projection_error`, emitted at `:2766` |
| `test_first_principles_geometry.py::test_reviewed_rod_mask_requires_resolved_rods` | PASS | resolution gate raises `ValueError("...cells across a rod diameter")` for under-resolved reviewed mask |
| `test_first_principles_long_run_integrity.py::test_checkpoint_load_into_mismatched_grid_fails_attributably` | PASS | `state_checkpoint.py` `deck_fingerprint` + grid/shape guard; attributable failure on mismatch |

The Codex doc's "180 passed, 4 xfailed" reflects an earlier tree state; at HEAD
those four tests are unmarked and passing.

---

## WP-1 / SSR-006 — Circuit / Power-Port Authority

| WP | Finding | Status | Evidence |
| --- | --- | --- | --- |
| WP-1 | G1/B1: WP-1 negative-test suite absent (sign reversal, domain corruption, time-centering, low-current, Sigma rejection) | superseded | Codex re-scoped power-port closure into WP-N1; the negative-test set is a WP-N1 deliverable ("Add negative tests for sign reversal, domain corruption, time-centering downgrade, omitted electrode work, residual-policy failure, and low-current fallback"). Partial coverage exists (`test_first_principles_runner.py` low-current blocker test). → **WP-N1** |
| WP-1 | G2: Auluck-mode first-step fallback not pinned by a test | still_open | No test asserts the `input_sequence_fallback_first_step` tag for the Auluck mode. → **WP-N1** |
| WP-1 | G3/B3: Energy ledger is 4/5 — `wall_poynting_flux_excluding_declared_port_J` + `electrode_interface_work_J` missing | still_open | `power_port.py` Stage-0 ledger still marks both terms `missing_or_blocked`. Codex WP-N1 explicitly requires both as emitted ledger channels. → **WP-N1** |
| WP-1 | G4/B7: `PF1000_AKEL_DECK_SOURCE_REFS` line range `108-142` not independently re-opened | still_open | Low-severity citation-range tightening; folded into WP-N1 source-packet re-open of the Auluck/RadPhysChem refs. → **WP-N1** |
| WP-1 | G5/B2: No declared power-port domain; `interface_surface_or_volume_domain: not_declared`, `power_port_domain_review: blocked_domain_packet_not_available` | still_open | `power_port.py` still emits `not_declared` / `blocked_domain_packet_not_available`. The named Auluck `Ω` runtime domain is the core WP-N1 deliverable ("Implement a named runtime integration domain for Auluck Omega"). → **WP-N1** |
| WP-1 | G6: Default `circuit_udpf_mode` (`lagged_volume_j_dot_e`) not pinned by a default-mode test | still_open | Conservative default is correct, but no test asserts a deck without explicit mode cannot inherit a P/I path. → **WP-N1** |
| WP-1 | G7/B4: No segmented/checkpointed 12 µs source-sign run path | still_open | No segmented driver chains `lagged_field_work` across checkpoints for the power-port branch. Codex WP-N4 adds `--segment-steps`; the 12 µs source-sign artifact is a joint WP-N1+WP-N4 item. → **WP-N4** (with **WP-N1**) |
| WP-1 | G8/B5: Residual tolerance `accepted_residual_tolerance: not_attached` | still_open | Correctly deferred by design; acceptance still needs a reviewed `R_pp` rule. Codex WP-N1 requires "a reviewed residual definition with sign and time-centering". → **WP-N1** |
| WP-1 | B6: Committed `100ns/1ns/12us` artifacts predate the new code; lack Stage-0 packets + deck-diff; carry stale fields | still_open | Codex A-1 and **WP-N0** (evidence hygiene): regenerate/quarantine stale artifacts, add `scripts/audit_first_principles_artifacts.py`. → **WP-N0** |
| WP-1 | B8: Pre-existing Beresnyak refs (`beresnyak_2018:170-200`, `beresnyak_2022:44-72`) not re-verified | still_open | Outside the WP-1 diff; non-blocking citation re-verification. → **WP-N1** |
| WP-1 | Source-sign Auluck mode `U_DPF = -P_JE/I` implemented, signed, candidate-labelled; negative `J·E` not clipped; low-current `1/I` reported as blocker; Sigma deferred; deck-diff locks PF-1000/Akel | fixed | `hybrid_simulator.py:540-565` (`_circuit_udpf_for_step`, Auluck `-power_W/I`); `:568-619` (`_low_current_p_over_i_feedback_packet`, `blocked_low_current_p_over_i_singularity_not_validation`); `runner.py:2244` `_deck_source_diff_packet`. Tests: `test_first_principles_runner_reports_low_current_p_over_i_feedback_blocker`, `test_lagged_auluck_j_dot_e_feedback_uses_source_sign_candidate`. WP-1 verdict was `accept_engineering_progress` — this part is not contested. |
| WP-1 | Every new power-port packet/sub-packet fail-closed; no overclaim | fixed | `power_port.py` `build_engineering_power_port_packet` hard-codes `can_support_first_principles_acceptance: False`; `accepted_load_power_source: "none"`. Confirmed by `claim_safety_sweep` §b. |

## WP-2 / SSR-004 — Startup BVP

| WP | Finding | Status | Evidence |
| --- | --- | --- | --- |
| WP-2 | Blocker 1 (BLOCKING): `build_startup_bvp_packet` grants `accepted_startup_bvp_packet` on caller-declared `accepted_channels` alone, contradicting `startup_payload_review` | **fixed** | The WP-2 doc was written **before** the `e7bedea` startup gate fix. Current code `startup_bvp.py:214-223`: `payload_acceptance_eligible = bool(startup_payload_review["channel_acceptance_eligible"])` and `can_support = whole_shot_requested and mode_is_accepted and reviewed and not missing and payload_acceptance_eligible`. The headline status now consumes the payload review — exactly the fix the WP-2 doc proposed. Test: `tests/test_first_principles_startup_bvp.py` (all pass at HEAD, incl. accepted-mode-without-payload stays blocked). |
| WP-2 | Missing `tests/test_first_principles_startup_bvp.py`; no test catches blocker 1 | fixed | `tests/test_first_principles_startup_bvp.py` exists (Codex A-3 + the WP-2 patch text were applied as the negative-control suite in `e7bedea`). Verified present and passing at HEAD. |
| WP-2 | `seeded_layer` rejected; certificate gate carries `rejected_startup_mode_for_first_principles`; `startup_breakdown.py` hard-pins acceptance False | fixed | `startup_bvp.py` rejected-mode handling intact; verdict in WP-2 §c confirms seeded path is honest. Not contested. |
| WP-2 | Blocker 2: No source-backed breakdown/flashover model; only the training-data CIV/Paschen scaffold | still_open | `civ_breakdown.py` remains an uncited CIV/Paschen scaffold; the 9 SSR-004 startup channels are not source-backed-and-implemented. Codex WP-N2 requires one source packet per channel from `KnowledgeReference/`. → **WP-N2** |
| WP-2 | Blocker 3: 0 of 9 SSR-004 startup channels source-backed-and-implemented | still_open | Channels are fail-closed packet slots; ~2 are candidate inputs only. → **WP-N2** |
| WP-2 | Blocker 4: No explicit handoff interval wired into the field/PIC loop | still_open | `sheath_liftoff_and_handoff_interval` yields only `engineering_estimate_not_reviewed_startup_bvp`; `startup_handoff_interval` is a *missing* acceptance channel. Codex WP-N2 requires `startup_handoff_interval_s` + initial `E/B/J/density/ionization/Te/Ti`/sheath fields. → **WP-N2** |

## WP-3 / SSR-005 — Reviewed Geometry And Material Boundaries

| WP | Finding | Status | Evidence |
| --- | --- | --- | --- |
| WP-3 | Blocker 1/2: No mask hash, no projection-error / error-from-source-dimensions field | **fixed** | WP-3 doc written before the WP-N3 geometry patches. Current code `runner.py:2604` `_conductor_mask_sha256`, `:2621` `_conductor_mask_projection_error` (emits `mask_sha256`, `max_radial_discretization_error_m`, `max_axial_discretization_error_m`, `cells_per_rod_diameter`), attached to the conductor-mask packet at `:2766`. Tests: `test_conductor_mask_packet_emits_mask_hash`, `test_conductor_mask_hash_is_deterministic`, `test_conductor_mask_packet_emits_projection_error` — all PASS. |
| WP-3 | Blocker 3: Rod-fidelity claim not resolution-gated; `cathode_rods_projected: True` at 0.73 cells/rod with no gate | **fixed** | `runner.py:2571-2576` emits `cathode_rod_diameter_grid_cells` and `cathode_rods_resolution_reviewed: False`; the resolution gate raises `ValueError("...cells across a rod diameter")` for an under-resolved reviewed mask. Tests: `test_cathode_rod_diameter_grid_cells_reported`, `test_coarse_grid_reports_low_cells_per_rod`, `test_reviewed_rod_mask_requires_resolved_rods` — all PASS. |
| WP-3 | Blocker 10: No `tests/test_first_principles_geometry.py` | fixed | File exists; 9 tests, all pass at HEAD. The WP-3 patch-text tests were applied as the WP-N3 suite. |
| WP-3 | Blocker 4: Alumina insulator mask absent (string label only) | still_open | `runner.py:2581-2582` still emits `insulator_material_surface_resolved: False`; no insulator voxel/material region built. Codex WP-N3 requires a separate insulator mask. → **WP-N3** |
| WP-3 | Blocker 5: Hollow anode not realized — PF-1000 deck never sets `device_anode_inner_radius_m`, anode rendered solid | still_open | `runner.py:2578` `hollow_anode_inner_radius_supplied` reflects the deck never supplying it. Honestly disclosed but the mask does not match source 109633:112. → **WP-N3** |
| WP-3 | Blocker 6: Electrode backplate / source-interface mask absent | still_open | No backplate mask, no `source_interface` geometry label. Codex WP-N3 requires separate masks for source interface. → **WP-N3** |
| WP-3 | Blocker 7: Vacuum-chamber wall mask absent | still_open | Outer grid is open/PML, not a reviewed chamber wall. Codex WP-N3 requires a separate chamber mask. → **WP-N3** |
| WP-3 | Blocker 8: Krauz-vs-Akel geometry disagreement (4.3% anode length, 20% cathode radius) invisible in deck-diff | still_open | `_deck_source_diff_packet` diffs only the self-consistent Akel lock; no cross-source advisory. Codex WP-N3 requires explicit resolution of source-dimension disagreements. → **WP-N3** |
| WP-3 | Blocker 9: `source_geometry.py` `HybridPICSourceGeometry` is the LLNL device, not PF-1000 | rejected_after_review | Not a defect — the packet is honestly scoped `llnl_like_180ka_axisymmetric_hybrid_pic` and never claimed as PF-1000 (WP-3 §g item 9 itself says "Not a defect, but a structural note"). No code change required; PF-1000 geometry correctly lives in `runner.py`/`deck.py`. |
| WP-3 | Every geometry/boundary packet `candidate_*` with `can_support_first_principles_acceptance: False`; boundary policy wired into Maxwell core; 12-rod mask geometrically correct | fixed | `runner.py` `_conductor_mask_packet` status `candidate_engineering_conductor_mask_not_validation`; confirmed by WP-3 §g "already correct" list. Not contested. |

## WP-4 / SSR-007 — Long-Run Field/PIC/Electron Runtime

| WP | Finding | Status | Evidence |
| --- | --- | --- | --- |
| WP-4 | B-WP4-1 (highest): Conservation ledger sets `passed = finite`; a −58% energy run reports `passed: true` | **fixed** | WP-4 doc written before the conservation re-wording in `e7bedea`. Current code `runner.py:2203-2241` `_conservation_telemetry` has **no `passed` key at all** — it emits `finite_state`, `energy_conservation_assessed: "not_assessed_no_accepted_tolerance"`, `status: "engineering_candidate_conservation_telemetry_not_validation"`, `can_support_first_principles_acceptance: False`. Test: `test_conservation_telemetry_has_no_passed_key_and_is_honest` — PASS. The false-green signal is removed. |
| WP-4 | B-WP4-2: Hidden electron-density floors F1/F2 (`hybrid_loop.py:190-193, 203`) with no floored-cell telemetry | **fixed** | Current code `hybrid_loop.py:192-212` computes `electron_density_floor_active_cells` and `electron_density_floor_source`; `:866-871` emits an `electron_density_floor` telemetry block (`floor_active_cells`, `floor_source`, floor value). `hybrid_stepper.py:271` emits `numerical_electron_density_floor_m3`. Test: `test_electron_density_floor_is_telemetered_in_source_workflow` — PASS. The floor is no longer silent. |
| WP-4 | B-WP4-5: Checkpoint/restart and split-continuation equivalence machinery untested | **fixed** | `tests/test_first_principles_long_run_integrity.py` exists; `test_checkpoint_restart_reproduces_uninterrupted_run` and `test_split_continuation_reproduces_uninterrupted_run` exercise the equivalence — both PASS at HEAD. |
| WP-4 | B-WP4-6: Checkpoint loader does not validate grid/deck against the checkpoint; mismatch fails late and generically | **fixed** | `state_checkpoint.py` now records a `deck_fingerprint` and validates grid/shape on load; `test_checkpoint_load_into_mismatched_grid_fails_attributably` PASSES — a mismatch produces an attributable failure (this is the 4th Codex xfail, now resolved). |
| WP-4 | No `tests/test_first_principles_long_run_integrity.py` exists at all | fixed | File exists; 6 tests, all pass at HEAD. |
| WP-4 | B-WP4-3: `dpf.telemetry.apply_floor()` mandated by CLAUDE.md does not exist; 0 of 12 WP-4 floors comply | still_open / partially superseded | `import dpf.telemetry` still raises `ModuleNotFoundError`. However the `claim_safety_sweep` §e independently judged the FP/fields floors as **disclosed via a documented alternative pattern** (named validated parameters + dedicated telemetry packets), and flagged this as a *process observation, non-blocking*. The contradiction between CLAUDE.md and the substitute pattern remains unresolved at the policy level. Codex WP-N5 ("turn warnings into telemetry / closure-regime gates") is the closest home; primarily a documentation/policy reconciliation. → **WP-N5** (policy note) |
| WP-4 | B-WP4-4: Explicit heat-flux subcycle energy floor F5 (`electron_energy.py:732-733`) un-telemetered while implicit path counts it | still_open | The `claim_safety_sweep` §e row for `electron_energy.py:732-734` reports `temperature_floor_contact_count` + `raw_min_temperature_before_floor_K` are emitted and judged "disclosed — not hidden", which conflicts with WP-4's claim that the *explicit subcycle branch specifically* lacks the count. The asymmetry between explicit and implicit branches needs a direct code re-check and, if confirmed, a count on the explicit path. → **WP-N5** |
| WP-4 | B-WP4-7 (latent): Ohmic-CFL limiter disabled in the primary long-run mode (`use_source_backed_conductivity=True`); raw σ exceeds CFL bound on 100% of steps; `zero_acceptance_blockers_observed: true` misleading | still_open | `runner.py` still hard-wires `use_source_backed_conductivity=True`; the exceedance IS telemetered (`conductivity_ohmic_cfl_raw_exceeds_explicit_limit_steps`) and `review_required` flags it, but the limiter-zero probe headline needs a top-level `raw_method_limiter_exceedance_present`. Needs a physics review on whether the source-backed branch may legitimately bypass the Ohmic-CFL cap. → **WP-N5** (closure/numerics) with physics review |
| WP-4 | B-WP4-8 (latent): PML field damping has no conservation-ledger entry | still_open | `maxwell_3d.py` multiplies E/B by damping factors each step; inert in the 12 µs artifact (`pml_strength` default 0) but no per-step PML-removed-energy ledger term. → **WP-N4** |
| WP-4 | Maxwell in plasma+vacuum, charge/current deposition, source-ordered updates, electron-energy + ionization persisted through restart, cumulative ledgers independent of retained payload | fixed | WP-4 §honesty-summary marks all PRESENT/SATISFIED/VERIFIED; `state_checkpoint.py:135-197` restores full state. Not contested. |

## WP-5 / SSR-008 — Physics Closures

| WP | Finding | Status | Evidence |
| --- | --- | --- | --- |
| WP-5 | Blocker 1: No `tests/test_first_principles_closures.py` | fixed | `tests/test_first_principles_closures.py` exists; the WP-5 patch-text tests were applied. All pass at HEAD. |
| WP-5 | Blocker 9: Two imprecise citations — S2 (`conductivity.py` NRL Spitzer `2660-2725`), S6 (`electron_energy.py` heat-flux `.json:57-62`) | still_open | Citation-range repointing (WP-5 Patch 2/3). Not fabricated; ranges exist but do not point at the backed formula. Folded into the WP-N5 closure source-packet re-open. → **WP-N5** |
| WP-5 | Empirical-knob gap: `closure_packet.py` does not enumerate the 5 fenced-out empirical modules (`line_radiation`, `ablation`, `qmf_suppression`, `transport`, `atomic/ionization`) | still_open | WP-5 §d confirms all 5 are correctly fenced OUT of the FP runtime (not imported by `hybrid_loop`/`hybrid_stepper`/`runner`), but the closure packet does not name them. WP-5 Patch 1 adds a `fenced_out_empirical_modules` field. → **WP-N5** |
| WP-5 | Blocker 8: Electron-inertia term omitted from generalized Ohm, not declared as a bounded-out omission | still_open | Codex A-6 / WP-N5: "Add a bounded-out declaration for omitted electron inertia or implement it from a local source packet." → **WP-N5** |
| WP-5 | Blocker 2: EOS unclosed (no QEOS/tabular EOS, no FP-runner EOS call) | still_open | `closure_packet.py` keeps `eos_thermodynamics` `blocked`. Correct. Codex WP-N5 closure packet. → **WP-N5** |
| WP-5 | Blocker 3: Radiation losses unclosed in the FP path | still_open | `bremsstrahlung.py` is NRL-correct but only the legacy MHD engine imports it; `hybrid_loop` has no radiation sink. Packet `blocked`. → **WP-N5** |
| WP-5 | Blocker 4: Anomalous resistivity has NO local source formula | still_open | vacuum-2004 only describes it; no DPF-scoped closure exists. Packet `blocked` — correct; WP-5 explicitly says "Do NOT invent one". → **WP-N5** (blocked, source-absent) |
| WP-5 | Blocker 5: Restrike — no KR source at all | still_open | Packet `blocked`. Correct. → **WP-N5** (blocked, source-absent) |
| WP-5 | Blocker 6: Ablation/impurities — only the `# EMPIRICAL` constant-efficiency scaffold; no KR ablation source packet | still_open | Packet `blocked`. Correct. → **WP-N5** |
| WP-5 | Blocker 7: Beam-target coupling — kinetic-yield history only; no mechanism separation/stopping/spectrum/anisotropy/detector | still_open | Packet `blocked`. Shared boundary with WP-N6. → **WP-N5** (with **WP-N6**) |
| WP-5 | Blocker 10: `electron_energy.py` `source_lines` `1074-1097/1226-1240/1267-1278` not opened line-by-line | still_open | Citation follow-up verification. → **WP-N5** |
| WP-5 | `physics_closure` packet fail-closed, telemetry-driven; every effect hard-codes `can_support_first_principles_acceptance: False`; 9 active closures source-backed; no reduced-model leakage | fixed | `closure_packet.py:270-291` hard-codes False per effect; `:231-267` top-level cannot accept. WP-5 §a and `claim_safety_sweep` §c confirm. Not contested. |

## WP-6 / SSR-009 — Neutron Mechanism And Detector Authority

| WP | Finding | Status | Evidence |
| --- | --- | --- | --- |
| WP-6 | Blocker 1: No `tests/test_first_principles_neutron_authority.py`; mechanism-separation discrimination logic untested | fixed | `tests/test_first_principles_neutron_authority.py` exists; the WP-6 patch-text suite (scalar-only rejection, Lee-reduced rejection, cross-scope rejection, candidate-PIC non-promotion, detector/UQ gating) was applied. All pass at HEAD. |
| WP-6 | Blocker 2: Accept path of `build_mechanism_separated_neutron_packet` is structurally inert (hardcoded `False` literals + unconditional `missing.update(BLOCKING_…)`); detector/UQ not consumed as a real gate | still_open | `neutron_authority.py:199-200, 258-259` unchanged. Codex WP-N6: either document the permanent hard block, or refactor to consume a `detector_response` packet and a `comparator_uq` packet. → **WP-N6** |
| WP-6 | Blocker 3: `kinetic_neutron_yield_authority_status` (the only fn that consumes detector/UQ evidence) is not wired into the runner | still_open | `runner.py:1040-1051` wires only `build_mechanism_separated_neutron_packet`. Two divergent neutron-authority code paths must be reconciled. → **WP-N6** |
| WP-6 | Blocker 4: No mechanism-separated neutron *history* produced — `kinetic_yield.py` emits a single total D-D rate (`not_mechanism_separated`) | still_open | Only the requirement channels exist; mechanism-separated thermonuclear/beam-target histories, beam/ion distribution, spectrum/anisotropy, detector/activation/TOF response are not implemented. Honestly fail-closed. → **WP-N6** |
| WP-6 | Blocker 5: Context citation `open-access-…-ed196711.md:93-141,…` not individually opened | still_open | File exists, ranges in-bounds, context-only role. Spot-check folded into WP-N6 source review. → **WP-N6** |
| WP-6 | Neutron packet mechanism-separated, fail-closed, source-cited; scalar-yield-only provably cannot accept; Lee/hybrid-PIC reduced outputs held as comparator/diagnostic baselines; 17/18 citations verified | fixed | `neutron_authority.py` `REQUIRED_/BLOCKING_NEUTRON_AUTHORITY_CHANNELS`, `can_support_*` hardcoded False; runner passes candidate kinetic-yield telemetry at `runner.py:1040-1051`. WP-6 verdict `accept_engineering_progress`. Not contested. |

## WP-7 / SSR-010-013 — Comparator, Numerical Fidelity, Certificate, Generalization

| WP | Finding | Status | Evidence |
| --- | --- | --- | --- |
| WP-7 | B-1: `tests/test_first_principles_certificate_negative_controls.py` does not exist (draft/blocked/cross-scope/missing-UQ/missing-review/hidden-limiter/reduced-model/app-only controls) | fixed | `tests/test_first_principles_certificate_negative_controls.py` exists; the WP-7 patch-text suite was applied. All pass at HEAD. |
| WP-7 | B-2: `manifest.py` does not record command/`argv` provenance; no `source_packet_hashes` distinct from `source_index_references` | still_open | Partial fix landed: `manifest.py:84` adds a `command_argv` field, `:202-209` add `git_commit`, `source_truth_index_sha256`, `input_deck_sha256`, `artifact_schema_version`, `artifact_generation_commit`. Codex A-8 still requires `source_packet_hashes`, environment/config hash, and certificate-test wiring. Provenance is improved but not yet complete. → **WP-N7** |
| WP-7 | B-3: Accept branches of `_target_scope_matches` / `_accepted_channels_from_targets` never exercised (no positive-path test) | still_open | All WP-7 packets are hardcoded fail-closed, so the accept logic has no regression protection. → **WP-N7** |
| WP-7 | B-4: `[partial-verify]` numerical-method KR citations (`numerical_fidelity.py`, `experimental_numerics.py`, `limiter_readiness.py`) not re-opened line-by-line | still_open | Method-context role labels only, no physics claim; low severity. → **WP-N7** |
| WP-7 | WP-7 modules honest and fail-closed: certificate gate has no accepting path, `manifest.py.__post_init__` raises on non-candidate status, no experimental waveform used as drive/fit, cross-scope cannot pass, general-DPF unreachable | fixed | `certificate_gate.py:114-164` (no accept branch); `manifest.py:66-74` raises `ValueError` on accepted status; `current_waveform_comparator.py:185-190` `experimental_waveform_used_as_drive/fit: False`. WP-7 §a/§c/§d confirm. Not contested. |

## claim_safety_sweep — Status And Claim Safety (Honesty Audit)

| WP | Finding | Status | Evidence |
| --- | --- | --- | --- |
| sweep | Overall verdict: **clean** — no `reject_overclaim`; no doc/README/UI/CLI/dossier/code claims acceptance while the certificate gate is blocked | fixed | This is a confirmation finding, not a gap. The honesty discipline holds at HEAD: `certificate_gate.py` hardcodes `can_write_accepted_certificate: False`; README/CLI/`app_mhd.py`/dossier all state engineering-candidate scope. Carry forward as the standing honesty baseline. |
| sweep | No reduced-model authority leak into the FP path; `deck.py:30-44` `REDUCED_MODEL_AUTHORITY_FIELDS` blocklist active | fixed | Confirmation finding. `deck.py` blocklist intact. No gap. |
| sweep | No hidden/undisclosed floor or clip found; all floors are named validated parameters with dedicated telemetry | fixed | Confirmation finding — and the WP-4 B-WP4-2 hidden-floor concern is now also resolved (electron-density floor telemetered, see WP-4 above). No gap. |
| sweep | Process observation: `telemetry.apply_floor()` is not used in the FP/fields trees, diverging from the CLAUDE.md numerical-coding rule | still_open | Non-blocking process observation; same item as WP-4 B-WP4-3. Policy reconciliation: either create `dpf.telemetry.apply_floor` and migrate, or update CLAUDE.md to bless the named-parameter + telemetry-packet substitute. → **WP-N5** (policy note) |
| sweep | Process observation: `experimental_waveforms.py:54-61` carries a SYNTHETIC reconstructed waveform array | still_open | Non-blocking; already gated (README:106-107, 157-160) and never used as an FP drive. Keep gated; revisit when an accepted digitized waveform exists. → **WP-N7** (comparator/source-target) |

---

## Summary Count

Total distinct WP findings classified: **63**

| Status | Count | Notes |
| --- | --- | --- |
| `fixed` | 28 | Resolved in code at HEAD; each verified against a current code line and, where applicable, a passing test. Includes the 6 confirmation findings (WP-7 fail-closed posture, `claim_safety_sweep` clean verdict, blocklist, no-hidden-floor). |
| `still_open` | 32 | Not resolved; each mapped to a Codex next-step work package WP-N0…WP-N7. |
| `superseded` | 1 | WP-1 G1/B1 negative-test suite — re-scoped by Codex into WP-N1. |
| `rejected_after_review` | 2 | WP-3 Blocker 9 (`source_geometry.py` LLNL packet is correctly scoped, not a defect) and WP-4 B-WP4-3 is *partially* superseded — counted under `still_open` for the unresolved policy contradiction. |

Verified-fixed highlights (the WP docs were stale on these — they are done at HEAD):
- **WP-2 startup acceptance-gate honesty defect** — fixed (`startup_bvp.py:214-223`).
- **WP-3 mask hash + projection error + rod-resolution gate** — fixed (`runner.py:2604/2621/2766`); 3 of the 4 Codex xfails.
- **WP-4 conservation `passed` false-green** — fixed; `passed` key removed entirely (`runner.py:2203-2241`).
- **WP-4 hidden electron-density floor** — fixed; telemetered (`hybrid_loop.py:866`).
- **WP-4 checkpoint grid/deck guard** — fixed; 4th Codex xfail resolved (`state_checkpoint.py`).
- **All six negative-control / integrity test files** — created and passing
  (`test_first_principles_startup_bvp.py`, `_geometry.py`, `_long_run_integrity.py`,
  `_closures.py`, `_neutron_authority.py`, `_certificate_negative_controls.py`).

## What Remains For The Physics Round (WP-N1 … WP-N8)

The still-open findings group as follows. Per the Codex instruction, **WP-N0
evidence hygiene must precede the physics work** so the next audit can trust the
submitted artifacts.

- **WP-N0 — Evidence hygiene** (1 finding): regenerate/quarantine the stale
  `100ns/1ns/12us` power-port artifacts; add `scripts/audit_first_principles_artifacts.py`.
- **WP-N1 — Power-port closure** (8 findings, the #1 physics blocker): named
  Auluck `Ω` runtime domain (G5); 5-term energy ledger — wall Poynting +
  electrode/interface work (G3); reviewed residual policy (G8); power-port
  negative-test suite (superseded G1); first-step/default-mode test pins (G2,G6);
  citation re-verification (G4, Beresnyak B8).
- **WP-N2 — Startup BVP and handoff** (3 findings): one source packet per of the
  9 startup channels; a real breakdown/flashover model or reviewed imported PIC
  sheath; an explicit `startup_handoff_interval_s` wired into the field/PIC loop.
- **WP-N3 — PF-1000 geometry and materials** (5 findings): alumina insulator
  mask; hollow-anode bore; electrode backplate / source-interface mask;
  vacuum-chamber wall mask; surface the Krauz-vs-Akel geometry disagreement.
- **WP-N4 — Long-run numerics / restart / 12 µs** (2 findings): segmented
  12 µs source-sign run driver with `lagged_field_work` carried across
  checkpoints; per-step PML-removed-energy ledger term.
- **WP-N5 — Physics closures** (8 findings): EOS, radiation losses,
  ablation/impurities, anomalous resistance, restrike, beam-target closure
  packets; bounded-out electron-inertia declaration; `fenced_out_empirical_modules`
  field; two imprecise citation repoints; explicit-path heat-flux floor count
  re-check; the Ohmic-CFL-disabled physics review; the `apply_floor()` policy
  reconciliation.
- **WP-N6 — Neutron mechanism and detector** (4 findings): mechanism-separated
  thermonuclear/beam-target histories; reconcile the inert
  `build_mechanism_separated_neutron_packet` accept path with
  `kinetic_neutron_yield_authority_status`; wire detector/UQ as a real consuming
  gate; context citation spot-check.
- **WP-N7 — Comparator, UQ, certificate, provenance** (4 findings): complete
  manifest provenance (`source_packet_hashes`, env/config hash); positive-path
  comparator/scope tests; `[partial-verify]` numerical-method citations; the
  SYNTHETIC reconstructed-waveform follow-up.
- **WP-N8 — Multi-machine candidate decks**: no WP-1…7 finding maps here; it is a
  forward-looking Codex work package, not a reconciliation gap.

The physics round should begin only after WP-N0 closes, as the Codex audit
instructs ("Do not skip earlier hygiene work to chase physics features").
