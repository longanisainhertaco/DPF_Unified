# SSR Audit Phase 3 — Status And Claim Safety Sweep

Date: 2026-05-18
Repo: `/Users/anthonyzamora/dpf-unified`
Branch: `codex/corpus`
Spec: `docs/FIRST_PRINCIPLES_SIMULATOR_SSR_AND_IMPLEMENTATION_AUDIT_SPEC_2026_05_18.md`
Scope: honesty / overclaim audit only. READ-ONLY. No code or existing docs modified.

## (a) Overall Verdict

**clean** — no `reject_overclaim` finding. No doc, README, UI, CLI, dossier, or
code path asserts the simulator is accepted / validated / first-principles-ready
while the certificate gate is blocked. Every readiness-class string found is
either (i) an honest fail-closed status, (ii) a hard-coded `False`, or (iii) a
gate input field name. The honesty discipline is holding.

Caveat: this is an honesty audit, not a physics audit. The candidate power port
(`lagged_volume_j_dot_e`) and seeded-layer startup remain physics blockers; they
are correctly disclosed (see sections c/e) and are out of scope for a verdict.

## (b) Overclaim Findings

Search: `rg -n "accepted|validated|first-principles ready|full first-principles|can_support_first_principles_acceptance" src docs tests` — 3824 raw hits. Bulk are `validated_quantities` / `validated_features` field names in `src/dpf/validation/quality_assessment.py` (not claims) and `*_not_validation` status strings (honest fail-closed). Representative + every potentially-load-bearing claim judged below.

| File:line | Text (abbrev) | Verdict | Reasoning |
| --- | --- | --- | --- |
| `src/dpf/first_principles/certificate_gate.py:160-163` | `can_write_accepted_certificate: False`, `can_release_first_principles_claim: False`, `can_support_first_principles_acceptance: False` | honest | Hard-coded `False`. Gate cannot emit an accepted certificate. Fail-closed by construction. |
| `src/dpf/first_principles/certificate_gate.py:134,137` | `status: blocked_first_principles_certificate_not_available`, `decision: do_not_write_accepted_first_principles_certificate` | honest | Explicit blocked status + negative instruction. |
| `src/dpf/validation/first_principles_limiters.py:213` | `"can_support_first_principles_acceptance": not active_acceptance_blockers` | honest | Computed from blockers; cannot be True while blockers present. Baseline: all blocked. |
| `src/dpf/validation/first_principles_limiters.py:287,319` | `"can_support_first_principles_acceptance": False` | honest | Hard `False` defaults. |
| `src/dpf/validation/first_principles_mhd.py:177,256,320,363,392,412,456` | `can_support_first_principles_acceptance: False` / `== "validated"` | honest | All emit `False` or gate on a `passed` flag that requires accepted evidence. `:424` returns `True` only inside a branch already guarded by accepted upstream evidence. |
| `src/dpf/validation/hybrid_pic_3d_validation_packet.py:61-72` | `status: accepted if accepted else blocked`; `_accepted()` requires `passed is True and status in {accepted,validated}` | honest | Acceptance is derived from per-channel accepted evidence; with all FP gates blocked it returns `blocked`. |
| `src/dpf/cli/main.py:142,1166,1285,1292` | `can_support_first_principles_acceptance` echoed from packet / `False` | honest | CLI relays packet value; literal-`False` fallbacks at `:1242,1292`. No CLI override. |
| `docs/DPF_REQUIREMENTS_BASELINE.md:59,63,66-69,76,82` | requirements marked `blocked` / `partial` / `planned` with "accepted ... remains blocked" | honest | Requirements register; consistently states blocked/partial, never claims acceptance. |
| `docs/FIRST_PRINCIPLES_BLOCKER_SOURCE_SEARCH_2026_05_15.md:25-127` | "seeded ionized layer is not accepted startup physics"; FP-5 "remains blocked" | honest | Blocker analysis; explicitly keeps `seeded_layer` rejected. |
| `docs/phase1_agent_dialogue.md:1218` | "8.9/10 fidelity claim is RETRACTED ... 0/10 validated" | honest | Self-correcting retraction of a prior overclaim — discipline working. |
| `src/dpf/validation/experimental_waveforms.py:54-61` | "SYNTHETIC — reconstructed waveform ... rescaled to Akel Table 1"; "NOTE: Replace with accepted digitized data" | honest (flagged) | Block is labeled SYNTHETIC and reconstructed; not used as FP drive. Not an overclaim, but a reconstructed array — keep gated (it already is, README:106-107, 157-160). |
| `src/dpf/validation/kr_corpus.py:843-851` | "User-validated arXiv ... not accepted validation targets until typed ... packets ... reviewed" | honest | "User-validated" = source ingested; explicitly denies target/validation acceptance. |
| `README.md:5` | "it is not yet an end-to-end predictive DPF simulator" | honest | Front-matter disclaimer. |
| `README.md:98-101,109-110,141-143` | "Predictive-readiness gate: blocked by default"; "Not yet validated against ... experimental data"; engineering tests "not scientific validation evidence" | honest | README explicitly fail-closed. |
| `README.md:218-221` | first-principles CLI output "is an engineering probe artifact ... not scientific validation evidence and does not accept the Akel draft waveform packet" | honest | Accurate scope statement. |

No row carries an `overclaim` verdict.

## (c) Reduced-Model-Leak Findings

Search: `rg -n "lee|snowplow|fcr|empirical|fit|scaling" src/dpf/first_principles src/dpf/fields`. Spec Non-Negotiable Rule 4: reduced models may be baselines/comparators only.

| File:line | Context | Verdict | Reasoning |
| --- | --- | --- | --- |
| `src/dpf/first_principles/deck.py:30-44` | `REDUCED_MODEL_AUTHORITY_FIELDS = {lee_model, lee_fit_factors, lee, radpf, snowplow, fcr, fcr_eff, beam_fraction, empirical_yield, ...}` | honest (defensive) | This is a BLOCKLIST. The deck actively rejects these keys from carrying authority. Reduced-model authority leak is structurally prevented at the input contract. |
| `src/dpf/first_principles/source_targets.py:354-374,464-476,536` | "Lee_model_baseline_and_fit_boundary_context", "Lee 5/6-phase fits are baseline/comparator context only", "baseline-only Lee fit metadata", "accepted scaling laws without target extraction and review" (blocked) | honest | Lee outputs explicitly tagged baseline/comparator-only; scaling-law targets explicitly blocked pending review. No drive role. |
| `src/dpf/first_principles/spatial_field_temperature.py:12,72-75,153` | `lee_output_*` density/radius/length scalars; `"lee_output_scalars_and_other_scope_diagnostics_cannot_support_..."` | honest | Lee outputs named only to state they CANNOT support the comparator. Negative use. |
| `src/dpf/first_principles/neutron_authority.py:12,105-108` | `lee_thermonuclear_and_beam_target_model_text`, `lee_beam_target_formula_context`, `average_yield_series_fit_context` | honest | Listed as baseline/context for the neutron packet; spec SSR-009 permits Lee neutron outputs as comparator baselines. Packet remains blocked. |
| `src/dpf/first_principles/same_scope.py:66` | `pinch_geometry_lee_output` | honest | A same-scope target label; comparator binding, not a driver. |
| `src/dpf/first_principles/waveform_phase.py:67` | `current_fit_through_dip_text` | honest | Source-text reference name; comparator phase context. |
| `src/dpf/first_principles/current_waveform_comparator.py:4,144,187,257` | "without using the target as a fit, drive"; `experimental_waveform_used_as_fit: False` | honest | Comparator explicitly asserts the experimental waveform is NOT a fit/drive — matches spec SSR-010. |
| `src/dpf/first_principles/generalization.py:70-73,128` | Soto scaling-laws KR file; `cross_device_drive_and_energy_density_scaling_matrix` | honest | Generalization packet; scaling matrix is requirement material, packet blocked (`:305` `single_scope_engineering_candidate_not_generalized`). |
| `src/dpf/first_principles/inverse_parameters.py:744-849` | `fitted_pressure_Pa`, "GV verified input deck fitted pressure" | honest | "fitted pressure" is the deck's published gas pressure value (source_targets uses same term for digitized published columns); an input, not a closure knob. |
| `src/dpf/first_principles/inverse_calibration.py:5,177-271,320` | "intentionally non-promoting: a best fit is not treated as a closure"; `acceptance_policy: non_promoting_experimental_inverse_fit`; `calibration_is_non_promoting_experimental_fit: True` | honest | Inverse calibration explicitly non-promoting; cannot leak into FP authority. Matches CLAUDE.md "calibration = bug" rule. |
| `src/dpf/first_principles/deck.py:1315,1331` | "GV bundle supplies ... fitted gas" | honest | Deck-construction comment; published GV deck values. |
| `src/dpf/first_principles/runner.py` (5x) | `reduced_models_used: False` (lines 343,1213,2126,2674,2683) | honest | Runtime asserts no reduced model used; CLI relays it (`main.py:1165`). |
| `src/dpf/first_principles/experimental_shot.py:244,485,552` | `_particle_scaling_projection` | honest | PIC macro-particle count scaling (numerical), unrelated to reduced physics models. |

No reduced-model authority leak into the first-principles predictive path. `# EMPIRICAL:` markers: **none found** in `src/dpf/first_principles` or `src/dpf/fields` — no hidden empirical knobs flagged or unflagged in those trees. The deck blocklist (`deck.py:30-44`) is the strongest single guard.

## (d) External-Surface Claim Findings (CLI / README / dossier / app)

| Surface | Finding | Verdict |
| --- | --- | --- |
| `README.md` | Line 5 "not yet an end-to-end predictive DPF simulator"; predictive gate "blocked by default" (98); MHD "not yet validated against ... experimental data" (110); engineering tests "not scientific validation evidence" (141-143); first-principles CLI "engineering probe artifact ... not scientific validation evidence" (218-221) | honest — README never claims FP readiness; explicitly blocked. |
| `src/dpf/cli/main.py` | `first-principles-3d` / `experimental-whole-shot` route through package-native `run_first_principles_3d_deck` (`:1016,1097`); `command_status` = `package_native_first_principles_3d_engineering_run` / `experimental_whole_shot_engineering_candidate_run`; relays `can_support_first_principles_acceptance` and `reduced_models_used` from packet, no override | honest — CLI carries packet status verbatim, "engineering" labels, no readiness claim. |
| `app_mhd.py` | `_apply_first_principles_engineering_bounds` docstring (`:288-292`) "must not be interpreted as accepted physics closure"; "the run cannot support first-principles acceptance" (`:244`); `first_principles_total_yield_authority: "blocked"` (`:554,1099`); `can_support_first_principles_neutron_yield: False` (`:1111`); `can_support_first_principles_startup: False` (`:3657`); bound changes routed through `limiter_event` telemetry (`:345,398,423,453`); comment "not a hidden first-principles closure" (`:3774`) | honest — app explicitly blocks FP authority, routes engineering bounds through limiter telemetry, never claims acceptance. |
| `docs/FIRST_PRINCIPLES_ENGINEERING_FIRM_DOSSIER_2026_05_16.md` | Line 7 "does not promote any neutron-yield, same-scope, or generalized" claim; whole-shot status `experimental_whole_shot_candidate_not_validation` (86); "does not yet run the requested microsecond-scale whole shot" (121); power port "lagged J.E feedback without accepted sign/time-centering" (180); "accepted first-principles power-port authority remains blocked" (905); residual tolerance `not_attached` (534); closing list of blockers — certificate gate + second-scope generalization still open (981) | honest — dossier is explicitly an engineering-candidate review packet; enumerates blockers; makes no accepted/ready claim while certificate gate blocked. |
| `src/dpf/server/*` | `app.py`, `readiness.py`, `simulation.py` present; no `first-principles ready / accepted / validated` assertion surfaced for the FP path (search 1 produced no server hits) | honest — server does not assert FP readiness. |

No external surface claims readiness while `certificate_gate` is blocked. Rejection Criterion "Certificate or docs claim readiness while upstream packets remain blocked" is **not triggered**.

## (e) Hidden-Floor Findings

Searches: `rg -n "np\.maximum|np\.clip|1e-|floor|back.?emf|current_floor"` across `src/dpf/first_principles` and `src/dpf/fields`. Note: `telemetry.apply_floor()` (the project-mandated floor helper) is **not used anywhere** in these two trees — but the FP/fields runtime uses an alternative, explicit disclosure pattern instead. Each floor below is judged on whether it is disclosed in telemetry and physically justified.

| File:line | Floor / clamp | Routed through telemetry? | Verdict |
| --- | --- | --- | --- |
| `src/dpf/fields/hybrid_loop.py:190,203,660,712` | `np.maximum(electron_density, density_floor_m3 / 1.0)` | Yes — `density_floor_m3` is a named, validated (`>0`, `:160-161`) deck parameter; floor-contact reported via `hybrid_stepper` numerical-floor telemetry | disclosed — numerical vacuum-density floor, named parameter, contact-counted. Not hidden. |
| `src/dpf/fields/hybrid_stepper.py:264-285,318-337` | electron-density floor gating | Yes — emits `numerical_electron_density_floor_m3`, `excluded_numerical_floor_cell_count`, `max_excluded_numerical_floor_current_A_m2`, `can_support_first_principles_acceptance: False`, limitation string "vacuum-floor cells are not a physical electron-fluid conductor" | disclosed — exemplary; full floor accounting in telemetry. |
| `src/dpf/fields/electron_energy.py:732-734,797,862,987,1018-1020` | energy/temperature floor (`energy = max(energy, energy_floor)`); `raw_min_temperature_before_floor_K` recorded | Yes — `temperature_floor_K` named param (validated `>0`, `:130-131`); emits `temperature_floor_contact_count`, `raw_min_temperature_before_floor_K`, `all_cells_at_numerical_floor` | disclosed — pre-floor raw value preserved in telemetry; contact counted. Not hidden. |
| `src/dpf/fields/electron_energy.py:662-665,680,683,721-722,954` | `np.maximum(kappa, 0.0)`, `max(..., 1e-300)`, `np.maximum(heat_capacity, 1e-300)` | Partial | benign — non-negativity of conductivity and divide-by-zero guards on a strictly-positive heat capacity; not a physics repair. Acceptable. |
| `src/dpf/fields/conductivity.py:186,199` | `ne_safe = np.maximum(ne, 1.0)`; `sigma = np.where(ne>0, sigma, 0.0)` | Partial | benign — `ne` already validated `>= 0` (`:182-183`); `ne_safe` only guards Coulomb-log/`1/ne` divides, and the final `np.where` zeroes conductivity where `ne==0`, so the floor cannot fabricate vacuum conduction. Defensible; ideally would emit a contact count. |
| `src/dpf/fields/circuit_boundary.py:73,109-122,255,263` | `radius_floor_m` on `B_theta = mu0 I / (2 pi max(r, radius_floor))` | Yes — `radius_floor_m` is a named field, validated `>0`, serialized in packet (`:215`) | disclosed — geometric `1/r` singularity guard at the axis; named, validated, serialized. Not hidden. |
| `src/dpf/fields/ionization_transport.py:112,154-157,210,219,227,269-270` | `np.clip(delta, -ion, neutral)`, `np.maximum(..., 0.0)`, `np.maximum(Te, 1e-12)` | Partial | benign — clamps ionization exchange to conserve particle inventory (cannot create mass) and floor `Te` only inside rate-coefficient evaluation. Physically required, not a result repair. |
| `src/dpf/fields/maxwell_3d.py:500` | `np.clip((pml_cells-distance)/max(pml_cells,1), 0, 1)` | n/a | benign — PML absorbing-layer profile normalization; standard, not a physics floor. |
| `src/dpf/fields/particle_boundaries.py:132-137` | `np.floor` / `np.clip(cell_idx, 0, n-1)` | n/a | benign — particle-to-cell index computation; `floor` here is array indexing, not a state floor. |
| `src/dpf/first_principles/runner.py:548,832` | `np.maximum(deuterium_density * ionization_fraction, 1.0)` | Implicit | disclosed-by-design — initial electron-density vacuum floor of 1 m^-3; the matching `_resolved_plasma_*` gates (hybrid_stepper/electron_energy) treat `n_e <= floor` cells as excluded vacuum and report them. Consistent with the disclosed floor regime. |
| `src/dpf/first_principles/runner.py:1632` | `np.clip(ionization_fraction, 0, 1)` | n/a | benign — physical bound on a fraction. |
| `src/dpf/first_principles/runner.py:1711` | `np.maximum(active_ion_temperature, 1.0)` inside `sqrt(3 k T / m)` | Partial | benign — guards thermal-speed sqrt against `T<=0`; PIC particle seeding only. |
| `src/dpf/first_principles/runner.py:148,316,624,898 / deck.py:418,447` | `density_floor_m3` default `1.0e20` (FP3D deck) / `1.0` (hybrid presets) | Yes — named deck field, validated `>0` (`runner.py:1445-1446`, `deck.py:433`), serialized into deck telemetry | disclosed — deck-level named parameter, not a hidden runtime clamp. |
| `src/dpf/first_principles/power_port.py:248` | `hidden_current_floor_or_back_emf_clip_rejection_required: True` | Yes | disclosed (control) — power port emits an explicit flag REQUIRING rejection of hidden `1/I` / back-EMF clips. This is the anti-hidden-floor guard itself. |
| `src/dpf/first_principles/limiter_readiness.py:65-71` | `state_mutating_floor_cap_clip` family: `density_floor`, `temperature_floor_or_cap`, `pressure_floor`, `back_emf_clip` | Yes | disclosed (inventory) — limiter-readiness packet enumerates exactly the floor/clip families that must be inventoried. Anti-hidden-floor mechanism. |

No hidden, undisclosed floor/clip/repair found. The FP/fields runtime does **not** use `telemetry.apply_floor()`, but substitutes a documented alternative: every state-mutating floor is a named, range-validated parameter whose contact count and (where applicable) pre-floor raw value are emitted in dedicated telemetry packets (`numerical_electron_density_floor_m3`, `temperature_floor_contact_count`, `raw_min_temperature_before_floor_K`, etc.), and the `limiter_readiness` / `power_port` packets actively inventory and demand rejection of hidden floors. The remaining `np.maximum(..., small)` calls are divide-by-zero / non-negativity guards on already-validated non-negative inputs, not result repairs. No `back_emf` clip is applied in code; the term appears only as a rejection-required flag and a limiter-family label.

Process note (not a finding against this audit's scope): the absence of `telemetry.apply_floor()` in these trees diverges from the project numerical-coding rule in `CLAUDE.md`. The substitute disclosure is honest and arguably more granular, but a maintainer may wish to confirm the PostToolUse floor-detection hook is satisfied by the named-parameter pattern.

## (f) Summary Counts

| Category | Count |
| --- | --- |
| Overclaim findings (`reject_overclaim`) | 0 |
| Honest fail-closed claim surfaces verified | 14 (table b) + 5 external surfaces (table d) |
| Reduced-model authority leaks into FP path | 0 |
| Reduced-model references verified as baseline/comparator/blocklist only | 13 (table c) |
| `# EMPIRICAL:` markers in first_principles/fields | 0 |
| Hidden / undisclosed floor or clip concerns | 0 |
| Floors verified as disclosed (named param + telemetry) | 7 |
| Floors verified as benign divide-by-zero / physical-bound guards | 8 |
| Process observations (non-blocking) | 2 (no `apply_floor()` in FP/fields trees; SYNTHETIC reconstructed waveform array — already gated) |

## Verdict Statement

**clean.** The DPF-Unified first-principles path's honesty discipline is holding.
The certificate gate hard-codes `can_write_accepted_certificate: False` and
`can_release_first_principles_claim: False`; every runtime artifact labels itself
`engineering_candidate_not_validation` / `*_not_validation`; the input deck
maintains an active reduced-model authority blocklist; README, CLI, app_mhd, and
the engineering-firm dossier all explicitly state engineering-candidate scope and
enumerate open blockers. No doc, UI, CLI, or code path claims acceptance,
validation, or first-principles readiness while upstream packets remain blocked.
No reduced model drives a first-principles predictive result. No hidden floor,
current-floor, back-EMF clip, or state repair was found; all floors are named,
validated parameters with dedicated telemetry. Phase 3 finds no rejection-grade
overclaim.
