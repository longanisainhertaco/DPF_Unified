# First-Principles 3-D DPF Engineering Test Dossier

Date: 2026-05-16

This note records the current experimental package-native 3-D first-principles
DPF simulator path for independent engineering review. It is not a validation
certificate and does not promote any neutron-yield, same-scope, or generalized
DPF-machine claim.

## Runnable Command

```bash
.venv312/bin/python -m dpf.cli.main first-principles-3d \
  --deck-preset pf1000_akel_16kv \
  --steps 1000 \
  --history-stride 100 \
  --max-step-results 8 \
  --output results/engineering_firm_first_principles_long_probe_2026_05_16.json
```

## Current Run Result

- Artifact: `results/engineering_firm_first_principles_long_probe_2026_05_16.json`
- Device deck: `pf1000_akel_16kv_1p2torr_shot_12581_engineering_candidate`
- Steps requested/completed: `1000 / 1000`
- Simulated time: `1.0e-10 s`
- Termination reason: `completed_step_budget`
- Finite-state check: passed for fields, particles, electron energy,
  ionization state, and circuit state
- Full step results retained: `8`
- History stride: `100`
- Final macro-particle count: `15200`
- Relative tracked energy change: `-5.6818832673290046e-09`
- Reduced models used: `false`
- Acceptance status: `engineering_candidate_not_validation`

## Python 3.12 Startup Probe

- Artifact:
  `results/first_principles_3d_py312_startup_probe_2026_05_16.json`
- Command interpreter: `.venv312/bin/python`, Python 3.12.13.
- Steps requested/completed: `20 / 20`
- Termination reason: `completed_step_budget`
- Startup packet status: `blocked_startup_bvp_packet_not_available`
- Candidate startup audit:
  `candidate_civ_paschen_breakdown_audit_engineering_only`
- Candidate breakdown mechanism for this PF-1000/Akel probe: `Paschen`
- Candidate breakdown time: `2.3524660435688997e-09 s`
- Candidate liftoff delay: `2.523524660435689e-07 s`
- Acceptance status: `engineering_candidate_not_validation`

## Python 3.12 PF-1000 Geometry Probe

- Artifact:
  `results/first_principles_3d_py312_pf1000_geometry_probe_2026_05_16.json`
- Command interpreter: `.venv312/bin/python`, Python 3.12.13.
- Steps requested/completed: `20 / 20`
- Termination reason: `completed_step_budget`
- Conductor mask mode: `pf1000_rod_hollow_projection`
- Mask source: `candidate_pf1000_rod_hollow_projection`
- Active conductor cells: `27`
- Cathode rods projected: `true`
- Hollow-anode inner radius supplied: `false`
- Insulator material: `alumina`
- Acceptance status: `engineering_candidate_not_validation`

## Python 3.12 Microsecond Request Probe

- Artifact:
  `results/first_principles_3d_py312_microsecond_request_probe_2026_05_16.json`
- Target time: `1.0e-6 s`
- Final simulated time: `2.0e-12 s`
- Duration request satisfied: `false`
- Termination reason: `completed_step_budget`
- Steps completed: `20`

## Experimental Whole-Shot Engineering Command

- Artifact:
  `results/experimental_whole_shot_pf1000_py312_2026_05_16.json`
- Command:
  `.venv312/bin/python -m dpf.cli.main experimental-whole-shot --deck-preset pf1000_akel_16kv --steps 20 --history-stride 5 --max-step-results 8 --target-time-s 1e-6 --output results/experimental_whole_shot_pf1000_py312_2026_05_16.json`
- Tool field: `dpf experimental-whole-shot`
- Command status: `experimental_whole_shot_engineering_candidate_run`
- Experimental packet status:
  `experimental_whole_shot_candidate_not_validation`
- Steps requested/completed: `20 / 20`
- Target time: `1.0e-6 s`
- Final simulated time: `2.0e-12 s`
- Duration request satisfied: `false`
- Termination reason: `completed_step_budget`
- Current deck timestep: `1.0e-13 s`
- Required steps at current timestep: `10000000`
- Explicit 3-D Yee vacuum-CFL timestep at `CFL=0.95`:
  `2.1590829252331814e-10 s`
- Required steps at the vacuum-CFL timestep: `4632`
- Current 20-step budget satisfies target: `false`
- Vacuum-CFL 20-step budget satisfies target: `false`
- Particle-growth projection from the 20-step run:
  `particle_growth_projection_high`, with an engineering-only linear
  projection of `224000102` particles at the current `dt_s` target and
  `103859` particles at the vacuum-CFL target.
- Candidate module records executed: `20`
- Unique candidate module names include:
  `package_native_3d_hybrid_em_pic_fluid_loop`,
  `startup_breakdown_liftoff_audit`,
  `pf1000_rod_hollow_conductor_projection`,
  `volume_j_dot_e_power_accounting`,
  `lagged_volume_j_dot_e_power_feedback`,
  `true_3d_grid`,
  `separate_electron_energy_source_terms`,
  `ionization_charge_state_transport`,
  `source_backed_partial_ionized_conductivity`,
  `plasmapy_community_formula_audit`, and
  `kinetic_neutron_yield_history`.
- Acceptance status: `engineering_candidate_not_validation`
- First-principles acceptance support: `false`

This command answers the experimental question directly: the package can run a
source-grounded whole-shot engineering attempt and produce a single review
packet. It does not yet run the requested microsecond-scale whole shot, and it
does not promote candidate physics into validation authority.

## Vacuum-CFL Nanosecond Probe

- Artifact:
  `results/experimental_whole_shot_pf1000_vacuum_cfl_ns_probe_2026_05_16.json`
- Command:
  `.venv312/bin/python -m dpf.cli.main experimental-whole-shot --deck-preset pf1000_akel_16kv --dt-policy vacuum-cfl --vacuum-cfl 0.95 --target-time-s 1e-9 --auto-step-budget --max-auto-steps 20 --history-stride 5 --max-step-results 4 --output results/experimental_whole_shot_pf1000_vacuum_cfl_ns_probe_2026_05_16.json`
- Steps requested/completed: `5 / 5`
- Target time: `1.0e-9 s`
- Timestep: `2.1590829252331814e-10 s`
- Final simulated time: `1.0795414626165907e-9 s`
- Duration request satisfied: `true`
- Termination reason: `target_time_reached`
- Acceptance status: `engineering_candidate_not_validation`

This proves the experimental lane can now compute and use a source-grounded
explicit-Maxwell duration plan for short horizons. It does not prove that a
microsecond PF-1000 shot is physically complete, numerically converged, or
validated.

## Vacuum-CFL Microsecond Experimental Shot

- Artifact:
  `results/experimental_whole_shot_pf1000_vacuum_cfl_1us_probe_2026_05_16.json`
- Command:
  `.venv312/bin/python -m dpf.cli.main experimental-whole-shot --deck-preset pf1000_akel_16kv --dt-policy vacuum-cfl --vacuum-cfl 0.95 --target-time-s 1e-6 --auto-step-budget --max-auto-steps 5000 --history-stride 500 --max-step-results 8 --output results/experimental_whole_shot_pf1000_vacuum_cfl_1us_probe_2026_05_16.json`
- Steps requested/completed: `4632 / 4632`
- Target time: `1.0e-6 s`
- Timestep: `2.1590829252331814e-10 s`
- Final simulated time: `1.0000872109680097e-6 s`
- Duration request satisfied: `true`
- Termination reason: `target_time_reached`
- Finite-state check: `true`
- Retained full step results: `8`
- Final macro-particle count: `78933`
- Final circuit current: `415959.7399182063 A`
- Final tracked field energy: `2610.76868743777 J`
- Relative tracked energy change: `0.053497034664624125`
- Final candidate ionization fraction maximum: `0.15007875531617856`
- Candidate cumulative neutron count: `5.812613362679801`
- Acceptance status: `engineering_candidate_not_validation`
- First-principles acceptance support: `false`
- Runtime numerical audit status:
  `experimental_numerical_runtime_audit_not_validation`
- Vacuum-CFL audit: `dt_s / stable_vacuum_dt_s = 1.0`,
  `dt_within_vacuum_cfl = true`
- Runtime history samples retained in the numerical audit: `8`
- History finite numeric samples: `true`
- History max `|div B|`: `0.6349675052723037 T/m`
- Audit troubleshooting priorities:
  `run_mesh_and_timestep_family_without_physics_changes`,
  `add_restart_reproducibility_artifact`, and
  `instrument_limiter_activation_counts_before_acceptance`.

This is the first package-native experimental 3-D run in this lane that reaches
a microsecond target horizon. It remains a coarse-grid, candidate-physics run:
startup is still a blocked BVP, PF-1000 geometry is still a candidate projection,
the power port is lagged `J.E` feedback without accepted sign/time-centering,
the current waveform target is not bound, and numerical fidelity/convergence,
restart, UQ, and mechanism-separated neutron authority remain blocked.

## Source-Truth Numerical Audit

The latest source-truth audit found enough local guidance to instrument
runtime numerical troubleshooting, but not enough same-scope evidence to
promote a PF-1000/GV run. The new
`experimental_numerical_runtime_audit_not_validation` packet cites:

- `docs/FIRST_PRINCIPLES_3D_HYBRID_PIC_REVIEW_2026_05_14.md:39,70-80`
  for Marder/Gauss-law control and Ohmic-CFL/nondominance risk.
- `docs/FIRST_PRINCIPLES_DIMENSIONALITY_SOURCE_SEARCH_2026_05_15.md:50-53`
  for the hybrid PIC loop, limiter, and sensitivity scope boundary.
- `docs/FIRST_PRINCIPLES_CLOSURE_SOURCE_SEARCH_2026_05_15.md:48-55,63-70`
  for closure support and remaining EOS/transport/radiation/ablation gaps.
- `docs/FIRST_PRINCIPLES_FINISH_LINE_PLAN.md:186-188`
  for limiter-zero and numerical-fidelity completion gates.
- `KnowledgeReference/fully-electromagnetic-hybrid-pic-fluid-dpf-neutron-yield-acb71fa9.md:410-424,468-606,1030-1068`
  for Marder, Ohmic CFL, refinement, and sensitivity context.
- `KnowledgeReference/particle-simulation-of-plasmas-review-and-advances-6d7355ba.md:456-530,671-705,744-755`
  for Yee/Courant and charge-conservation method context.

What this closes: the artifact now reports CFL budget, horizon coverage,
conservation snapshot, `div B`, retained-history trends, last-step limiter
signals, full-horizon limiter activation counts, restart gap, and convergence
gap in one packet.

What remains open: observable norms and tolerances, observed convergence,
multi-offset/full-horizon checkpoint-restart acceptance, backend/precision
parity, full-horizon limiter-zero proof, Marder/Ohmic-CFL nondominance
interpretation, an accepted power residual budget, and an independent
engineering review decision.

## Source-Truth Exhaustion And Module Vetting

The non-validating source-truth gates were refreshed after the runtime audit:

- `scripts/verify_first_principles_source_truth_exhaustion.py --date 2026_05_16 --refresh-index`
  wrote `docs/FIRST_PRINCIPLES_SOURCE_TRUTH_EXHAUSTION_2026_05_16.{json,md}`.
  Result: `exhausted=true`, `open_issue_count=0`, `indexed_file_count=1397`,
  and `actual_file_count=1397`.
- `scripts/verify_first_principles_module_source_vetting.py --date 2026_05_16`
  wrote `docs/FIRST_PRINCIPLES_MODULE_SOURCE_VETTING_2026_05_16.{json,md}`.
  Result: `strict_passed=true`, `total_modules=288`,
  `active physics modules needing source vetting=0`,
  `missing source-reference paths=0`, and `unsafe claim surfaces=0`.

These gates do not validate the simulator. They only prove the current local
source-truth index, user-verified ledgers, first-principles target packets,
and module authority classifications are internally consistent enough to keep
coding against the source of truth.

## Experimental Numerical Family Probe

- Artifact:
  `results/experimental_numerical_family_pf1000_timestep_probe_2026_05_16.json`
- Command:
  `.venv312/bin/python -m dpf.cli.main experimental-numerical-family --family timestep --deck-preset pf1000_akel_16kv --target-time-s 1e-9 --timestep-scales 1,0.5 --max-auto-steps 20 --history-stride 5 --max-step-results 4 --output results/experimental_numerical_family_pf1000_timestep_probe_2026_05_16.json`
- Family status:
  `experimental_numerical_family_probe_not_validation`
- Cases: `2`
- Duration-satisfied cases: `2`
- Pairwise comparisons: `1`
- Convergence decision:
  `not_assessed_no_accepted_tolerances`
- Tolerance claim: `false`
- Numerical acceptance support: `false`

Case summaries:

- `timestep_scale_1`: `dt_s=2.1590829252331814e-10`,
  `steps_completed=5`, `final_time_s=1.0795414626165907e-9`,
  `final_field_energy_J=0.25771076839392254`,
  `final_circuit_current_A=4376.544376849281`.
- `timestep_scale_0.5`: `dt_s=1.0795414626165907e-10`,
  `steps_completed=10`, `final_time_s=1.0795414626165907e-9`,
  `final_field_energy_J=0.9337677755397134`,
  `final_circuit_current_A=8113.262092369086`.

This closes the immediate tooling gap for timestep-family troubleshooting:
the package can now run same-deck, same-horizon variants and report
observable deltas. It does not close convergence because the source-truth
audit still lacks accepted PF-1000 tolerances, observed-order rules, limiter
limiter-zero interpretation, restart hashes, and independent review.

The same command now also supports mesh-family smoke probes. Artifact
`results/experimental_numerical_family_pf1000_mesh_probe_2026_05_16.json`
ran `5x5x5` and `6x6x6` PF-1000/Akel cases to the same requested
`1e-9 s` horizon, preserving the deck physical span while recomputing grid
spacing and vacuum-CFL timesteps. It produced one pairwise comparison and
kept the same `experimental_numerical_family_probe_not_validation` status.

## Experimental Reproducibility Probe

- Artifact:
  `results/experimental_reproducibility_pf1000_probe_2026_05_16.json`
- Command:
  `.venv312/bin/python -m dpf.cli.main experimental-reproducibility --deck-preset pf1000_akel_16kv --target-time-s 1e-9 --repeat-count 2 --dt-policy vacuum-cfl --vacuum-cfl 0.95 --max-auto-steps 20 --history-stride 5 --max-step-results 4 --output results/experimental_reproducibility_pf1000_probe_2026_05_16.json`
- Reproducibility status:
  `experimental_reproducibility_probe_not_validation`
- Reruns: `2`
- Duration-satisfied reruns: `2`
- Finite reruns: `2`
- State-observable hashes identical: `true`
- State-observable hash:
  `2ff168ede8091ec996450ff2f5d4f868edc2883d121848d34313b6378b95bd6f`
- Terminal state fingerprint hash:
  `1f113f1765586d5f3c05984e22b369c9168c48d16e2b6091614b0f33385c3a06`
- Checkpoint restart available: `false`
- Continued-run equivalence available: `false`
- First-principles acceptance support: `false`
- Full-horizon limiter steps observed per rerun: `5`

This closes only the deterministic-rerun tooling slice: two independent
package-native PF-1000/Akel runs with the same deck, seed, timestep policy,
history policy, and requested horizon produced identical hashes over terminal
simulation scalars, retained history summaries, last-step telemetry,
conservation telemetry, terminal state fingerprints, and packet statuses. It
does not close restart acceptance. The first-principles runner still needs
complete checkpoint file serialization, checkpoint readback into a live
package-native runner, split-run continuation, lagged-field-work and circuit
sequence preservation, and comparison against uninterrupted runs at multiple
restart offsets.

## Experimental Terminal-State Checkpoint Probe

- JSON artifact:
  `results/experimental_state_checkpoint_pf1000_probe_2026_05_16.json`
- Checkpoint artifact:
  `results/experimental_state_checkpoint_pf1000_terminal_2026_05_16.npz`
- Command:
  `.venv312/bin/python -m dpf.cli.main experimental-state-checkpoint --deck-preset pf1000_akel_16kv --target-time-s 1e-9 --dt-policy vacuum-cfl --vacuum-cfl 0.95 --max-auto-steps 20 --history-stride 5 --max-step-results 4 --checkpoint-output results/experimental_state_checkpoint_pf1000_terminal_2026_05_16.npz --output results/experimental_state_checkpoint_pf1000_probe_2026_05_16.json`
- Checkpoint status:
  `experimental_state_checkpoint_roundtrip_not_restart_acceptance`
- Checkpoint arrays written/read: `17`
- Checkpoint content hash:
  `f523f40a0f9aa7777f6c581675b0e15a28013ec1bf2475a92af12af850d61a85`
- Terminal state fingerprint hash:
  `1f113f1765586d5f3c05984e22b369c9168c48d16e2b6091614b0f33385c3a06`
- Write/read hashes match: `true`
- Live runner restart from checkpoint: `false`
- Continued-run equivalence: `false`

This closes terminal-state checkpoint roundtrip plumbing for a short
experimental PF-1000/Akel run: evolved fields, particles, electron-energy
state, ionization state, and circuit state are serialized to an NPZ artifact
and read back with a matching content hash. It does not close restart
reproducibility because the package-native runner still cannot load that
checkpoint and continue the same shot.

## Experimental Split-Continuation Probe

- Artifact:
  `results/experimental_split_continuation_pf1000_probe_2026_05_16.json`
- Command:
  `.venv312/bin/python -m dpf.cli.main experimental-split-continuation --deck-preset pf1000_akel_16kv --steps 6 --split-after-steps 3 --dt-policy vacuum-cfl --vacuum-cfl 0.95 --history-stride 1 --max-step-results 6 --output results/experimental_split_continuation_pf1000_probe_2026_05_16.json`
- Split-continuation status:
  `experimental_split_continuation_probe_not_restart_acceptance`
- Total steps: `6`
- Split: `3 + 3`
- Split total steps completed: `6`
- State fingerprints match: `true`
- Tracked observables match exactly: `true`
- State fingerprint hash:
  `03003c3658b7d5e0c52789dba53319753c2cfbbfdb5626a5243753ad6fd5fb8d`
- Final circuit current: `4538.248508471329 A`
- Lagged field-work preserved into segment two: `true`
- Checkpoint restart available: `false`

This closes live same-process split continuation for the package-native
first-principles 3-D path: a reusable session now preserves Maxwell/PIC state,
electron energy, ionization, circuit state, predictor-corrector loop state,
kinetic-yield history, global step offset, vector circuit-drive offsets, and
lagged `J.E` field-work feedback across segment boundaries. It does not close
checkpoint restart because the NPZ checkpoint is not yet loadable into a fresh
runner/session for continuation.

## Experimental Checkpoint-Restart Probe

- JSON artifact:
  `results/experimental_checkpoint_restart_pf1000_probe_2026_05_16.json`
- Checkpoint artifact:
  `results/experimental_checkpoint_restart_pf1000_midpoint_2026_05_16.npz`
- Command:
  `.venv312/bin/python -m dpf.cli.main experimental-checkpoint-restart --deck-preset pf1000_akel_16kv --steps 6 --split-after-steps 3 --dt-policy vacuum-cfl --vacuum-cfl 0.95 --history-stride 1 --max-step-results 6 --checkpoint-output results/experimental_checkpoint_restart_pf1000_midpoint_2026_05_16.npz --output results/experimental_checkpoint_restart_pf1000_probe_2026_05_16.json`
- Restart status:
  `experimental_checkpoint_restart_probe_not_validation`
- Total steps: `6`
- Checkpoint split: `3 + 3`
- Restart total steps completed: `6`
- Checkpoint write/read hash matched: `true`
- Loaded lagged field work: `true`
- Loaded predictor previous current: `true`
- Loaded kinetic-yield state: `true`
- State fingerprints match: `true`
- Tracked observables match exactly: `true`
- Final state fingerprint hash:
  `03003c3658b7d5e0c52789dba53319753c2cfbbfdb5626a5243753ad6fd5fb8d`
- Midpoint checkpoint content hash:
  `b5df1a0c70f24851ff284405d4debda4974a1d7176fa05c55780a5f4bbe7ee3d`

This closes checkpoint-loaded restart plumbing for the short PF-1000/Akel
experimental probe: a fresh session can load the midpoint NPZ checkpoint,
restore evolved fields, particles, closure states, circuit state,
predictor-corrector previous current, kinetic-yield counters, global step
offset, and lagged field work, then continue to the same terminal fingerprint
as the uninterrupted run. It remains non-validating until this is repeated at
multiple restart offsets and whole-shot horizons with accepted tolerances,
limiter-zero interpretation, backend/precision scope, and engineering review.

## Experimental Checkpoint-Restart Family

- JSON artifact:
  `results/experimental_checkpoint_restart_family_pf1000_probe_2026_05_16.json`
- Checkpoint directory:
  `results/experimental_checkpoint_restart_family_pf1000_2026_05_16/`
- Command:
  `.venv312/bin/python -m dpf.cli.main experimental-checkpoint-restart-family --deck-preset pf1000_akel_16kv --steps 6 --split-after-steps 2,3,4 --dt-policy vacuum-cfl --vacuum-cfl 0.95 --history-stride 1 --max-step-results 6 --checkpoint-dir results/experimental_checkpoint_restart_family_pf1000_2026_05_16 --output results/experimental_checkpoint_restart_family_pf1000_probe_2026_05_16.json`
- Restart-family status:
  `experimental_checkpoint_restart_family_probe_not_validation`
- Cases: `3`
- Split offsets: `2`, `3`, `4`
- Matching cases: `3`
- All cases match: `true`

This closes the short-horizon multi-offset restart tooling gap for the
experimental PF-1000/Akel runner. It remains non-validating because the same
family has not yet been run at whole-shot horizon, no accepted nonbitwise
tolerances or backend/precision parity matrix exists, and limiter-zero
interpretation plus engineering review are still open.

## Experimental Target-Time Checkpoint-Restart Family

- JSON artifact:
  `results/experimental_checkpoint_restart_family_pf1000_ns_probe_2026_05_16.json`
- Checkpoint directory:
  `results/experimental_checkpoint_restart_family_pf1000_ns_2026_05_16/`
- Command:
  `.venv312/bin/python -m dpf.cli.main experimental-checkpoint-restart-family --target-time-s 1e-9 --auto-step-budget --max-auto-steps 20 --split-after-steps 2,3,4 --dt-policy vacuum-cfl --vacuum-cfl 0.95 --history-stride 5 --max-step-results 4 --checkpoint-dir results/experimental_checkpoint_restart_family_pf1000_ns_2026_05_16 --output results/experimental_checkpoint_restart_family_pf1000_ns_probe_2026_05_16.json`
- Total steps selected from target time: `5`
- Cases: `3`
- Split offsets: `2`, `3`, `4`
- Matching cases: `3`
- All cases match: `true`
- Terminal state fingerprint:
  `1f113f1765586d5f3c05984e22b369c9168c48d16e2b6091614b0f33385c3a06`

This closes the command plumbing gap between physical target-time runs and
multi-offset checkpoint-restart probes. It is still an experimental
engineering packet, not acceptance evidence.

## Experimental Limiter-Zero Probe

- JSON artifact:
  `results/experimental_limiter_zero_pf1000_ns_probe_2026_05_16.json`
- Command:
  `.venv312/bin/python -m dpf.cli.main experimental-limiter-proof --target-time-s 1e-9 --auto-step-budget --max-auto-steps 20 --dt-policy vacuum-cfl --vacuum-cfl 0.95 --history-stride 5 --max-step-results 4 --output results/experimental_limiter_zero_pf1000_ns_probe_2026_05_16.json`
- Steps completed: `5`
- Final simulated time: `1.0795414626165907e-9 s`
- Inventory complete for completed steps: `true`
- Zero acceptance blockers observed: `false`
- Total acceptance-blocking limiter activations: `5`
- Acceptance-blocking counts:
  `conductivity_ohmic_cfl_limited_steps=5`,
  `conductivity_density_blend_applied_steps=0`,
  `marder_dominant_correction_steps=0`,
  `electron_temperature_floor_contact_steps=0`,
  `blocked_heat_flux_steps=0`
- Method-review counts:
  `marder_correction_steps=5`

This replaces a vague limiter-readiness blocker with a concrete runtime
ledger. The PF-1000 nanosecond probe is still blocked on Ohmic-CFL limiter
activation and Marder nondominance review before it can support any
limiter-zero claim.

## Experimental Combined-CFL Limiter Probe

- JSON artifact:
  `results/experimental_limiter_zero_pf1000_combined_cfl_short_probe_2026_05_16.json`
- Command:
  `.venv312/bin/python -m dpf.cli.main experimental-limiter-proof --target-time-s 5e-14 --auto-step-budget --max-auto-steps 10 --dt-policy combined-cfl --vacuum-cfl 0.95 --history-stride 1 --max-step-results 3 --output results/experimental_limiter_zero_pf1000_combined_cfl_short_probe_2026_05_16.json`
- Timestep selected by combined vacuum/Ohmic CFL policy:
  `1.839511686878276e-14 s`
- Steps completed: `3`
- Final simulated time: `5.5185350606348273e-14 s`
- Zero acceptance blockers observed: `true`
- Acceptance-blocking limiter activations: `0`
- Method-review counts:
  `marder_correction_steps=3`
- Marder nondominance observation:
  `max_relative_correction_linf=0.09902109928920329`,
  `nondominance_threshold=0.5`,
  `dominant_correction_steps=0`,
  `status=candidate_method_limiter_nondominant_observed`

2026-05-17 source-zero Marder rerun after pressure-density coupling:

- JSON artifact:
  `results/experimental_limiter_proof_pf1000_combined_cfl_source_zero_marder_2026_05_17.json`
- Command:
  `.venv312/bin/python -m dpf.cli.main experimental-limiter-proof --target-time-s 5e-14 --auto-step-budget --max-auto-steps 10 --dt-policy combined-cfl --history-stride 1 --max-step-results 3 --output results/experimental_limiter_proof_pf1000_combined_cfl_source_zero_marder_2026_05_17.json`
- PF-1000 gas pressure:
  `159.98684210526315 Pa`
- Startup/background density from ideal gas:
  `3.862599934409183e22 m^-3`
- Closure density floor:
  `3.862599934409183e22 m^-3`
- Marder factor scale:
  `0.0`
- Timestep selected by combined vacuum/Ohmic CFL policy:
  `8.411478422160001e-14 s`
- Steps completed:
  `1`
- Zero acceptance blockers observed:
  `true`
- Acceptance-blocking limiter activations:
  `0`
- Acceptance-blocking counts:
  `conductivity_ohmic_cfl_limited_steps=0`,
  `conductivity_density_blend_applied_steps=0`,
  `marder_dominant_correction_steps=0`,
  `electron_temperature_floor_contact_steps=0`,
  `blocked_heat_flux_steps=0`
- Marder decision:
  `steps_observed=0`,
  `status=candidate_method_limiter_requires_review`

This proves the Ohmic-CFL limiter activation is removable by using the
source-grounded explicit Ohmic relaxation timestep for a short target horizon.
The 2026-05-17 rerun also aligns the production Marder setting with the local
hybrid-PIC source, which reports `d = 0` for production runs after sensitivity
checks. This is not a validation claim; it is a cleaner experimental limiter
inventory. The computational implication remains: explicit small-timestep
continuation to a full microsecond PF-1000 shot would still require a very large
step count before mesh refinement, convergence families, or restart families.
The next physics/numerics decision is therefore explicit small-dt continuation
versus a source-grounded implicit/semi-implicit resistive update.

The same artifact now carries a candidate power-port residual budget:

- Residual budget status:
  `candidate_power_residual_budget_not_validation`
- Tracked energy delta:
  `-0.008413708303123713 J`
- Retained volume `J.E` work:
  `0.012864167379605766 J`
- Candidate `delta - retained J.E work`:
  `-0.02127787568272948 J`
- Candidate `delta + retained J.E work`:
  `0.004450459076482054 J`
- Full retained history available: `true`
- Accepted residual tolerance: `not_attached`

This closes the empty residual-budget placeholder with a concrete
non-promoting engineering ledger. It still cannot support acceptance until the
power-port sign convention, time-centering, interface/domain, electrode-work
partition, and residual tolerance are reviewed.

## GV PF-24 Waveform-Bound Experimental Probe

- Artifact:
  `results/experimental_whole_shot_gv_pf24_vacuum_cfl_1us_probe_2026_05_16.json`
- Command:
  `.venv312/bin/python -m dpf.cli.main experimental-whole-shot --deck-preset gv_pf24_krakow_16092202 --dt-policy vacuum-cfl --vacuum-cfl 0.95 --target-time-s 1e-6 --auto-step-budget --max-auto-steps 20000 --history-stride 2000 --max-step-results 8 --output results/experimental_whole_shot_gv_pf24_vacuum_cfl_1us_probe_2026_05_16.json`
- Steps requested/completed: `17598 / 17598`
- Target time: `1.0e-6 s`
- Timestep: `5.6825805481234365e-11 s`
- Final simulated time: `1.0000205248587623e-6 s`
- Duration request satisfied: `true`
- Finite-state check: `true`
- Final macro-particle count: `125`
- Current waveform comparison status:
  `engineering_current_waveform_comparison_not_validation`
- Current waveform overlap points: `10`
- Target waveform points: `649`
- Target waveform time range: `-0.5 us` to `6.0 us`
- Simulation comparison time range: `0.0 us` to `1.0000205248587624 us`
- Temporal coverage fraction of target: `0.15384931151673267`
- Current RMSE over overlap: `47.84087376616515 kA`
- Peak-current error fraction vs full target: `0.09800069763286687`
- Acceptance status: `engineering_candidate_not_validation`

This probe closes one experimental plumbing gap for a second-scope device: a
user-verified workbook current waveform can be bound to the package-native 3-D
run as a non-promoting comparator. It still does not use the waveform as a
drive, fit, reduced-model closure, or validation target.

## Experimental Inverse-Parameter Completion

- JSON artifact:
  `results/experimental_inverse_parameters_all_machines_2026_05_16.json`
- Command:
  `.venv312/bin/python -m dpf.cli.main experimental-inverse-parameters --scope all --include-gv-waveforms --output results/experimental_inverse_parameters_all_machines_2026_05_16.json`
- Packet status:
  `experimental_inverse_parameter_completion_not_validation`
- Machines covered: `12`
  (`PF-1000/Akel`, the three May 15 second-scope decks, and eight verified GV
  shots)
- Source policy:
  `reduced_models_used=false`, `gv_reduced_model_output_used=false`,
  `measured_waveforms_used_as_drive=false`, `algebraic_inference_only=true`
- Status counts:
  `known_source_value=155`, `direct_algebraic_inference=44`,
  `bracketed_source_range=2`, `waveform_derived_candidate=40`,
  `underdetermined_requires_additional_observable=25`,
  `contradiction_or_scope_mismatch=1`
- Concrete fills:
  PF-1000/Akel bank energy from `C,V` is `170496 J`;
  compact Chinese DPF current-implied executable inductance is
  `1.0000000000000002e-7 H`;
  GV PF-24 `16092202` measured workbook peak current is `401600 A`.
- Remaining unresolved parameters are explicitly named instead of fitted
  silently. Examples include PF-1000 hollow-anode inner radius, reviewed Akel
  waveform peak timing, startup initial states, compact-device resistance,
  surface-flashover state, and GV dynamic plasma impedance.
- The one contradiction/scope mismatch is IR-MPF-100: the listed source
  inductance and source theoretical peak current do not match the undamped LC
  formula at the listed maximum voltage. The packet therefore keeps source
  inductance and source current as separate observables and does not overwrite
  the deck.

This closes the algebraic deck-completion tooling gap for experimental runs:
where the source gives enough independent values, the code now solves the
missing circuit/pressure/geometry candidates directly and records the formula.
It does not solve non-unique physics. Startup, dynamic plasma impedance,
transport closures, detector UQ, neutron mechanism separation, and same-scope
histories remain independent first-principles requirements.

## Experimental Machine-Shot Family

The CLI now includes a registry runner:
`dpf experimental-machine-shot-family`. It builds the source-backed PF-1000,
May 15, and verified GV engineering decks, applies one target-time/timestep
policy, runs the cases that fit the explicit step cap, and records cases that
do not fit as blocked before run.

Short-horizon registry artifact:

- JSON artifact:
  `results/experimental_machine_shot_family_all_1ns_2026_05_16.json`
- Command:
  `.venv312/bin/python -m dpf.cli.main experimental-machine-shot-family --scope all --target-time-s 1e-9 --dt-policy vacuum-cfl --vacuum-cfl 0.95 --auto-step-budget --max-auto-steps 5000 --history-stride 5 --max-step-results 2 --include-gv-waveforms --output results/experimental_machine_shot_family_all_1ns_2026_05_16.json`
- Cases: `12`
- Completed cases: `12`
- Blocked cases: `0`
- Duration-satisfied cases: `12`
- Finite cases: `12`
- Embedded inverse-parameter summary:
  `machine_count=12`, `unresolved_parameter_count=25`,
  `contradiction_or_scope_mismatch_count=1`

Microsecond target with a deliberate explicit-step cap:

- JSON artifact:
  `results/experimental_machine_shot_family_all_1us_cap5000_2026_05_16.json`
- Command:
  `.venv312/bin/python -m dpf.cli.main experimental-machine-shot-family --scope all --target-time-s 1e-6 --dt-policy vacuum-cfl --vacuum-cfl 0.95 --auto-step-budget --max-auto-steps 5000 --history-stride 500 --max-step-results 2 --no-include-gv-waveforms --output results/experimental_machine_shot_family_all_1us_cap5000_2026_05_16.json`
- Cases: `12`
- Completed cases: `1`
- Blocked cases: `11`
- Duration-satisfied cases: `1`
- Finite cases: `1`
- Completed microsecond case:
  `pf1000_akel_16kv_shot_12581`, `4632` steps,
  final time `1.0000872109680097e-6 s`
- Blocked microsecond step requirements at the same vacuum-CFL policy:
  IR-MPF-100 `18099`, compact Chinese DPF `45321`,
  Willenborg/Hendricks `34759`, LPP-FF1 shots `17903`, PF-24 shots `17598`,
  PF-360 `11321`, Gemini `8175`, and OneSys `10942`.

This makes the current experimental state concrete: the registry can run short
3-D source-backed shots across every typed machine. A PF-1000 microsecond shot
runs under the 5,000-step cap. The other machines are not physics-blocked at
this layer; they are explicit-compute blocked by the chosen cap and can be run
by raising `--max-auto-steps` deliberately. None of these family artifacts is
validation evidence.

## Experimental Inverse Calibration

The CLI now includes `dpf experimental-inverse-calibration`. It takes
source/deck baseline values, builds a bounded candidate grid, runs each
candidate through the package-native first-principles shot path, scores the
retained circuit-current history against typed source observables, and reports
identifiability instead of silently choosing a fit.

- JSON artifact:
  `results/experimental_inverse_calibration_all_inductance_1ns_2026_05_16.json`
- Command:
  `.venv312/bin/python -m dpf.cli.main experimental-inverse-calibration --deck-preset all --parameters inductance --candidate-scales 0.75,1,1.25 --target-time-s 1e-9 --dt-policy vacuum-cfl --vacuum-cfl 0.95 --auto-step-budget --max-auto-steps 5000 --history-stride 5 --max-step-results 2 --output results/experimental_inverse_calibration_all_inductance_1ns_2026_05_16.json`
- Calibrations completed: `3`
  (`ir_mpf_100`, `compact_chinese_dpf`, `gv_pf24_krakow_16092202`)
- Candidates per calibration: `3`
- Parameter tested: static circuit `inductance`
- Source observables:
  IR-MPF-100 source theoretical peak current `1224000 A`;
  compact Chinese DPF source approximate delivered current `400000 A`;
  GV PF-24 workbook peak current `401600 A` and peak time `1.36e-6 s`.
- Identifiability result for all three:
  `underdetermined_or_correlated_on_candidate_grid`
- Best candidate for all three at the 1 ns horizon:
  `candidate_0000`, the `0.75x` inductance case.

This is the first actual infer-test-conclude loop in the first-principles
experimental lane. The conclusion is intentionally negative at this short
horizon: all candidates remain far from their peak-current or peak-time source
observables, and multiple inductance values are effectively indistinguishable
within the configured near-best margin. The correct conclusion from the
available test is therefore not “the inductance is solved,” but “this horizon
and single-parameter grid cannot identify the missing parameter.” The next
calibration step is longer-horizon or staged runs with additional fitted axes
and held-out observables.

Troubleshooting update:

- Dense current history is now retained independently from capped full step
  results, so long calibration runs do not silently collapse to only a few
  current samples.
- Inverse calibration now scores source-backed GV current-waveform shape as a
  non-promoting metric. The waveform is not used as a drive, not used as a
  reduced closure, and cannot promote validation.
- GV PF-24 `1.6e-6 s` artifact:
  `results/experimental_inverse_calibration_gv_pf24_inductance_waveform_1p6us_2026_05_17.json`
  remains `horizon_limited_requires_longer_run`; the best candidate peak is at
  the terminal sample and only `0.246` of the full workbook waveform is covered.
- GV PF-24 `6.0e-6 s` artifact:
  `results/experimental_inverse_calibration_gv_pf24_inductance_waveform_6us_2026_05_17.json`
  reaches full nonnegative shot-window waveform coverage after excluding the
  `-0.5 us` pretrigger baseline from coverage accounting.
- Combined GV PF-24 `6.0e-6 s` four-candidate artifact:
  `results/experimental_inverse_calibration_gv_pf24_inductance_waveform_6us_combined_2026_05_17.json`
  compares `0.75x`, `1.0x`, `1.25x`, and `1.5x` static inductance factors
  over the full positive-time waveform. It reports
  `underdetermined_or_correlated_on_candidate_grid`, not a solved parameter:
  `0.75x` is the best score (`0.2730`), but `1.0x` is within the near-best
  margin, giving an interval from `1.575e-8 H` to `2.100e-8 H`. The best
  candidate matches peak time (`0.285%` relative error) while overpredicting
  peak current (`30.49%`) and carrying waveform NRMSE `0.3613`.
- The inverse-calibration CLI now supports repeatable per-parameter scale
  lists, e.g. `--parameter-scale inductance=0.875,1 --parameter-scale
  resistance=1.5`, so experimental grids can stay targeted instead of applying
  one shared scale list to every parameter.
- Combined GV PF-24 full-horizon L/R ranked artifact:
  `results/experimental_inverse_calibration_gv_pf24_inductance_resistance_waveform_6us_ranked_2026_05_17.json`
  compares tested static inductance/resistance combinations and reports
  `underdetermined_or_correlated_on_candidate_grid`. The best tested region is
  `L=0.875x`, `R=1.5x` with score `0.1240`, peak-current relative error
  `0.0383`, peak-time relative error `0.00285`, and waveform NRMSE `0.2114`.
  `L=1.0x`, `R=1.5x` remains near-best, so the interval is not closed.
- Pressure-screen artifact:
  `results/experimental_inverse_calibration_gv_pf24_l_r_pressure_waveform_1p6us_2026_05_17.json`
  varies pressure `0.75x`, `1.0x`, and `1.25x` at fixed `L=0.875x`,
  `R=1.5x`. This artifact has now advanced from a blocked pressure screen to a
  finite, scored `1.600044e-6 s` experimental run for all three pressure
  candidates. The source-backed path uses raw conductivity with
  `ohm_time_centering_theta=1.0` and
  `backward_euler_resistive_ampere_ohm`; the source-ordered
  predictor-corrector now uses the same time-centered Ohm solve instead of the
  previous zero-dt explicit correction.

  All three candidates satisfy the requested duration and pass finite-state
  checks. The best current score is essentially pressure-insensitive over this
  short horizon: score `0.0441533467`, simulated peak current
  `386.373 kA` versus target `401.600 kA`, peak-time relative error
  `0.01804`, waveform NRMSE `0.06392`, and 59 overlap points. The packet
  remains `horizon_limited_requires_longer_run` because the verified GV target
  waveform spans `-0.5 us` to `6.0 us`; this run covers only about `26.67%` of
  that waveform.

  Pressure is now visible in plasma-loading telemetry instead of being masked
  by an Ohmic-CFL clamp. Terminal source-backed conductivity changes from
  `9668.55 S/m` to `9976.70 S/m`, terminal electron temperature changes from
  `24070 K` to `22710 K`, and terminal electron closure validity remains
  `candidate_nonrelativistic_electron_closure_in_range`. Terminal heat flux is
  `candidate_braginskii_anisotropic_heat_flux_applied` in every pressure case.
- Unclamped conductivity trial:
  `results/experimental_inverse_calibration_gv_pf24_l_r_pressure_unclamped_failure_2026_05_17.json`
  records the attempted removal of the explicit Ohmic CFL cap for the
  source-backed generalized-Ohm path. That trial overflowed current, Ohmic
  heating, and Maxwell field energies, then failed the conservation manifest
  with `ValueError: value must be finite`. Conclusion: simply removing the
  numerical guard was not a first-principles solution. The current code path
  supersedes this failure by using backward-Euler Ohm, source-backed
  Braginskii heat-flux handling, electron-current pressure work, and a
  nonrelativistic electron-closure validity gate.
- Current-source-backed full-window PF-24 artifact:
  `results/experimental_inverse_calibration_gv_pf24_l_r_pressure_waveform_6us_single_2026_05_17.json`
  runs the same calibrated `L=0.875x`, `R=1.5x`, `pressure=1.0x` candidate
  through the complete positive-time GV waveform window. It completes
  `105586` steps to `6.000009497541612e-6 s`, satisfies the requested duration,
  keeps a finite state, and scores the full nonnegative target waveform
  coverage (`0.0 us` to `6.0 us`) with no blocked candidates.

  This closes the immediate runtime question for that experimental candidate:
  the current source-backed 3-D path can execute a full PF-24 current-waveform
  horizon without the earlier heat-flux, Ohmic-CFL, or electron-drift aborts.
  It does not close predictive physics. The simulated peak is now
  `328.654 kA` at `2.386741 us` versus the user-verified target peak
  `401.600 kA` at `1.36 us`; waveform NRMSE is `0.22897`, peak-current
  relative error is `0.18164`, and peak-time relative error is `0.75496`.
  The experimental artifact remains non-promoting, and the next inference loop
  must re-rank source-bounded circuit/startup/power-port candidates against the
  full waveform under the latest physics path.
- Packeted PF-24 `6.0 us` rerun:
  `results/experimental_inverse_calibration_gv_pf24_l_r_pressure_waveform_6us_packeted_2026_05_17.json`
  reproduces the same `L=0.875x`, `R=1.5x`, `pressure=1.0x` full-window
  metrics after the runtime packet wiring change. The run again completes
  `105586` steps to `6.000009497541612e-6 s`, score `0.467396`, and full
  target coverage. Its inverse-calibration candidate now carries the actual
  startup, power-port, limiter-zero, numerical-fidelity, experimental-whole-shot,
  and experimental-numerics packets.

  The power-port residual budget now uses a simulator cumulative all-completed
  step volume `J.E` ledger instead of only retained sparse history:
  cumulative `J.E` work is `-0.118977481171472 J` over `105586` completed
  steps, `integrated_volume_j_dot_e_work_source` is
  `simulator_cumulative_all_completed_steps`, and
  `full_completed_step_j_dot_e_integral_available` is `true`. This reduces the
  bookkeeping blocker but does not accept the power port; sign convention,
  time centering, electrode work, interface domain, and residual tolerance
  remain unreviewed. The same artifact reports startup payload
  `startup_payload_not_supplied` and numerical fidelity
  `blocked_numerical_fidelity_packet_not_available`.

  Limiter accounting has been tightened for the source-backed implicit-Ohm
  path. The explicit Ohmic CFL cap is not applied in this run, so it is no
  longer counted as an acceptance-blocking limiter. The updated packet reports
  `total_acceptance_blocking_activations=0` and
  `zero_acceptance_blockers_observed=true`; it separately records
  `conductivity_ohmic_cfl_raw_exceeds_explicit_limit_steps=105586` as a method
  review item with status
  `candidate_raw_explicit_ohmic_cfl_exceedance_observed_not_applied`. Remaining
  limiter-review work is therefore
  `review_unapplied_raw_ohmic_cfl_exceedance`,
  source-backed physical bounds or method proofs, backend/precision parity,
  and independent engineering review.

  The numerical-fidelity packet consumes this runtime observation without
  promoting it. Its `limiter_zero_acceptance` test surface is
  `candidate_runtime_limiter_zero_observed_not_acceptance` with artifact
  `runtime_experimental_limiter_zero_probe`, and candidate runtime channels now
  include `candidate_limiter_zero_no_applied_blockers_observed` and
  `candidate_full_completed_step_j_dot_e_integral`. The overall numerical gate
  remains `blocked_numerical_fidelity_packet_not_available` because
  mesh/timestep convergence, restart reproducibility, backend/precision parity,
  source-backed tolerances, method proofs, and independent review are still
  missing.

  Full-horizon checkpoint restart has now been exercised for PF-24. The
  source-deck artifact
  `results/experimental_checkpoint_restart_gv_pf24_6us_probe_2026_05_17.json`
  completes `105586` steps with a midpoint checkpoint at `52793` steps;
  uninterrupted and checkpoint-loaded restart paths have matching terminal
  fingerprints and exact tracked observables. The calibrated-candidate artifact
  `results/experimental_checkpoint_restart_gv_pf24_l0875_r15_p1_6us_probe_2026_05_17.json`
  repeats the same full `6.000009497541612e-6 s` comparison for the
  `L=0.875x`, `R=1.5x`, `pressure=1.0x` candidate. It also matches exactly:
  terminal fingerprint
  `dd81f04ec3d4a757370990f3a54f24f8d60928369e30858a958c1f979ce77c0f`,
  final current `110263.02172600424 A`, loaded lagged field work,
  previous-current state, and kinetic-yield state all preserved. This closes
  full-horizon experimental restart plumbing for this candidate. It does not
  close numerical acceptance because restart tolerances, multi-offset families,
  backend/precision parity, source-backed tolerance packets, and independent
  review remain missing.
- Low-inductance inference screens:
  `results/experimental_inverse_calibration_gv_pf24_l_r_waveform_2p6us_lowL_screen_2026_05_17.json`
  tested `L=0.30x` and `L=0.40x` at `R=1.5x` through `2.6 us`. Both
  candidates completed `45754` steps and moved the retained-history peak to
  `1.704831 us`, closer to the target `1.36 us` than the `L=0.875x`
  full-window candidate. The best early-window score was `0.22268` for
  `L=0.30x`, but the run covered only `43.33%` of the verified waveform.

  `results/experimental_inverse_calibration_gv_pf24_l_r_waveform_2p2us_lowL_tight_screen_2026_05_17.json`
  tested the LC-scaled `L=0.18x` and `L=0.22x` region through `2.2 us`.
  Both candidates completed `38715` steps, but the peak moved later
  (`1.841213 us`) and current stayed low (`313.6-317.3 kA`). This rejects a
  simple monotone LC-only tuning explanation.

  `results/experimental_inverse_calibration_gv_pf24_l_r_waveform_6us_lowL_single_2026_05_17.json`
  then ran the early-window best `L=0.30x`, `R=1.5x` candidate through the full
  `6 us` window. It completed `105586` steps to `6.000009497541612e-6 s` with
  finite state and full waveform coverage, but the full-window peak moved to
  `4.205166 us` and dropped to `168.645 kA`; score worsened to `1.26562` and
  waveform NRMSE was `0.30382`. The artifact now includes the attached
  runtime packets showing the actual blockers: startup payload
  `startup_payload_not_supplied`, power port
  `candidate_engineering_power_port_not_validation`, limiter-zero
  `experimental_limiter_zero_probe_not_validation`, and numerical fidelity
  `blocked_numerical_fidelity_packet_not_available`.
- Current conclusion: the code can now infer/test/rank source-bounded candidate
  parameters against a positive-time GV waveform and can run the PF-24
  L/R/pressure screen through the requested `1.6 us` experimental horizon and
  one full `6 us` PF-24 candidate horizon. Static L/R calibration must be
  re-ranked after the latest source-backed electron-energy and backward-Euler
  Ohm changes, because the latest full-window candidates are finite but late
  and low, and low-inductance early-window tuning does not hold over the full
  waveform. Pressure is measurable in plasma telemetry, and the early
  electron-energy/current runaway is closed for this screen. The next
  first-principles blockers are source-grounded startup payload ingestion,
  accepted time-centered power-port coupling, limiter-zero/numerical-fidelity
  evidence, and then full-window candidate re-ranking under those packets.
- PF-1000/Akel seeded-domain limiter proof:
  `results/experimental_limiter_proof_pf1000_seeded_power_domain_6us_2026_05_17.json`
  runs the package-native PF-1000/Akel shot-12581 candidate through
  `6.000091449223011e-6 s` using the vacuum-CFL timestep policy. The run
  completes `27790` steps, satisfies the requested duration, keeps a finite
  state, and reports `zero_acceptance_blockers_observed=true` with
  `total_acceptance_blocking_activations=0`. Final terminal current is
  `1.682957 MA`, final field energy is `115359.442 J`, and the electron
  closure remains nonrelativistic (`current_drift_to_c=7.1369e-6`,
  electron-temperature range `287.831 K` to `42018.004 K`).

  This run supersedes the earlier PF-1000 microsecond aborts. The fixes are
  source-domain guards, not accepted validation evidence: the PF-1000 startup
  seed now carries zero arbitrary volume electric field, the resolved
  `J.E`/electron-fluid/current domains exclude numerical vacuum-floor cells,
  and lagged `J.E` feedback fails closed when it would require an unsupported
  negative active power-port voltage. The artifact records that active-port
  fallback explicitly: `candidate_lagged_volume_j_dot_e=1076`,
  `input_sequence_fallback_first_step=1`, and
  `input_sequence_fallback_negative_j_dot_e_active_port_blocked=26713`.
  Therefore the code can now run an experimental 6 us PF-1000 shot horizon, but
  accepted first-principles power-port authority remains blocked on sign,
  time-centering, electrode-work partition, interface/domain, residual
  tolerance, convergence, and independent review.

## PlasmaPy Use

PlasmaPy is integrated only as an optional community-formulary audit packet via
the `audit` extra. Current PlasmaPy stable/PyPI release information checked on
2026-05-16 points to PlasmaPy 2026.2.0, with current install documentation
requiring Python 3.12-3.14, so the optional extra is marker-gated to Python
3.12+. It can cross-check quantities such as Debye length,
Coulomb logarithm, Alfven speed, and electron gyrofrequency when installed.
It is not a source of authority, and it cannot promote a first-principles
acceptance gate. Local `KnowledgeReference/` material remains the source of
truth for physics claims.

## Engineering Review Focus

The simulator now has a bounded-history experimental run path suitable for
engineering inspection, but the artifact still blocks first-principles
acceptance on the following packet families:

- Accepted startup BVP, surface flashover, preionization, secondary emission,
  and sheath liftoff. The new CIV/Paschen audit is telemetry only.
- Reviewed PF-1000 conductor/material mask. The new PF-1000 rod/hollow
  projection is source-backed telemetry only; the hollow bore and insulator
  material surfaces are still not accepted field-boundary regions.
- Limiter-readiness and zero hidden fallback/floor proof
  (`conductivity_ohmic_cfl_limited_steps` is active in the nanosecond PF-1000
  limiter probe).
- Numerical-fidelity and convergence evidence
- Same-scope PF-1000/Akel source targets and current waveform UQ
- Spatial field, density, and temperature diagnostics
- Mechanism-separated neutron authority
- Comparator/UQ matrix
- Certificate gate and second-scope generalization
