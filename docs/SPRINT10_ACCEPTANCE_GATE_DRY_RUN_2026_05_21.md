# Super-Sprint 10 SS10-7 — Report-Only Acceptance-Gate Dry Run

**Date:** 2026-05-21
**Workstream:** SS10-7 (Add Report-Only Acceptance-Gate Dry Run)
**Scope:** Report-only. This document and the command it describes promote no
acceptance flag. The dry run runs the eight acceptance gates against the
six-step PF-1000 full-energy engineering probe and reports a fail-closed
ledger.

## Purpose

Make the next physics sprint measurable. SS10-7 adds a report-only command that
runs the eight first-principles acceptance gates against the
`pf1000_scholz_2001_24rod_full_energy` engineering probe and emits a
**fail-closed ledger**: each gate is `pass` or `blocked`, and every blocked
gate names the exact missing source packet, runtime field, or numerical check
plus a concrete next action.

The dry run **reads** the eight gate packets already emitted by the
package-native 3-D runner — it does not re-evaluate any gate. The gate modules
(`numerical_fidelity.py`, `same_scope.py`, `comparator_uq.py`,
`certificate_gate.py`, the conductor-mask geometry packet, `startup_bvp.py`,
`power_port.py`, `neutron_authority.py`) remain the sole acceptance authority.

## Module and entry point

| Item | Path |
|---|---|
| Dry-run module | `src/dpf/first_principles/acceptance_gate_dry_run.py` |
| CLI subcommand | `dpf first-principles-acceptance-dry-run` (`src/dpf/cli/main.py`) |
| Test file | `tests/test_first_principles_acceptance_gate_dry_run.py` |

### Reproduce

```bash
cd /Users/anthonyzamora/dpf-unified
PYTHONPATH=src .venv312/bin/python -m dpf first-principles-acceptance-dry-run
```

The command runs the PF-1000 full-energy probe, prints the ledger, and writes
the JSON ledger to a **temp path** by default
(`$TMPDIR/dpf_first_principles_acceptance_gate_dry_run.json`). A transient
ledger is therefore never written into `results/` and never dirties
`git_status_clean` or the first-principles artifact linter.

To capture the JSON ledger at a caller-chosen path:

```bash
PYTHONPATH=src .venv312/bin/python -m dpf first-principles-acceptance-dry-run \
  --output /path/of/your/choosing/acceptance_gate_dry_run.json
```

The committed deliverable is **this markdown**, not a `results/` JSON: per the
SS10-7 instruction, a `results/` JSON is allowed only if it cleanly passes the
artifact linter (`artifact_generation_commit == HEAD`), so the durable record
is this document plus a temp-path JSON for reproducibility.

## Fail-closed ledger contract

The ledger (`AcceptanceGateDryRunLedger`) hard-codes every acceptance-bearing
flag and has no code path that can flip one:

| Flag | Value |
|---|---|
| `report_only` | `true` |
| `promotes_acceptance` | `false` |
| `accepted_runtime_claim` | `false` |
| `can_support_first_principles_acceptance` | `false` |

Each gate result (`GateDryRunResult`) additionally carries
`promotes_acceptance=false` and `can_support_first_principles_acceptance=false`.
The ledger's `is_fail_closed` property is `false` if any gate is neither `pass`
nor `blocked`, if any blocked gate has an empty `missing` list, or if any
acceptance flag is set — so a non-fail-closed ledger cannot pass its own test.

## Dry-run result — eight gates

Runtime: `pf1000_scholz_2001_24rod_full_energy`, six-step engineering probe,
`scientific_status = engineering_candidate_not_validation`.

**All eight gates are `blocked`. No gate passes. Every blocked gate names a
non-empty list of missing inputs.**

### 1. numerical_fidelity — blocked

Packet status: `blocked_numerical_fidelity_packet_not_available` (25 missing
channels).

Missing inputs (named): `test_surface_registry`,
`source_backed_numerical_method_map`,
`analytic_or_manufactured_reference_solutions`, `mesh_family_definitions`,
`timestep_family_definitions`, `norms_by_test_surface`,
`tolerances_by_test_surface`, `observed_order_or_monotonic_convergence`,
`finite_volume_shock_packet`, `cylindrical_source_term_packet`,
`maxwell_yee_courant_packet`, `divergence_b_packet`,
`gauss_law_or_continuity_packet`, `resistive_diffusion_packet`,
`joule_heating_energy_packet`, `circuit_power_port_numerical_packet`,
`particle_push_deposition_packet`, `mesh_timestep_convergence_packet`,
`restart_reproducibility_packet`, `backend_precision_parity_packet`,
`limiter_zero_packet`, `same_scope_numerical_observable_mapping`,
`artifact_links_and_hashes`, `negative_tests_for_failed_tolerance`,
`independent_review_certificate`.

**Next action:** Build the numerical-fidelity acceptance suite — source-backed
reference solutions, norms/tolerances, mesh+timestep convergence, restart
reproducibility, backend/precision parity, and a limiter-zero proof for the
PF-1000 full-energy scope.

### 2. same_scope_comparator — blocked

Packet status: `blocked_same_scope_source_packet_not_available` (17 missing
channels).

Missing inputs (named): `device_geometry_and_electrode_dimensions`,
`bank_circuit_drive`, `gas_species_pressure_temperature`,
`accepted_digitized_current_waveform`, `startup_breakdown_preionization`,
`density_spatial_history`, `em_field_history`, `electron_temperature_history`,
`ion_temperature_or_distribution_history`, `neutron_scalar_yield`,
`neutron_timing_history`, `neutron_spectrum`, `neutron_anisotropy`,
`detector_response_and_calibration`, `uncertainty_budget`,
`source_review_certificate`, `cross_scope_transfer_rule_or_rejection_tests`.

**Next action:** Extract a same-scope PF-1000 full-energy (27–40 kV, 24-rod)
source packet from KnowledgeReference: digitized current waveform, startup,
density/field/temperature histories, neutron timing/spectrum/anisotropy,
detector calibration, an uncertainty budget, and an independent review
certificate. (This is the SS10 "same-scope PF-1000 full-energy 3-D validation
packet design" follow-up.)

### 3. uq — blocked

Packet status: `blocked_comparator_uq_matrix_not_available` (19 missing
channels).

Missing inputs (named): `accepted_same_scope_target_registry`,
`source_hashes_and_review_status`, `output_field_mapping_by_observable`,
`unit_conversion_and_coordinate_mapping`, `time_alignment_policy`,
`comparator_metric_by_observable`, `comparator_tolerance_by_observable`,
`measurement_uncertainty_by_observable`, `model_uncertainty_by_observable`,
`numerical_uncertainty_by_observable`, `closure_sensitivity_uncertainty`,
`detector_response_uncertainty`, `shot_to_shot_uncertainty_or_scope_rule`,
`uq_propagation_method`, `pass_fail_rule_by_observable`,
`negative_control_cases`, `requirement_links`, `artifact_links_and_hashes`,
`independent_review_certificate`.

**Next action:** Construct the comparator/UQ matrix once same-scope targets
exist — per-observable output mapping, metrics, tolerances,
measurement/model/numerical uncertainty, a UQ propagation method, negative
controls, and an independent review certificate.

### 4. certificate — blocked

Packet status: `blocked_first_principles_certificate_not_available` (31 missing
channels). The certificate gate depends on every upstream gate; it names each
unaccepted upstream packet (`same_scope_source_packet_accepted`,
`waveform_phase_packet_accepted`, `spatial_field_temperature_packet_accepted`,
`neutron_authority_packet_accepted`, `comparator_uq_packet_accepted`,
`numerical_fidelity_packet_accepted`, `physics_closure_packet_accepted`,
`limiter_zero_or_physical_bounds_packet`, `power_port_packet_accepted`,
`startup_packet_accepted`, `dimensionality_handoff_packet_accepted`) plus the
full negative-test matrix, run/evidence hashes, reviewer metadata, and the
release decision.

**Next action:** Assemble the first-principles certificate only after every
upstream gate is accepted — run/evidence hashes, reviewer metadata, the full
negative-test matrix, and a release decision.

### 5. geometry — blocked

Packet status: `candidate_engineering_conductor_mask_not_validation` (5 blocked
geometry fields, from `boundary_policy.conductor_mask.blocked_geometry_fields`).

Missing inputs (named): `anode_hollow_bore_length_m`,
`insulator_wall_thickness_m`, `backplate_radial_extent_m`,
`backplate_axial_thickness_m`, `same_scope_reviewed_geometry_mask`.

**Next action:** Request the four absent PF-1000 hollow-bore geometry
dimensions (anode hollow-bore length, insulator wall thickness, backplate
radial extent and axial thickness) from IPPLM and supply a same-scope reviewed
conductor-mask geometry packet. (This is the SS10 "four absent PF-1000 geometry
dimensions and IPPLM facility request" follow-up.)

### 6. startup — blocked

Packet status: `rejected_startup_mode_for_first_principles` (24 missing
channels).

Missing inputs (named): `device_geometry_and_insulator`,
`insulator_wall_geometry`, `backplate_geometry`,
`surface_material_secondary_emission`, `gas_species_pressure_temperature`,
`pressure_regime_classifier`, `bank_voltage_and_early_circuit`,
`preionization_state`, `breakdown_model`, `breakdown_or_flashover_model`,
`surface_flashover_closure`, `sheath_liftoff`,
`sheath_liftoff_and_handoff_interval`, `initial_magnetic_field`,
`initial_electric_field`, `initial_current_density_distribution`,
`initial_density_ionization_charge_state`, `initial_velocity_distribution`,
`electron_temperature_initial`, `ion_temperature_initial`,
`initial_resistivity_or_conductivity`,
`charge_current_divb_energy_consistency`, `same_scope_startup_bvp`,
`source_paths_hashes_units_and_review`.

**Next action:** Author the startup BVP source packet for D2
breakdown/flashover/liftoff handoff with same-scope geometry, insulator, and
early-circuit evidence. Imported-PIC startup stays context-only and cannot
satisfy this gate.

### 7. power_port — blocked

Packet status: `candidate_engineering_power_port_not_validation` (8 missing
channels).

Missing inputs (named): `named_interface_surface_or_volume_domain`,
`poynting_or_j_dot_e_power_integral`, `accepted_sign_convention`,
`accepted_time_centering`, `residual_tolerance`,
`auluck_eq6_six_term_completeness`, `startup_handoff_interval`,
`same_scope_power_port_review`.

**Next action:** Construct the reviewed sigma-p face set for power-port terms
II/IV/V/VI, supply terminal current/voltage and the active-load relation, fix
sign convention and time centering, and close the Auluck Eq.6 six-term energy
ledger with a residual tolerance.

### 8. neutron — blocked

Packet status: `blocked_mechanism_separated_neutron_authority_not_available`
(19 missing channels).

Missing inputs (named): `mechanism_separated_yield_channels`,
`accepted_beam_target_yield_history`, `accepted_thermonuclear_yield_history`,
`same_scope_scalar_yield`, `dd_cross_section_source_and_units`,
`ion_energy_distribution_history`, `target_density_path_length_history`,
`beam_transport_stopping_model`, `beam_angular_distribution_history`,
`neutron_energy_spectrum`, `neutron_timing_history`,
`neutron_anisotropy_angular_yield`, `direct_scattered_neutron_transport`,
`detector_response_model`, `activation_counter_response_model`,
`electron_temperature_yield_sensitivity_uq`, `yield_uncertainty_budget`,
`output_mapping_and_comparator`, `source_review_certificate`.

**Next action:** Provide mechanism-separated beam-target and thermonuclear
yield histories with a DD cross-section source, a deuteron transport/stopping
model, an activation-counter response model, and a same-scope reviewed neutron
source packet.

## Summary table

| Gate | Status | Backing packet | # missing | Blocking theme |
|---|---|---|---|---|
| numerical_fidelity | blocked | `numerical_fidelity` | 25 | No numerical-fidelity acceptance suite (references, norms, convergence, restart, parity, limiter-zero). |
| same_scope_comparator | blocked | `same_scope_source` | 17 | No same-scope PF-1000 full-energy source packet. |
| uq | blocked | `comparator_uq` | 19 | No comparator/UQ matrix (depends on same-scope targets). |
| certificate | blocked | `certificate_gate` | 31 | Every upstream gate unaccepted; no certificate payload. |
| geometry | blocked | `boundary_policy.conductor_mask` | 5 | Four absent PF-1000 hollow-bore dimensions + no reviewed mask. |
| startup | blocked | `startup` | 24 | No startup BVP source packet; imported-PIC startup is context-only. |
| power_port | blocked | `power_port` | 8 | No reviewed sigma-p face set; Auluck Eq.6 ledger not closed. |
| neutron | blocked | `neutron_authority` | 19 | No mechanism-separated yield histories or detector model. |

## Module-vetting note

`src/dpf/first_principles/acceptance_gate_dry_run.py` lives under the
`first_principles/` physics prefix, so the first-principles module
source-vetting script would otherwise classify it `physics_needs_source_vetting`
and place it in the `inactive_physics_unvetted` blocker bucket — that bucket
participates in `strict_passed`, so the module would trip strict.

The module is genuine **non-promoting report-only infrastructure**: it computes
no physics, holds no KnowledgeReference authority, and has no code path that
promotes acceptance. Following the `channel_state.py` precedent, it is
registered in `NONPHYSICS_INFRASTRUCTURE_FILES` in
`scripts/verify_first_principles_module_source_vetting.py`. With that
registration the vetting script reports `strict_passed: true` and
`--strict` exits 0.

The dated module-vetting artifacts
(`docs/FIRST_PRINCIPLES_MODULE_SOURCE_VETTING_2026_05_20.json` / `.md`) are
regenerated by running the vetting script; this workstream left those tracked
artifacts at HEAD. The lead regenerates them when committing.

## Verification

```bash
.venv312/bin/python -m pytest \
  tests/test_first_principles_acceptance_gate_dry_run.py \
  tests/test_first_principles_runner.py \
  tests/test_cli_first_principles_3d.py -q -rx
# 65 passed

.venv312/bin/python -m ruff check \
  src/dpf/first_principles/acceptance_gate_dry_run.py \
  tests/test_first_principles_acceptance_gate_dry_run.py \
  src/dpf/cli/main.py
# All checks passed

.venv312/bin/python scripts/verify_first_principles_module_source_vetting.py \
  --strict --date 2026_05_20
# strict_passed: true
```
