# First-Principles Execution Audit

Date: 2026-05-13

Status: pivot directive for implementation. This document does not promote any
simulation result, digitization packet, source target, or readiness status.

Follow-on baseline: `docs/FIRST_PRINCIPLES_FINISH_LINE_PLAN.md` is now the
active finish-line roadmap. This audit explains the pivot; the finish-line plan
defines the phase gates, requirement IDs, acceptance evidence, and critical path
from engineering probe to accepted first-principles PF-1000/Akel simulation.

## Verdict

The plan has been protecting scientific claims correctly, but it is no longer
optimized for getting first-principles MHD working. The recent work added useful
fail-closed metadata across app, API, CLI, manifests, and monitor surfaces. That
was necessary once, but continuing in that direction is now tail chasing.

The optimized path is to stop expanding readiness/reporting surfaces and make
one PF-1000/Akel `first_principles_mhd` run compute its circuit loading from
resolved fields. Testing should move behind that milestone: use only compile
checks and tiny smoke probes while wiring the physics, then restore broader
regression coverage after the run produces coherent field-coupled histories.

## Current Misoptimization

The current plan is too broad at the top level:

- It mixes product/SRS controls, validation dashboards, test matrices, and
  first-principles physics implementation in one active lane.
- It treats missing validation evidence and missing working physics as peers.
  They are not peers right now. Working field-coupled physics is the blocker.
- It has over-rotated on fail-closed labels. The labels are now good enough;
  more labels will not create a first-principles simulator.
- It still lets snowplow/Lee own too much of the production path. Even when
  snowplow is labeled baseline-only, the circuit path still uses snowplow load
  or density-weighted inductance scaffolds rather than a resolved field power
  balance.
- It keeps validation gates in the critical path before the physics exists.
  Same-scope Akel validation matters, but it should not block constructing a
  working field-coupled implementation candidate.

## First-Principles Definition For The Next Sprint

For the next sprint, "first principles" means the circuit update is driven by
quantities computed from resolved MHD fields and local conservation laws, not by
Lee/RADPF closure factors.

Required candidate outputs:

- `I(t)` and `V(t)` from one circuit step per MHD step.
- Field-derived magnetic energy and plasma inductance:
  `L_field = 2 * E_B / max(I^2, eps)`.
- Field-derived interface power, preferably Poynting or `J.E` based, with the
  exact sign convention recorded.
- Back-EMF or equivalent terminal voltage from the field-power relation:
  `V_field = P_field / max(I, eps)` with a documented sign convention.
- `dL_field/dt` as a diagnostic, not the primary authority.
- Energy ledger:
  capacitor energy, external inductive energy, magnetic field energy,
  resistive/Joule heating, radiation losses if active, and residual.
- Interval authority labels:
  `snowplow_loaded` for initialization only, `field_coupled_candidate` once the
  field-power circuit update owns the step.

Rejected for this sprint:

- More dashboard categories.
- More API/UI/CLI propagation unless needed to run the candidate.
- More readiness tests.
- More source packet promotion.
- More reduced-model calibration.
- Late-pinch/neutron/kinetic validation.

## Optimized Execution Path

### 1. Freeze Reporting Surfaces

Do not add new readiness schemas, response fields, badges, manifests, or
dashboard categories unless the field-coupled implementation cannot be run
without them. The existing fail-closed mode is sufficient for now.

Exit condition: new code changes are in solver/circuit/field-coupling modules,
not primarily in app/API/test/reporting files.

### 2. Build A Minimal Field-Coupled Candidate Path

Target one path first: the current Python/MLX MHD execution path used by
`first_principles_mhd`.

Implementation objective:

- Compute magnetic field energy from the MHD state on the cylindrical grid.
- Compute `L_field` from magnetic energy and circuit current.
- Compute Poynting or `J.E` interface power from fields already available in
  `mlx_coupling.py`/MHD state.
- Feed a nonzero field-derived voltage/back-EMF into the circuit step.
- Record snowplow only as initialization/comparison, not as circuit authority
  after the handoff.

Exit condition: a short PF-1000/Akel candidate run produces finite histories for
current, voltage, `L_field`, `P_field`, `back_emf_V`, magnetic energy, Joule
heating, and energy residual.

### 3. Make Startup Just Good Enough To Run

Do not solve full breakdown physics before the field-coupled circuit path works.
Use the existing seeded plasma layer as an explicit engineering initialization
only, but make sure it produces a resolved initial conductive sheath/field state
that the MHD solver can evolve without snowplow authority.

Exit condition: axial rundown can begin from resolved fields without using Lee
closure factors for the circuit update after startup.

### 4. Reduce The Physics Stack To The Minimum Working Set

For the first candidate, keep the physics scope narrow:

- ideal/resistive MHD,
- Joule heating,
- simple single-fluid or existing two-temperature hooks if already stable,
- existing transport/radiation modules only if they do not block the run.

Defer tabulated EOS, detailed ionization, radiation, ablation, kinetic neutron
production, and 3D instability scope until after current/voltage field coupling
works.

Exit condition: the run is finite and the energy ledger is physically
interpretable, even if accuracy is not yet accepted.

### 5. Only Then Reopen Validation

After the candidate produces coherent field-coupled histories:

- compare the waveform against the Akel draft packet only as an engineering
  visual/probe, still `blocked_by_review`;
- identify the dominant mismatch;
- then decide whether startup, transport, EOS/ionization, or circuit coupling
  is the next physics blocker.

Exit condition: validation work is driven by a running first-principles
candidate, not by preemptive gates.

## Immediate Next Code Target

Start in these files:

- `src/dpf/metal/mlx_coupling.py`
- `src/dpf/metal/mlx_engine.py`
- `src/dpf/validation/circuit_field_coupling.py`
- whichever Python MHD path backs `app_mhd.py` for non-MLX fallback

Do this first:

1. Extract a reusable field-energy and field-power diagnostic from the MHD state.
2. Wire that diagnostic into `run_mlx_discharge()` or the selected candidate
   path.
3. Replace `back_emf = 0.0` in the first-principles path with the field-derived
   terminal voltage/back-EMF, gated for finite values and sign sanity.
4. Keep snowplow `Lp` in the result as `baseline_reduced_model`, but stop using
   it as the circuit load after the handoff in the candidate path.
5. Run one short candidate probe. Do not run the broad test suite yet.

## Success Criteria Before More Testing

The next milestone is not "all tests pass." The next milestone is:

- the candidate run completes a short PF-1000/Akel interval;
- no nonfinite state is produced;
- `back_emf_V` is nonzero during field-coupled intervals;
- field magnetic energy and circuit energy move in a plausible direction;
- residual energy is bounded and inspectable;
- snowplow closure factors are not used for predictive scoring.

Only after this milestone should broader tests and acceptance gates resume.

## Execution Update 2026-05-13

The first implementation pass has moved the candidate path from metadata-only
readiness into resolved-field circuit coupling.

Implemented:

- annular radial coordinates for the PF-1000/Akel Python cylindrical MHD path;
- resolved-field magnetic energy and `L_field = 2 E_B/I^2`;
- `integral(J dot E)dV`, field terminal voltage/back-EMF, and interface power;
- first-principles candidate circuit stepping with field-derived `L_field` and
  field-terminal-voltage feedback;
- field-coupled interval labels and energy-residual histories.

Engineering probe evidence:

- 0.2 us coarse PF-1000/Akel `first_principles_mhd`: 201 steps,
  `nan_detected=False`, nonzero `B_max`, `L_field`, `J.E` power, and
  `back_emf_V`.
- 1.0 us coarse PF-1000/Akel `first_principles_mhd`: 201 steps,
  `nan_detected=False`, `I_peak=0.5289253407371984 MA`, `B_max` up to
  `0.9049918942042798 T`, field-derived inductance up to
  `4.163413325993563 nH`, and nonzero `back_emf_V`.

Follow-on ratchet:

- explicit capped Spitzer/Braginskii resistivity and nonzero Joule heating are
  now wired into the candidate;
- `L_field` is exported as a diagnostic while the circuit load is driven by
  resolved field-energy change plus Joule power;
- a recorded engineering limiter bounds density, pressure, velocity, magnetic
  field magnitude, and total magnetic field energy;
- a coarse 12 us PF-1000/Akel `first_principles_mhd` shot now completes 24,000
  steps with `nan_detected=False`, nonzero field-coupled back-EMF, nonzero Joule
  energy, and bounded magnetic energy.

Immediate next physics target:

- replace or reduce the explicit engineering limiter with verified
  finite-volume stability controls and source-backed startup/transport physics.

This remains an engineering probe. It does not promote any same-scope Akel
validation evidence or first-principles readiness status.

## Neutron-Yield Authority Update 2026-05-13

The user-level target is stricter than a field-coupled current waveform: a
first-principles simulator should eventually reproduce paper neutron yields to
10% or better. That target is now encoded as a fail-closed acceptance criterion,
not as a present claim.

Implemented:

- `first_principles_neutron_yield_authority_status()` classifies neutron
  outputs before first-principles acceptance.
- App-level neutron totals report that final-state thermonuclear duration
  estimates and Lee/Saw beam-target estimates are non-promoting.
- Server readiness exports the same neutron-yield authority packet.
- Yield-tracker summaries now preserve the boundary between mechanism-separated
  reporting and first-principles predictive authority.

Acceptance remains blocked until all of these are true for the same local
`KnowledgeReference/` scope:

- thermonuclear yield comes from resolved field-history integration;
- beam-target yield comes from accepted kinetic/hybrid beam production, not
  Lee/Saw calibration or empirical beam fractions;
- scalar yield, mechanism timing, spectrum, anisotropy, detector/activation
  response, and uncertainty all pass together;
- numerical-fidelity and physics-fidelity evidence are accepted for the run.

Verification:

- `python3 -m py_compile app_mhd.py src/dpf/validation/first_principles_mhd.py src/dpf/validation/__init__.py src/dpf/server/readiness.py src/dpf/server/models.py src/dpf/diagnostics/yield_tracker.py tests/test_first_principles_mhd.py tests/test_mhd_physics_integration.py`
- `python3 -m pytest tests/test_first_principles_mhd.py tests/test_mhd_physics_integration.py::test_neutron_mechanism_output_summary_keeps_estimates_non_promoting tests/test_mhd_physics_integration.py::test_post_processing_preserves_field_history_thermonuclear_yield tests/test_neutron_yield.py tests/test_yield_tracker.py tests/test_server_readiness.py -q -o addopts=` -> `103 passed`

## Thermonuclear History Update 2026-05-13

The candidate now removes the final-state duration shortcut for the
thermonuclear component:

- each Python MHD step computes DD thermonuclear production from the resolved
  density and ion-temperature fields;
- the volume integral uses cylindrical annular cell volumes;
- `yield_time_resolved.source_authority` is
  `resolved_field_history_candidate`;
- post-processing preserves the field-history result instead of replacing it
  with a final-state estimate.

This improves the thermonuclear component only. Total neutron-yield authority
remains blocked until beam-target production is kinetic/hybrid and same-scope
neutron evidence passes.

Verification:

- `python3 -m py_compile app_mhd.py tests/test_mhd_physics_integration.py`
- covered by the combined focused neutron/readiness run above -> `103 passed`

## Tool Entrypoint Update 2026-05-13

The runnable first-principles tool is now a narrow PF-1000/Akel engineering
candidate entrypoint, not a new default backend.

Implemented:

- `run_pf1000_akel_first_principles()` in `app_mhd.py`, which locks the app
  path to `backend="first_principles_mhd"` and `preset_name="pf1000_akel"`;
- `dpf first-principles`, which calls the locked helper, rejects reduced-model
  fallback by requiring `field_coupled_candidate=True` and `has_snowplow=False`,
  and writes a compact JSON engineering-probe artifact when requested;
- usage docs in `README.md` and `docs/USAGE.md`.

Smoke evidence:

- `dpf first-principles --sim-time-us=0.2 --history-stride=20 --output results/first_principles_pf1000_akel_smoke.json`
- Result: 400 finite steps, `nan_detected=False`,
  `I_peak_MA=0.1128993`, `back_emf_abs_max_V=2728.926`,
  `L_field_max_nH=2.939681`, final `joule_energy_kJ=0.01092775`,
  and `readiness=blocked`.

Verification:

- `python3 -m py_compile app_mhd.py src/dpf/cli/main.py tests/test_cli_backend_options.py tests/test_mhd_physics_integration.py`
- `python3 -m pytest tests/test_cli_backend_options.py::test_first_principles_command_runs_field_coupled_candidate tests/test_cli_backend_options.py::test_first_principles_command_fails_on_reduced_fallback tests/test_cli_backend_options.py::test_first_principles_runner_loader_adds_checkout_root tests/test_mhd_physics_integration.py::test_pf1000_akel_first_principles_helper_locks_scope tests/test_mhd_physics_integration.py::test_first_principles_mhd_mode_exports_fail_closed_app_readiness tests/test_first_principles_mhd.py tests/test_readme_claims.py -q -o addopts=` -> `17 passed`

This remains an engineering probe. It does not promote same-scope Akel
validation evidence, accept the draft waveform packet, or remove the
first-principles readiness/neutron-yield authority blockers.
