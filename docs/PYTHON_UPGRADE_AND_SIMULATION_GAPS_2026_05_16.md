# Python Upgrade And First-Principles Simulation Gap Audit

Date: 2026-05-16

This note records the interpreter-upgrade decision and the remaining
not-yet-simulated DPF physics scope for the package-native first-principles
3-D path. It is an engineering planning artifact, not a validation packet.

## Current Local Interpreter

- Initial local interpreter before upgrade: Python 3.11.9.
- Installed interpreter for the first-principles path:
  `/opt/homebrew/bin/python3.12`, Python 3.12.13.
- Local 3.12 environment: `.venv312`.
- Project metadata now declares `requires-python = ">=3.12"`.
- CI and publish workflows now pin Python 3.12 for the active repo path.
- The optional PlasmaPy audit extra is marker-gated to Python 3.12+ and was
  installed locally as PlasmaPy 2026.2.0.

## Upstream Compatibility Snapshot

Sources checked on 2026-05-16:

- CPython devguide status page:
  - Python 3.11: security phase, EOL 2027-10.
  - Python 3.12: security phase, EOL 2028-10.
  - Python 3.13: bugfix phase, EOL 2029-10.
  - Python 3.14: bugfix phase, EOL 2030-10.
  - Source: `https://devguide.python.org/versions/`
- PlasmaPy install docs:
  - Current PlasmaPy 2026.2.0 requires Python 3.12-3.14.
  - Source: `https://docs.plasmapy.org/en/stable/install.html`
- Numba current support table:
  - Numba 0.65.x supports Python `>=3.10,<3.15`.
  - Numba 0.62.x supports Python `>=3.10,<3.14`.
  - Source: `https://numba.readthedocs.io/_/downloads/en/latest/pdf/`
- NumPy release notes:
  - NumPy 2.3.x supports Python 3.11-3.13, with later notes indicating
    Python 3.14 support in newer patch releases.
  - Source: `https://numpy.org/doc/stable/release/2.3.0-notes.html`
- SciPy release information:
  - SciPy 1.14.1 added Python 3.13 wheels.
  - Newer SciPy release metadata includes Python 3.11-3.14 support.
  - Sources:
    - `https://docs.scipy.org/doc/scipy-1.16.1/release/1.14.1-notes.html`
    - `https://github.com/scipy/scipy/releases`
- JAX support policy:
  - JAX support policy lists Python 3.13 support through at least July 2028
    and Python 3.14 through at least July 2029.
  - Source: `https://docs.jax.dev/en/latest/deprecation.html`

## Decision

Upgrade feasibility: complete for the controlled Python 3.12 environment.

Implemented path:

1. Added a local Python 3.12.13 virtual environment for the engineering
   first-principles path and PlasmaPy audit.
2. Moved project metadata, Ruff, mypy, CI, and publish workflows to Python
   3.12 for now.
3. Add Python 3.13 to CI only after the package install and focused
   first-principles tests pass locally.
4. Do not move the default development/runtime target to Python 3.14 yet.
   Numba support is now plausible in newer versions, and PlasmaPy allows 3.14,
   but this repo has not been tested against 3.14 and several optional
   accelerator/AI extras can lag the CPython release cycle.

Worth it:

- Yes for Python 3.12+: it unlocks the current PlasmaPy audit extra and keeps
  us on a supported scientific-Python lane.
- Probably yes for Python 3.13 after a local environment pass: CPython 3.13 is
  in bugfix support, and the core scientific stack now has a viable support
  path.
- Not yet for Python 3.14 as the default: the benefit does not outweigh the
  compatibility risk until the full optional dependency set is tested.

## What The Current Experimental Simulator Does Simulate

The current `first-principles-3d` path runs a package-native 3-D engineering
candidate with:

- Cartesian 3-D Yee-field state.
- Candidate conductor/PML/open-boundary field policy.
- Candidate absorbing particle boundaries.
- Candidate PF-1000 12-rod cathode / hollow-anode geometry projection onto
  the Cartesian conductor mask. The projection uses source-backed rod count,
  rod diameter, anode radius, cathode radius, anode length, and alumina
  insulator material, but remains non-promoting.
- Density-normalized deterministic deuteron macro-particle initialization.
- Hybrid ion PIC push/deposition.
- Candidate electron-energy source update.
- Candidate deuterium ionization/recombination charge-state transport.
- Candidate source-backed partial-ionized conductivity path.
- Candidate external circuit magnetic boundary coupling.
- Candidate lagged full-volume `J.E` load feedback.
- Candidate CIV/Paschen startup breakdown and liftoff audit telemetry for
  engineering inspection only.
- Candidate kinetic ion neutron-yield history.
- Conservation and finite-state runtime telemetry.
- Optional PlasmaPy community-formulary audit telemetry when installed.

## What We Are Not Simulating Yet

Whole-shot startup:

- Accepted neutral gas breakdown from the charged bank. The runner now emits a
  candidate CIV/Paschen breakdown audit, but this is not an accepted DPF
  surface-breakdown BVP.
- Accepted surface flashover/streamer/avalanche physics on the insulator.
- Accepted preionization source physics and measured/prepared initial
  ionization state.
- Accepted electrode/insulator material secondary-emission model.
- Self-consistent sheath liftoff from breakdown through rundown. The current
  liftoff delay is a candidate audit estimate only.
- Source-reviewed imported PIC sheath state for the start of the 3-D run.

## Python 3.12 Verification

Commands run after the upgrade:

- `.venv312/bin/python --version`: Python 3.12.13.
- `.venv312/bin/python -c "import sys, plasmapy; ..."`:
  Python 3.12.13 and PlasmaPy 2026.2.0 imported.
- `.venv312/bin/python -m pytest tests/test_plasmapy_audit.py -q -o addopts=`:
  `2 passed`.
- `.venv312/bin/python -m pytest tests/test_startup_breakdown_audit.py
  tests/test_plasmapy_audit.py tests/test_first_principles_input_deck.py
  tests/test_first_principles_source_targets.py -q -o addopts=`:
  `30 passed`.
- `.venv312/bin/python -m pytest tests/test_startup_breakdown_audit.py
  tests/test_plasmapy_audit.py tests/test_first_principles_input_deck.py
  tests/test_first_principles_source_targets.py tests/test_cli_first_principles_3d.py
  tests/test_first_principles_runner.py::test_first_principles_runner_projects_candidate_conductor_mask_from_package_deck
  tests/test_first_principles_runner.py::test_first_principles_3d_runner_carries_startup_policy_from_package_deck
  tests/test_first_principles_runner.py::test_first_principles_runner_marks_pf1000_akel_same_scope_as_blocked
  tests/test_first_principles_runner.py::test_first_principles_runner_propagates_long_run_history_controls
  -q -o addopts=`:
  `40 passed`.
- `.venv312/bin/python -m ruff check
  src/dpf/first_principles/startup_breakdown.py
  src/dpf/first_principles/startup_bvp.py src/dpf/first_principles/runner.py
  src/dpf/first_principles/__init__.py tests/test_startup_breakdown_audit.py`:
  passed.
- `.venv312/bin/python -m pytest tests/test_circuit_field_coupling.py
  tests/test_first_principles_runner.py::test_run_first_principles_3d_deck_returns_candidate_manifest_and_telemetry
  -q -o addopts=`:
  `17 passed` after fixing the NumPy 2.x `np.trapz` compatibility path.

## 3.12 Startup Probe

Command:

```bash
.venv312/bin/python -m dpf.cli.main first-principles-3d \
  --deck-preset pf1000_akel_16kv \
  --steps 20 \
  --history-stride 5 \
  --max-step-results 8 \
  --output results/first_principles_3d_py312_startup_probe_2026_05_16.json
```

Result summary:

- Steps requested/completed: `20 / 20`.
- Termination reason: `completed_step_budget`.
- Startup packet status: `blocked_startup_bvp_packet_not_available`.
- Candidate breakdown audit:
  `candidate_civ_paschen_breakdown_audit_engineering_only`.
- Candidate mechanism for this PF-1000/Akel probe: `Paschen`.
- Candidate breakdown time: `2.3524660435688997e-09 s`.
- Candidate liftoff delay: `2.523524660435689e-07 s`.
- Startup acceptance: `false`.

## 3.12 PF-1000 Geometry Probe

Command:

```bash
.venv312/bin/python -m dpf.cli.main first-principles-3d \
  --deck-preset pf1000_akel_16kv \
  --steps 20 \
  --history-stride 5 \
  --max-step-results 8 \
  --output results/first_principles_3d_py312_pf1000_geometry_probe_2026_05_16.json
```

Result summary:

- Steps requested/completed: `20 / 20`.
- Termination reason: `completed_step_budget`.
- Conductor mask mode: `pf1000_rod_hollow_projection`.
- Mask source: `candidate_pf1000_rod_hollow_projection`.
- Active conductor cells: `27`.
- Source-backed cathode rod count in telemetry: `12`.
- Cathode rods projected: `true`.
- Hollow-anode inner radius supplied: `false`.
- Insulator material in telemetry: `alumina`.
- Startup packet remains blocked:
  `blocked_startup_bvp_packet_not_available`.

## 3.12 Microsecond Duration Request Probe

Command:

```bash
.venv312/bin/python -m dpf.cli.main first-principles-3d \
  --deck-preset pf1000_akel_16kv \
  --steps 20 \
  --history-stride 5 \
  --max-step-results 8 \
  --target-time-s 1.0e-6 \
  --output results/first_principles_3d_py312_microsecond_request_probe_2026_05_16.json
```

Result summary:

- Target time: `1.0e-6 s`.
- Final simulated time: `2.0e-12 s`.
- Duration request satisfied: `false`.
- Termination reason: `completed_step_budget`.
- Steps completed: `20`.

## Experimental Whole-Shot Lane Added

The project now has an explicit non-promoting command for the engineer-review
case:

```bash
.venv312/bin/python -m dpf.cli.main experimental-whole-shot \
  --deck-preset pf1000_akel_16kv \
  --steps 20 \
  --history-stride 5 \
  --max-step-results 8 \
  --target-time-s 1e-6 \
  --output results/experimental_whole_shot_pf1000_py312_2026_05_16.json
```

Result summary:

- Tool field: `dpf experimental-whole-shot`.
- Experimental status:
  `experimental_whole_shot_candidate_not_validation`.
- Steps completed: `20`.
- Target time: `1.0e-6 s`.
- Final simulated time: `2.0e-12 s`.
- Duration request satisfied: `false`.
- Current deck timestep: `1.0e-13 s`.
- Required steps at current timestep: `10000000`.
- Explicit 3-D Yee vacuum-CFL timestep at `CFL=0.95`:
  `2.1590829252331814e-10 s`.
- Required steps at the vacuum-CFL timestep: `4632`.
- Current 20-step budget satisfies target: `false`.
- Vacuum-CFL 20-step budget satisfies target: `false`.
- Candidate module records executed: `20`.
- Active blocker records: includes requested-duration shortfall, current-step
  under-budget, vacuum-CFL under-budget, and particle-population growth.
- First-principles acceptance support: `false`.

This is the correct experimental lane for the current state: it exercises the
package-native source-grounded candidate modules and packages the outcome for
engineering review, while keeping validation, certificate, and whole-shot
authority gates closed.

The command also supports a non-promoting duration policy:

```bash
.venv312/bin/python -m dpf.cli.main experimental-whole-shot \
  --deck-preset pf1000_akel_16kv \
  --dt-policy vacuum-cfl \
  --vacuum-cfl 0.95 \
  --target-time-s 1e-9 \
  --auto-step-budget \
  --max-auto-steps 20 \
  --history-stride 5 \
  --max-step-results 4 \
  --output results/experimental_whole_shot_pf1000_vacuum_cfl_ns_probe_2026_05_16.json
```

This nanosecond probe completed `5 / 5` steps, reached
`1.0795414626165907e-9 s`, and reported
`duration_request_satisfied=true`. It is still experimental and non-validated,
but it proves the duration-control path can hit a requested horizon when the
step budget and explicit Maxwell timestep are compatible.

The same duration-control path was then pushed to a microsecond horizon:

```bash
.venv312/bin/python -m dpf.cli.main experimental-whole-shot \
  --deck-preset pf1000_akel_16kv \
  --dt-policy vacuum-cfl \
  --vacuum-cfl 0.95 \
  --target-time-s 1e-6 \
  --auto-step-budget \
  --max-auto-steps 5000 \
  --history-stride 500 \
  --max-step-results 8 \
  --output results/experimental_whole_shot_pf1000_vacuum_cfl_1us_probe_2026_05_16.json
```

Microsecond result summary:

- Steps completed: `4632 / 4632`.
- Target time: `1.0e-6 s`.
- Final simulated time: `1.0000872109680097e-6 s`.
- Duration request satisfied: `true`.
- Termination reason: `target_time_reached`.
- Finite-state check: `true`.
- Final macro-particle count: `78933`.
- Final candidate circuit current: `415959.7399182063 A`.
- Relative tracked energy change: `0.053497034664624125`.
- Candidate cumulative neutron count: `5.812613362679801`.
- Runtime numerical audit status:
  `experimental_numerical_runtime_audit_not_validation`.
- Vacuum-CFL audit: `dt_s / stable_vacuum_dt_s = 1.0`,
  `dt_within_vacuum_cfl = true`.
- Numerical-audit next priorities:
  mesh/timestep family runs, restart reproducibility artifacts, and limiter
  activation-count instrumentation.
- Acceptance status: `engineering_candidate_not_validation`.

This closes the immediate executable-duration blocker for a coarse-grid
experimental run. It does not close the physics or numerical authority blockers:
startup BVP, reviewed geometry/material surfaces, accepted power-port coupling,
same-scope waveform binding, convergence/restart proof, UQ, and
mechanism-separated neutron authority are still open.

The next numerical-fidelity tooling step is now executable through:

```bash
.venv312/bin/python -m dpf.cli.main experimental-numerical-family \
  --family timestep \
  --deck-preset pf1000_akel_16kv \
  --target-time-s 1e-9 \
  --timestep-scales 1,0.5 \
  --max-auto-steps 20 \
  --history-stride 5 \
  --max-step-results 4 \
  --output results/experimental_numerical_family_pf1000_timestep_probe_2026_05_16.json
```

That probe ran two same-horizon timestep cases and wrote
`experimental_numerical_family_probe_not_validation` with pairwise observable
deltas. The convergence decision remains
`not_assessed_no_accepted_tolerances`; this is troubleshooting infrastructure,
not accepted numerical fidelity.

The deterministic-rerun reproducibility probe is now executable through:

```bash
.venv312/bin/python -m dpf.cli.main experimental-reproducibility \
  --deck-preset pf1000_akel_16kv \
  --target-time-s 1e-9 \
  --repeat-count 2 \
  --dt-policy vacuum-cfl \
  --vacuum-cfl 0.95 \
  --max-auto-steps 20 \
  --history-stride 5 \
  --max-step-results 4 \
  --output results/experimental_reproducibility_pf1000_probe_2026_05_16.json
```

That probe wrote `experimental_reproducibility_probe_not_validation` with
two finite, duration-satisfied reruns and identical state-observable hashes:
`2ff168ede8091ec996450ff2f5d4f868edc2883d121848d34313b6378b95bd6f`.
Both reruns also emitted the same terminal state fingerprint hash:
`1f113f1765586d5f3c05984e22b369c9168c48d16e2b6091614b0f33385c3a06`.
This closes deterministic rerun and terminal-state hashing only.

Terminal checkpoint roundtrip is now executable through:

```bash
.venv312/bin/python -m dpf.cli.main experimental-state-checkpoint \
  --deck-preset pf1000_akel_16kv \
  --target-time-s 1e-9 \
  --dt-policy vacuum-cfl \
  --vacuum-cfl 0.95 \
  --max-auto-steps 20 \
  --history-stride 5 \
  --max-step-results 4 \
  --checkpoint-output results/experimental_state_checkpoint_pf1000_terminal_2026_05_16.npz \
  --output results/experimental_state_checkpoint_pf1000_probe_2026_05_16.json
```

That probe wrote
`experimental_state_checkpoint_roundtrip_not_restart_acceptance`; it serialized
17 evolved-state arrays to NPZ, read them back, and matched the content hash:
`f523f40a0f9aa7777f6c581675b0e15a28013ec1bf2475a92af12af850d61a85`.
Live same-process split continuation is now executable through:

```bash
.venv312/bin/python -m dpf.cli.main experimental-split-continuation \
  --deck-preset pf1000_akel_16kv \
  --steps 6 \
  --split-after-steps 3 \
  --dt-policy vacuum-cfl \
  --vacuum-cfl 0.95 \
  --history-stride 1 \
  --max-step-results 6 \
  --output results/experimental_split_continuation_pf1000_probe_2026_05_16.json
```

That probe wrote `experimental_split_continuation_probe_not_restart_acceptance`;
the `3 + 3` split matched the uninterrupted six-step run exactly by terminal
state fingerprint and tracked observables. The matching fingerprint hash was
`03003c3658b7d5e0c52789dba53319753c2cfbbfdb5626a5243753ad6fd5fb8d`.

Checkpoint-loaded restart is now executable through:

```bash
.venv312/bin/python -m dpf.cli.main experimental-checkpoint-restart \
  --deck-preset pf1000_akel_16kv \
  --steps 6 \
  --split-after-steps 3 \
  --dt-policy vacuum-cfl \
  --vacuum-cfl 0.95 \
  --history-stride 1 \
  --max-step-results 6 \
  --checkpoint-output results/experimental_checkpoint_restart_pf1000_midpoint_2026_05_16.npz \
  --output results/experimental_checkpoint_restart_pf1000_probe_2026_05_16.json
```

That probe wrote `experimental_checkpoint_restart_probe_not_validation`; the
fresh session loaded the midpoint checkpoint and matched the uninterrupted
six-step terminal state fingerprint exactly:
`03003c3658b7d5e0c52789dba53319753c2cfbbfdb5626a5243753ad6fd5fb8d`.

Multi-offset checkpoint-restart probing is now executable through:

```bash
.venv312/bin/python -m dpf.cli.main experimental-checkpoint-restart-family \
  --deck-preset pf1000_akel_16kv \
  --steps 6 \
  --split-after-steps 2,3,4 \
  --dt-policy vacuum-cfl \
  --vacuum-cfl 0.95 \
  --history-stride 1 \
  --max-step-results 6 \
  --checkpoint-dir results/experimental_checkpoint_restart_family_pf1000_2026_05_16 \
  --output results/experimental_checkpoint_restart_family_pf1000_probe_2026_05_16.json
```

That family wrote
`experimental_checkpoint_restart_family_probe_not_validation`; all three
offsets matched the uninterrupted six-step run. Full-horizon restart
acceptance remains open.

The GV PF-24 verified-shot preset was also run through the same experimental
path with its user-verified workbook waveform bound as an engineering comparator:

- Artifact:
  `results/experimental_whole_shot_gv_pf24_vacuum_cfl_1us_probe_2026_05_16.json`.
- Steps completed: `17598 / 17598`.
- Final simulated time: `1.0000205248587623e-6 s`.
- Duration request satisfied: `true`.
- Current waveform comparison status:
  `engineering_current_waveform_comparison_not_validation`.
- Simulation/target overlap points: `10`.
- Temporal coverage fraction of target: `0.15384931151673267`.
- Current RMSE over overlap: `47.84087376616515 kA`.
- Peak-current error fraction vs full target: `0.09800069763286687`.

This is not validation, but it confirms the experimental runner can bind a
user-verified current waveform as a non-driving, non-fitting comparator for a
second-scope device.

Machine geometry and boundaries:

- Resolved PF-1000 electrode topology at engineering CAD fidelity.
- Accepted cathode-rod voxel mask. A candidate 12-rod projection now exists,
  but no reviewed same-scope mask or mesh-quality certificate is attached.
- Accepted hollow-anode bore. The source establishes hollow-anode access, but
  the exact accepted inner radius/bore surface is still missing.
- Accepted insulator surface shape and material boundary. Alumina and exposed
  length are carried in telemetry, but material surfaces are not resolved in
  the active field/boundary solve.
- Axisymmetric/cylindrical geometric source terms in the package-native 3-D
  simulator.
- Reviewed same-scope conductor masks and boundary-validation evidence.

Full-shot duration and scaling:

- Microsecond-scale whole-shot horizon is now executable for a coarse-grid
  experimental PF-1000/Akel run under the vacuum-CFL timestep policy:
  `4632 / 4632` steps reached `1.0000872109680097e-6 s`. This is not
  accepted whole-shot authority because startup, geometry, power-port,
  convergence, restart, limiter-zero, UQ, and neutron-authority gates remain
  blocked.
- Production mesh/timestep convergence families.
- Backend/precision parity for the package-native first-principles path.
- Multi-offset checkpoint-restart is now available for physical target-time
  runs. A nanosecond PF-1000 probe selected `5` vacuum-CFL steps from
  `--target-time-s 1e-9` and matched split offsets `2,3,4` against the
  uninterrupted terminal fingerprint. Microsecond/full-horizon restart-family
  acceptance is still not complete.
- Full-horizon limiter-zero proof is now instrumented as an experimental
  packet. The PF-1000 nanosecond limiter probe reached the target horizon and
  completed the inventory, but still observed
  `conductivity_ohmic_cfl_limited_steps=5` and `marder_correction_steps=5`,
  so limiter-zero remains blocked pending Ohmic-CFL/nondominance review.
- An experimental `combined-cfl` timestep policy now takes the stricter of the
  3-D Yee vacuum CFL and the source-grounded explicit Ohmic relaxation limit.
  On a short PF-1000 limiter probe it selected
  `dt=1.839511686878276e-14 s`, reached `5.5185350606348273e-14 s`, and
  reduced acceptance-blocking limiter activations to `0`. Marder correction
  was observed as nondominant on that probe
  (`max_relative_correction_linf=0.09902109928920329` below threshold `0.5`,
  `dominant_correction_steps=0`). This is not a whole-shot solution yet: a
  `1.0e-6 s` PF-1000 run at that explicit timestep would require about
  `54,362,254` steps.

Power port and circuit authority:

- Accepted terminal voltage/current power-port definition.
- Time-centered Poynting or `J.E` power transfer with sign convention and
  residual budget.
- A candidate residual-budget ledger is now emitted in the package-native
  power-port packet. The combined-CFL short PF-1000 probe records tracked
  energy delta, retained volume `J.E` work, both sign hypotheses, and retained
  history coverage. It remains non-accepting because sign convention,
  centering, interface/domain, electrode-work partition, and reviewed residual
  tolerance are still missing.
- Electrode work partition and named interface/domain.
- Accepted active load relation. Current lagged `J.E` feedback is candidate
  only.

Physics closures:

- Accepted EOS or tabular thermodynamics for the whole plasma/material regime.
- Accepted two-temperature closure with collisional coupling, heat flux, and
  diagnostic validation.
- Full radiation-loss/transport model and opacity/diffusion decision.
- Electrode ablation, impurity source, and impurity transport.
- Restrike and anomalous resistance model.
- Accepted Hall/FLR/electron-kinetic scope or handoff.
- PIC Coulomb collision/stopping model in the active first-principles runner.
- Beam formation, beam transport, target stopping, and beam-target neutron
  source authority.

Instabilities and kinetic physics:

- Accepted 3-D mode evidence for kink, sausage, fragmentation, and pinch
  lifetime.
- Electron kinetic instabilities and acceleration physics.
- Mechanism-separated thermonuclear vs beam-target histories.
- Neutron spectrum, timing, anisotropy, direct/scattered transport, and
  detector/activation response.

Same-scope experimental comparison:

- Accepted digitized PF-1000/Akel current waveform with per-point time/current
  uncertainty.
- Current derivative/dip phase packet and phase semantics.
- Same-shot density, field, electron-temperature, ion-temperature, and ion
  distribution histories.
- Comparator metrics, tolerances, output mappings, units/coordinates, and UQ
  propagation.
- Independent review metadata and first-principles certificate gate.

Generalization:

- A second-device or second-shot first-principles certificate.
- Cross-scope transfer rules or rejection tests.
- No-hidden-PF-1000/Akel-assumption proof.
