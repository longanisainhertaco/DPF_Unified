# FP-4 Numerical-Fidelity Source Search

Date: 2026-05-15  
Scope: local `KnowledgeReference/`, first-principles docs, and package-native numerical test surfaces.

## Verdict

FP-4 is blocked. The repo has useful component tests and the package-native
runner emits finite conservation and `div B` diagnostics, but there is no
accepted numerical-fidelity packet for a full first-principles DPF shot. The
missing authority is not more ad hoc tests; it is a source-backed, typed packet
with named test surfaces, norms, mesh/time families, tolerances, convergence
evidence, backend/precision scope, limiter-zero proof, artifact hashes, and
independent review.

The package-native runner now emits a fail-closed `numerical_fidelity` packet
with status `blocked_numerical_fidelity_packet_not_available`.

## Source Findings

### Project Gate

- `docs/FIRST_PRINCIPLES_FINISH_LINE_PLAN.md:243-260` defines the numerical
  verification surfaces: finite-volume shocks, cylindrical source terms,
  `div B`, resistive diffusion, Joule heating, circuit-coupled energy, restart,
  backend/precision parity, and limiter-zero acceptance.
- `docs/DPF_REQUIREMENTS_BASELINE.md:70` states that first-principles
  numerical-fidelity packets must define named tests, norms, mesh families,
  tolerances, precision/backend scope, and limiter-zero acceptance.

### Numerical Method Sources

- `KnowledgeReference/fully-electromagnetic-hybrid-pic-fluid-dpf-neutron-yield-acb71fa9.md:410-424`
  supports Marder/Gauss-law correction as a candidate continuity-control
  requirement for hybrid PIC-fluid runs.
- `KnowledgeReference/fully-electromagnetic-hybrid-pic-fluid-dpf-neutron-yield-acb71fa9.md:609-645`
  supports Yee/FDTD grid placement, PML/conductor particle boundaries, and CFL
  time-step requirements for the hybrid architecture.
- `KnowledgeReference/fully-electromagnetic-hybrid-pic-fluid-dpf-neutron-yield-acb71fa9.md:1018-1110`
  supports mesh/time-step refinement, conductivity-threshold sensitivity,
  Ohmic CFL sensitivity, Marder-factor sensitivity, and electron-temperature
  closure sensitivity as required UQ dimensions.
- `KnowledgeReference/particle-simulation-of-plasmas-review-and-advances-6d7355ba.md:456-530`
  supports Yee/leapfrog Maxwell update structure and preservation of
  divergence constraints when initialized consistently.
- `KnowledgeReference/particle-simulation-of-plasmas-review-and-advances-6d7355ba.md:671-705`
  supports the multidimensional Courant time-step criterion.
- `KnowledgeReference/particle-simulation-of-plasmas-review-and-advances-6d7355ba.md:744-755`
  supports charge-conserving current deposition or explicit Gauss-law
  correction when current weighting is not exactly charge-conserving.
- `KnowledgeReference/a-constrained-transport-embedded-boundary-method-for-compressible-resistive-magnetohydrodynamics.md:55-90`
  supports finite-volume Riemann fluxes, face-centered constrained transport,
  and convergence verification.
- `KnowledgeReference/a-constrained-transport-embedded-boundary-method-for-compressible-resistive-magnetohydrodynamics.md:429-500`
  supports discrete-curl CT preservation of `div B`, resistive electric-field
  coupling, and resistive time-step constraints.
- `KnowledgeReference/2019nrlplasma-formulary-037290d4.md:2444-2576`
  provides shock-relation context for shock fixtures.

## Required Packet Channels

The numerical-fidelity packet must block until all channels are accepted:

- test surface registry;
- source-backed numerical method map;
- analytic or manufactured reference solutions;
- mesh and time-step family definitions;
- norms, tolerances, and observed order or monotonic convergence by surface;
- finite-volume shock packet;
- cylindrical source-term packet;
- Maxwell/Yee/Courant packet;
- `div B` packet;
- Gauss-law or charge-current-continuity packet;
- resistive-diffusion packet;
- Joule-heating and total-energy packet;
- circuit power-port numerical packet;
- particle push/deposition packet;
- mesh/time-step convergence packet;
- restart reproducibility packet;
- backend/precision parity packet;
- limiter-zero packet;
- same-scope numerical observable mapping;
- artifact links and hashes;
- negative tests for failed tolerance;
- independent review certificate.

## Existing Candidate Test Surfaces

These are useful engineering surfaces, not accepted FP-4 authority:

- `tests/test_maxwell_3d_field_core.py`: Maxwell/Yee diagnostics and `div B`
  component coverage.
- `tests/test_marder_correction.py`: Marder residual and nondominance coverage.
- `tests/test_pic_current_source_port.py`: PIC-current to Yee-edge mapping.
- `tests/test_hybrid_3d_loop.py`: hybrid PIC-field loop telemetry.
- `tests/test_circuit_magnetic_boundary.py`: external-circuit magnetic
  boundary slice.
- `tests/test_first_principles_runner.py`: package-native manifest and finite
  conservation telemetry.

Additional older verification surfaces (`test_phase_c_verification.py`,
`test_phase_f_verification.py`, `test_phase_n_cross_backend.py`,
`test_phase_o_physics_accuracy.py`) may supply candidate components, but they
must be reattached through the first-principles packet with current artifact
hashes, tolerances, scope, and review before acceptance.

## Implementation Ratchet

- Added `src/dpf/first_principles/numerical_fidelity.py`.
- Wired `numerical_fidelity` into `src/dpf/first_principles/runner.py`,
  certificate-gate upstream statuses, generalization upstream statuses,
  validation packet summaries, deck manifest config, and manifest candidate
  evidence.
- Exported `build_numerical_fidelity_packet` from
  `src/dpf/first_principles/__init__.py`.
- Added focused assertions in `tests/test_first_principles_runner.py`.

The packet records runtime conservation, `div B`, and hybrid-loop channels as
candidate telemetry only. It cannot promote without accepted tolerances,
convergence, limiter-zero proof, backend scope, hashes, and review.

Additional ratchet added in this pass:

- `src/dpf/first_principles/numerical_fidelity.py` now emits
  `numerical_channel_status`, `test_surface_status`, `runtime_observations`,
  `upstream_acceptance_gate`, `acceptance_gate`, and `negative_test_policy`.
- Current component and legacy verification tests are classified as
  candidate-only coverage until every required surface has source-backed
  reference solutions, norms, tolerances, convergence evidence, artifact hashes,
  limiter-zero scope, and review.
- Runtime conservation, `div B`, circuit, electron-energy, and kinetic-yield
  observations are explicitly non-promoting numerical telemetry.
- Startup, limiter-readiness, power-port, dimensionality, and closure packets
  are recorded as upstream blockers for accepted whole-shot numerical authority.

## Validated Commands

- `python3 -m pytest tests/test_first_principles_runner.py` -> 8 passed.
- `python3 -m json.tool docs/FIRST_PRINCIPLES_NUMERICAL_FIDELITY_SOURCE_SEARCH_2026_05_15.json` -> valid JSON.
- `python3 -m pytest tests/test_first_principles_input_deck.py tests/test_first_principles_runner.py tests/test_first_principles_manifest.py tests/test_cli_first_principles_3d.py tests/test_cli_backend_options.py tests/test_server_readiness.py tests/test_kinetic_yield_history.py tests/test_hybrid_3d_loop.py tests/test_hybrid_pic_3d_validation_packet.py` -> 60 passed.
