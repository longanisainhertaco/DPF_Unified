# FP-2/FP-3 Limiter-Readiness Source Search

Date: 2026-05-15  
Scope: local `KnowledgeReference/`, first-principles docs, and package-native runner state.

## Verdict

FP-2 and FP-3 remain blocked for accepted first-principles claims. The project
has partial app/Python-path limiter telemetry and bounded short probes, but the
package-native first-principles runner did not yet expose a formal
limiter-readiness packet. A true first-principles DPF shot cannot hide floors,
caps, clips, repairs, timestep caps, current/back-EMF caps, backend fallbacks,
precision fallbacks, or solver-layer repairs behind normal-looking outputs.

The package-native runner now emits a fail-closed `limiter_readiness` packet
with status `blocked_limiter_readiness_packet_not_available`.

## Source Findings

### Project Gate

- `docs/FIRST_PRINCIPLES_FINISH_LINE_PLAN.md:61-63` requires every numerical
  limiter, floor, cap, repair, fallback, source-term intervention, subcycle, and
  precision change to be classified as a verified method, source-backed physical
  bound, or acceptance blocker.
- `docs/FIRST_PRINCIPLES_FINISH_LINE_PLAN.md:159-160` defines FP-2 as the
  global limiter registry and FP-3 as the limiter-free or physically bounded
  candidate.
- `docs/FIRST_PRINCIPLES_FINISH_LINE_PLAN.md:210-239` defines the active-path
  registry fields and requires a full-run limiter ledger with zero
  `acceptance_blocker` activations.
- `docs/DPF_REQUIREMENTS_BASELINE.md:64` states that hidden engineering
  limiters must be rejected for accepted PF-1000/Akel first-principles claims.

### Source Method Context

- `KnowledgeReference/fully-electromagnetic-hybrid-pic-fluid-dpf-neutron-yield-acb71fa9.md:410-424`
  supports Marder correction as a numerical continuity-control mechanism that
  must be kept small enough not to perturb stability or dynamics.
- `KnowledgeReference/fully-electromagnetic-hybrid-pic-fluid-dpf-neutron-yield-acb71fa9.md:582-605`
  supports an Ohmic CFL conductivity limiter as a numerical stability guard,
  not an uninspected physical claim.
- `KnowledgeReference/fully-electromagnetic-hybrid-pic-fluid-dpf-neutron-yield-acb71fa9.md:1046-1067`
  supports sensitivity testing for conductivity threshold, Ohmic CFL factor, and
  Marder factor before those controls can be treated as nondominant.
- `KnowledgeReference/a-constrained-transport-embedded-boundary-method-for-compressible-resistive-magnetohydrodynamics.md:326,471-500,952`
  supports distinguishing verified numerical limiters/time-step constraints
  from hidden state repairs.

## Required Packet Channels

The limiter-readiness packet must block until all channels are accepted:

- active-path limiter inventory;
- limiter event schema;
- code path and affected-field mapping;
- classification by limiter;
- activation counts;
- before/after min/max;
- nonfinite counts;
- source or numerical-method justification;
- readiness effect by limiter;
- source-backed physical bounds;
- verified numerical method bounds;
- zero acceptance-blocker full run;
- full-horizon run manifest;
- backend/precision fallback inventory;
- fallback rejection tests;
- synthetic acceptance-blocker negative test;
- app-only runner rejection test;
- artifact links and hashes;
- independent review certificate.

## Known Limiter Families

- State-mutating floors/caps/clips: density floors, temperature floors/caps,
  pressure floors, velocity/current clips, and back-EMF clips. These block
  acceptance unless they are source-backed physical bounds or verified numerical
  methods with impact evidence.
- Method limiters and stability guards: finite-volume slope limiters, Ohmic CFL
  conductivity limits, Marder correction, and resistive time-step constraints.
  These remain candidate until nondominance, convergence, and sensitivity
  evidence are accepted.
- Repairs and fallbacks: nonfinite repair, backend precision fallback,
  unsupported-physics fallback, surrogate fallback, and reduced-model fallback.
  These block accepted first-principles claims.

## Implementation Ratchet

- Added `src/dpf/first_principles/limiter_readiness.py`.
- Wired `limiter_readiness` into `src/dpf/first_principles/runner.py`,
  numerical-fidelity upstream statuses, certificate-gate upstream statuses,
  generalization upstream statuses, validation packet summaries, deck manifest
  config, and manifest candidate evidence.
- Exported `build_limiter_readiness_packet` from
  `src/dpf/first_principles/__init__.py`.
- Added focused assertions in `tests/test_first_principles_runner.py`.

The packet records finite conservation and runtime telemetry as candidate
evidence only. It cannot promote without a full active-path limiter inventory,
zero acceptance-blocker full-horizon run proof, fallback rejection tests,
artifact hashes, and review.

## Validated Commands

- `python3 -m pytest tests/test_first_principles_runner.py` -> 7 passed.
- `python3 -m json.tool docs/FIRST_PRINCIPLES_LIMITER_READINESS_SOURCE_SEARCH_2026_05_15.json` -> valid JSON.
- `python3 -m pytest tests/test_first_principles_input_deck.py tests/test_first_principles_runner.py tests/test_first_principles_manifest.py tests/test_cli_first_principles_3d.py tests/test_kinetic_yield_history.py tests/test_hybrid_3d_loop.py` -> 33 passed.
