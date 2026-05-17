# First-Principles Blocker Source Search - Power Port

Date: 2026-05-15

Scope: local source of truth only. Scientific claims in this note are limited to
`KnowledgeReference/` and source-truth index artifacts already in the repo.

Blocker: `FP-6`, resolved field/circuit power-port coupling.

Question: can the source of truth close the circuit-to-field coupling blocker
for a whole-shot first-principles DPF simulator?

## Verdict

The source of truth gives a clear first-principles contract for circuit-field
coupling, but does not yet close accepted authority for the implemented
geometry and timestep path.

The accepted circuit load must come from electromagnetic field power, either as
a named Poynting surface flux or an equivalent volume `J.E` relation, with
terminal voltage/current, electrode work, time centering, sign convention, and
energy residuals recorded. A magnetic-energy inductance such as
`L_field = 2 E_B / I^2` can remain diagnostic, but it is not accepted as the
circuit load unless a reviewed power-port packet proves equivalence for the
claimed interval.

Therefore `FP-6` remains blocked for accepted whole-shot authority until the
package-native solver emits and tests a power-port packet for the implemented
geometry.

## Source Answers

| Source | What it answers | What remains blocked |
| --- | --- | --- |
| `docs/FIRST_PRINCIPLES_SOURCE_TRUTH_INDEX.md:43-49` and `docs/FIRST_PRINCIPLES_SOURCE_TRUTH_INDEX.md:89-94` | The index identifies circuit coupling and Poynting/`J.E` power port as direct capability buckets, and explicitly lists the exact implemented power-port evidence as still missing. | Source index is a routing and gap artifact; it does not itself validate the implemented port. |
| `KnowledgeReference/auluck-2021-dpf-circuit-element.md:151-200` | Defines terminal voltage for a plasma-focus circuit element from field power divided by current, using a 3D domain integral over `J.E`. It also warns that electrode-connected electric-field paths can be spatially complex. | The implementation still needs a concrete integration domain, electrode terminals, sign convention, and discretization for the active grid. |
| `KnowledgeReference/auluck-2021-dpf-circuit-element.md:206-262` and `KnowledgeReference/auluck-2021-dpf-circuit-element.md:426-445` | Uses Poynting theorem and identifies the power-source interface as a boundary surface. For the coaxial region, Poynting flux through the excluded generator interface equals input power. | The code must declare the interface surface and prove discrete Poynting and `J.E` ledgers agree or explain the controlled difference. |
| `KnowledgeReference/auluck-2021-dpf-circuit-element.md:1026-1047` | Shows why defining plasma inductance only from magnetic energy is incomplete for motional impedance and can create apparent unaccounted impedance. | `L_field` remains diagnostic-only until a reviewed field-power packet supports use as the load. |
| `KnowledgeReference/fully-electromagnetic-hybrid-pic-fluid-dpf-neutron-yield-acb71fa9.md:740-805` | Supplies a concrete hybrid PIC-fluid circuit pattern: solve current from the external circuit, use current to set magnetic boundary conditions, calculate DPF voltage feedback from magnetic-field integration, and update charge/current. | The source uses an end-of-rundown initialization and an explicit time update; the local code still needs accepted sign, centering, electrode-work partition, and residual tests. |
| `KnowledgeReference/fully-electromagnetic-hybrid-pic-fluid-dpf-neutron-yield-acb71fa9.md:992-1005` | Supports current/voltage histories as diagnostics, with voltage spikes correlated to pinch flux changes. | Current/voltage traces alone do not validate field coupling, spatial fields, or neutron authority. |
| `KnowledgeReference/beresnyak_2018_dpf_hawk_simulations.md:170-200` | Gives a working MHD-circuit coupling pattern: circuit current and `dI/dt` enter boundary conditions, and DPF terminal voltage is returned to the circuit by integrating electric field across terminals. | Hawk plasma-injection geometry is a separate scope, and the method does not provide accepted PF-1000/Akel whole-shot packet data. |
| `KnowledgeReference/beresnyak_2022_pulsed_power_ideal_mhd.md:44-72` | Reinforces that vacuum/low-density regions transmit electromagnetic power and therefore belong to the coupling problem. | The local implementation must still represent conductor/vacuum/plasma interfaces consistently enough for the claimed scope. |

## Accepted Power-Port Contract

An accepted first-principles DPF run needs a per-step power-port packet with:

- named interface surface or named volume domain;
- terminal current and terminal voltage definitions;
- Poynting flux ledger or equivalent `J.E` ledger;
- electrode work and external-circuit energy;
- magnetic, electric, thermal, kinetic, particle, radiation, and residual terms;
- sign convention and current direction;
- time-centering or subcycling metadata;
- geometry and boundary labels for conductor, insulator, plasma, vacuum, and
  power-source interface;
- explicit startup handoff interval if startup is imported or engineering-only;
- no hidden current floor, clipped back-EMF, or fallback reduced-model load in
  accepted mode.

Acceptance tests must include:

- standalone sign convention fixture;
- closed-form Poynting or `J.E` manufactured fixture;
- discrete conservation residual test;
- electrode-work partition test;
- handoff interval test proving no unrecorded startup or reduced-model segment;
- integrated PF-1000/Akel run packet with tolerances and UQ.

## Implementation Impact

Immediate implementation requirements:

- Keep the current implicit-midpoint field-load candidate as engineering-only.
- Add or extend a package-native `PowerPortPacket` emitted by the runner and
  manifest.
- Record both the active load used by the circuit update and diagnostic
  alternatives such as `L_field`.
- Fail accepted claims when the port lacks source references, sign convention,
  centering metadata, electrode-work partition, or residual tolerance.

Next blocker to search after this one: `FP-7`, dimensionality, 3D hybrid
PIC-fluid completeness, and MHD-to-kinetic handoff.

## Current Implementation Ratchet

Implemented after this source search:

- `src/dpf/first_principles/power_port.py` now emits a richer fail-closed
  package-native packet with terminal current/voltage, active placeholder load,
  per-step candidate energy-ledger fields, startup-handoff blocker, source
  references, and diagnostic-only field inductance.
- `src/dpf/fields/hybrid_stepper.py` now records a candidate full-grid volume
  `J.E` field-work integral from the solved generalized-Ohm current and
  cell-centered electric field. The package-native power-port packet carries
  this as `j_dot_e_power_W` and marks
  `poynting_power_or_j_dot_e` as candidate runtime evidence only.
- `src/dpf/fields/hybrid_simulator.py` can now use lagged candidate volume
  `J.E` feedback for the circuit update via
  `U_DPF = - integral(J.E)dV / I`. This is a real field-power feedback ratchet,
  but remains non-promoting because it is lagged, full-grid, and lacks accepted
  sign, centering, electrode-work, interface/domain, and residual packets.
- The packet explicitly separates the active load relation from
  both `L_field = 2 E_B/I^2` and candidate volume `J.E`; neither relation is
  accepted as the circuit load without a reviewed domain/sign/centering/
  electrode-work/residual packet.
- The built-in package-native first-principles deck now defaults to
  `circuit_udpf_mode = lagged_volume_j_dot_e`; compact JSON decks may override
  this, but any unknown mode fails closed.
- The packet now emits `power_port_channel_status`, `energy_ledger_status`,
  `active_load_decision`, `acceptance_gate`, `negative_test_policy`, and
  `residual_policy`.
- Candidate terminal current/voltage, volume `J.E`, and energy-ledger values
  are explicitly runtime-only evidence. Named Poynting surface or reviewed
  volume `J.E`, electrode work, sign, centering, residual tolerance, and review
  remain blocking acceptance channels.
- `tests/test_first_principles_runner.py` verifies the runner and manifest keep
  the power-port packet non-promoting and include diagnostic-only
  field-inductance plus candidate volume-`J.E` evidence.

Verified command:

- `python3 -m pytest tests/test_first_principles_runner.py tests/test_hybrid_3d_loop.py tests/test_cli_first_principles_3d.py`
  -> `22 passed`.
- `python3 -m pytest tests/test_hybrid_3d_simulator.py tests/test_first_principles_runner.py tests/test_first_principles_input_deck.py tests/test_cli_first_principles_3d.py`
  -> `26 passed`.
- `python3 -m pytest tests/test_first_principles_input_deck.py tests/test_first_principles_runner.py tests/test_first_principles_manifest.py tests/test_cli_first_principles_3d.py tests/test_cli_backend_options.py tests/test_server_readiness.py tests/test_kinetic_yield_history.py tests/test_hybrid_3d_loop.py tests/test_hybrid_pic_3d_validation_packet.py`
  -> `60 passed`.

Remaining blocker:

- No accepted package-native Poynting surface or reviewed volume `J.E` integral,
  electrode-work partition, sign convention, time-centering, residual tolerance,
  artifact hash, or review packet exists for the implemented 3D geometry.
