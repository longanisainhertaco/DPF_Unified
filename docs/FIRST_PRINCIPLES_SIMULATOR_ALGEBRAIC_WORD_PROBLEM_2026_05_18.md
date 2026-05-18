# The DPF Simulator As A Solve-Ready Algebraic Word Problem

Date: 2026-05-18

Input framing: `/Users/anthonyzamora/Downloads/deep-research-report.md`

## Executive Summary

The dense plasma focus simulator problem is not yet a fully solvable algebraic
word problem. It is a partially specified word problem with executable
subsystems.

That distinction matters. A normal word problem becomes solvable only after the
full plain-language statement supplies:

- what is given
- what is unknown
- the units
- the relationships between quantities
- the constraints
- what must be solved for
- how the answer will be checked

The DPF simulator currently has many givens and some working equations. It also
has a working experimental 3-D runtime that reached `12 us` on a coarse
PF-1000/Akel engineering deck. But it does not yet have every relationship
needed to determine a unique, accepted first-principles whole-shot solution.

So the correct algebraic answer is not to invent missing constants or tune
parameters until the output looks right. The correct answer is to make the
underdetermination explicit, list the missing information, solve one missing
relationship at a time, and verify each relationship against physics,
numerics, and same-scope observables.

Current status:

```text
Executable experimental simulator: yes
Complete first-principles whole-shot solution: no
Reason: missing relationships and constraints still make the problem underdetermined
```

## The Word Problem

A PF-1000/Akel dense plasma focus machine has a known capacitor bank, bank
voltage, static inductance, static resistance, deuterium gas fill, electrode
dimensions, insulator, rods, hollow anode, and diagnostics. A shot begins from
neutral gas and electrode/insulator surfaces, breaks down, forms a current
sheath, runs down the anode, pinches, emits radiation and neutrons, and decays.

We have a simulator that can advance a coarse 3-D experimental version of this
shot. It carries:

```text
RLC circuit state
terminal current
3-D electromagnetic fields
PIC particles
charge and current deposition
generalized-Ohm current
electron-energy telemetry
ionization telemetry
conservation ledgers
power-port telemetry
limiter inventory
fail-closed review packets
```

One current artifact reached:

```text
artifact = results/experimental_limiter_proof_pf1000_seeded_power_domain_12us_2026_05_18.json
time = 12.000182898446022 us
steps = 55580
finite_state = true
acceptance_blocking_limiter_activations = 0
final_current = 1.200240 MA
status = engineering_candidate_not_validation
```

Question:

```text
What missing physics, numerical constraints, source-backed inputs, and review
packets must be supplied so that this executable candidate becomes a complete
first-principles 3-D DPF whole-shot simulator?
```

## Current Problem Status

Let:

```text
T = full DPF simulator word-problem text
M(T) = model that translates T into equations, operators, constraints, and tests
```

For a complete first-principles DPF simulator, `T` must include the machine
inputs, startup physics, geometry, field equations, plasma closures, circuit
coupling, neutron mechanisms, diagnostic observables, and numerical acceptance
criteria.

Current state:

```text
T is incomplete
M(T) is only partially defined
```

Therefore:

```text
No unique accepted first-principles whole-shot solution exists yet.
```

This does not mean the simulator is useless. It means the simulator is an
experimental candidate that can reduce the unknowns. It can run parts of the
problem and expose which relationships are missing or inconsistent.

## Given

Known machine-level inputs, in algebraic form:

```text
C0 = capacitor-bank capacitance
V0 = bank voltage
L0 = static circuit inductance
R0 = static circuit resistance
p0 = deuterium fill pressure
Tgas = initial gas temperature
G_candidate = candidate 3-D PF-1000 geometry
M_candidate = candidate material and boundary description
```

Known executable runtime:

```text
S_current = package-native 3-D hybrid EM/PIC-fluid experimental simulator
t_reached = 12 us in conservative fallback power-port mode
state_finite = true
limiter_acceptance_blockers = 0
reduced_model_authority = false on this path
```

Known source-sign power-port candidate:

```text
U_DPF = - integral_over_Omega(J dot E)dV / I
```

Known problem flags:

```text
startup_BVP_status = not accepted
power_port_status = candidate
geometry_status = candidate
closure_status = candidate/scaffolded
neutron_authority_status = blocked
same_scope_comparison_status = blocked
numerical_fidelity_status = blocked
certificate_status = blocked
```

## Asked

Find the missing vector:

```text
X = {
  startup_BVP,
  reviewed_geometry_masks,
  accepted_power_port,
  accepted_plasma_closures,
  accepted_collision_and_stopping_closures,
  accepted_neutron_mechanism_model,
  same_scope_observable_targets,
  numerical_fidelity_packet,
  engineering_certificate_packet
}
```

such that:

```text
S_full = S_current + X
```

and:

```text
S_full can run a whole DPF shot from startup through post-pinch
S_full uses first-principles physics only
S_full exposes all residuals and assumptions
S_full can be tested and challenged by engineers
```

## Why This Is Underdetermined

A word problem is underdetermined when there are more unknowns than independent
relationships. The DPF simulator is currently underdetermined because multiple
missing relationships can affect the same output.

Example:

```text
measured_current(t) can change because of:
  startup timing
  preionization
  sheath conductivity
  electrode geometry
  plasma inductance
  U_DPF sign convention
  electrode work
  radiation loss
  anomalous resistance
  numerical diffusion
```

If we tune the simulator to match current alone, we cannot tell which of those
causes is physically correct. Current agreement by itself is not enough
independent information to solve for all unknowns.

The same problem applies to neutron yield:

```text
Y_neutron can change because of:
  thermal ion temperature
  beam formation
  beam-target reactions
  density history
  pinch lifetime
  impurity radiation
  detector response
  anisotropy
```

Matching a single final neutron number does not uniquely solve the mechanism.

Therefore the algebraic task is:

```text
Do not infer every unknown from the end result.
Instead, add independent source-backed relationships until the unknowns are determined.
```

## Minimum Information Needed To Make The Problem Solve-Ready

| Missing component | Why it is indispensable |
|---|---|
| Exact startup state and equations | Without breakdown, flashover, preionization, and sheath liftoff, the whole shot starts from an invented or candidate state. |
| Reviewed geometry and boundary masks | Field and particle boundary conditions depend on rods, hollow anode, insulator, material surfaces, and open boundaries. |
| Accepted power-port definition | The circuit cannot be first-principles unless `U_DPF` is tied to accepted electromagnetic terminal work. |
| Sign convention and time-centering | `U_DPF = - integral(J.E)dV / I` can change circuit energy flow depending on sign and temporal placement. |
| Electrode-work partition | A volume `J.E` integral is incomplete if electrode/interface work is omitted or double counted. |
| Plasma closure set | Electron/ion energy, conductivity, ionization, radiation, heat flux, collisions, stopping, ablation, and anomalous transport need source-backed operators. |
| Neutron mechanism separation | Total yield must be decomposed into thermonuclear, beam-target, spectrum, anisotropy, and detector-response pieces. |
| Same-scope observables | The simulator must compare to the same machine, shot/family, geometry, gas, voltage, diagnostics, and timing definitions. |
| Numerical acceptance criteria | Convergence, restart reproducibility, backend parity, limiter-zero proof, and residual tolerances are required to reject numerical coincidences. |
| Engineering certificate gate | The final packet must let reviewers see what passed, what failed, what is candidate, and what is not claimed. |

## Clarifying Questions For The DPF Word Problem

Before the problem can be considered fully solve-ready, these questions need
answers from source-truth documents, code artifacts, or engineering review.

| Clarifying question | Why it matters |
|---|---|
| What exact PF-1000 shot or accepted shot family is the target? | Prevents comparing a simulation to mismatched machine conditions. |
| What are the reviewed values of `C0`, `V0`, `L0`, `R0`, gas pressure, and fill species? | These are coefficients in the circuit and plasma equations. |
| What is the reviewed 3-D geometry including rods, hollow anode, insulator, and chamber boundaries? | Defines field, particle, and material boundary conditions. |
| What is the initial neutral/preionized state before breakdown? | Defines the initial condition rather than a fitted seed. |
| What breakdown and flashover equations are accepted for this surface/gas/voltage regime? | Determines startup and sheath formation. |
| What is the accepted control volume `Omega` for `integral(J.E)dV`? | Determines the power-port voltage. |
| Is terminal power represented by Poynting flux, volume `J.E`, electrode work, or a reviewed combination? | Defines energy transfer between circuit and plasma. |
| What sign convention makes positive power flow unambiguous? | Prevents active-load and generator-feedback confusion. |
| What time-centering is required for `I`, `U_DPF`, fields, and current? | Prevents numerical energy mismatch. |
| Which plasma closures are accepted in each phase of the shot? | Avoids applying formulas outside their validity range. |
| What observables are available at matching scope? | Determines whether model output can be checked independently. |
| What tolerances define success? | Turns qualitative agreement into an engineering acceptance test. |
| What numerical changes must leave the answer invariant? | Separates physics from grid, time-step, particle, and backend artifacts. |

## DPF Word-Problem Model Families

The generic report describes template families such as algebraic equations,
rate problems, mixtures, work, geometry, probability, and optimization. The DPF
version has analogous model families.

| DPF model family | Generic word-problem analogue | Core relationship | Current status |
|---|---|---|---|
| Circuit evolution | Rate/equation problem | `d(L0 I)/dt = V0 - R0 I - Q/C0 - U_DPF` | Executable, but `U_DPF` is not accepted. |
| Power port | Work/energy-transfer problem | `U_DPF I = terminal electromagnetic work` | Most urgent blocker. |
| Startup BVP | Initial-condition problem | neutral gas and surfaces -> breakdown -> sheath liftoff | Blocked. |
| Geometry/boundaries | Geometry word problem | reviewed dimensions -> field/particle/material masks | Candidate. |
| Plasma closures | Coupled-system problem | source-backed operators for state evolution | Candidate/scaffolded. |
| Neutron authority | Decomposition problem | `Y_total = Y_thermal + Y_beam_target + ...` | Blocked. |
| Same-scope comparison | Data-fitting with constraints | simulated observables compared to same-scope measurements | Blocked. |
| Numerical fidelity | Verification problem | solution invariant under controlled numerical changes | Blocked. |
| Engineering review | Acceptance-gate problem | all required packets pass or fail explicitly | Blocked. |

## Equations And Constraints

### Circuit Equation

```text
d(L0 I)/dt = V0 - R0 I - Q/C0 - U_DPF
dQ/dt = I
```

Known:

```text
C0, V0, L0, R0 can be supplied from machine data.
```

Unknown:

```text
U_DPF(t) as accepted terminal voltage.
```

### Power-Port Equation

Candidate source-sign form:

```text
U_DPF = - integral_over_Omega(J dot E)dV / I
```

Acceptance constraints:

```text
Omega must be reviewed.
J dot E must be computed over the correct current-carrying domain.
Poynting flux and volume work must be reconciled.
Electrode/interface work must be accounted for.
Sign convention must be explicit.
Time-centering must be explicit.
Residual must be below tolerance.
Negative tests must fail when sign, domain, or time-centering are wrong.
```

Current evidence:

```text
12 us fallback run:
  candidate_lagged_volume_j_dot_e = 1076 steps
  input_sequence_fallback_first_step = 1 step
  input_sequence_fallback_negative_j_dot_e_active_port_blocked = 54503 steps

100 ns source-sign run:
  candidate_lagged_auluck_volume_j_dot_e = 463 steps
  input_sequence_fallback_first_step = 1 step
```

Interpretation:

```text
The source-sign branch exists and is testable.
The fallback branch reaches 12 us.
The accepted power-port equation is not solved yet.
```

### Startup Equation

Required transformation:

```text
neutral gas + electrode/insulator surfaces + applied fields
  -> breakdown
  -> surface flashover
  -> plasma layer
  -> sheath liftoff
  -> rundown initial condition
```

Current state:

```text
startup_BVP = missing
seeded_layer = engineering candidate only
```

### Plasma State Equation

State vector:

```text
Y_plasma = {n_e, n_i, Z, rho, v_i, T_e, T_i, E, B, particles}
```

Required operator:

```text
dY_plasma/dt = F_fields + F_particles + F_collisions + F_ionization
              + F_energy + F_radiation + F_materials + F_boundaries
```

Current state:

```text
some candidate operators exist
complete accepted closure set does not
```

### Neutron Equation

Required decomposition:

```text
Y_total(t,E,angle) = Y_thermonuclear + Y_beam_target + Y_other
```

Required attached models:

```text
ion distribution
beam formation
beam transport
target density
spectrum
anisotropy
detector response
```

Current state:

```text
mechanism-separated neutron authority = missing
```

### Same-Scope Comparison Equation

For an engineering target:

```text
sim_current(t) ~= measured_current(t)
sim_phase_times ~= measured_phase_times
sim_density(x,t) ~= measured_density(x,t)
sim_temperature(x,t) ~= measured_temperature(x,t)
sim_fields(x,t) ~= measured_fields(x,t)
sim_neutrons(t,E,angle) ~= measured_neutrons(t,E,angle)
```

Constraint:

```text
same machine
same shot or accepted shot family
same geometry
same gas fill
same bank state
same diagnostic definitions
same timing convention
```

Current state:

```text
same_scope_observable_targets = missing or incomplete
```

### Numerical Fidelity Equation

The result must be stable under controlled numerical changes:

```text
Y(dt, dx, particles, backend, restart) -> same physical answer within tolerance
```

Required checks:

```text
time-step convergence
mesh convergence
particle-count convergence
backend parity
restart reproducibility
segmented long-run reproducibility
limiter-zero proof
energy residual closure
```

Current state:

```text
numerical_fidelity_packet = missing
```

## General Solving Workflow

The DPF version of the word-problem workflow is:

```text
1. Read the source-truth statement.
2. Extract givens, unknowns, units, and constraints.
3. Define variables and state vectors.
4. Select the model family for the next unsolved relationship.
5. Translate source language into equations or operators.
6. Implement the operator without promoting it beyond its evidence.
7. Run a narrow numerical test.
8. Run a same-path simulation artifact.
9. Check physics residuals, units, sign, time-centering, and conservation.
10. Reject or retain the relationship.
11. Repeat until no blocker remains.
12. Submit the complete packet for engineering review.
```

This workflow intentionally prevents the failure mode:

```text
published endpoint -> guessed parameters -> plausible-looking simulation
```

The accepted workflow is:

```text
source equations -> implemented operators -> independent residual checks
  -> same-scope observable checks -> numerical fidelity -> engineering review
```

## Verification Workflow

Each solved subproblem must pass these checks before it can leave candidate
status.

| Check | What it catches |
|---|---|
| Unit check | Coefficients or source formulas used in the wrong unit system. |
| Sign check | Reversed power flow, back-EMF, or active-load convention. |
| Domain check | Integrating `J.E` over numerical floor cells, electrodes, or omitted plasma regions. |
| Time-centering check | Energy mismatch from mixing beginning-step, midpoint, and end-step quantities. |
| Conservation check | Hidden energy gain/loss not accounted for by fields, particles, thermal energy, radiation, or circuit work. |
| Negative test | A deliberately wrong sign, domain, or closure must fail. |
| Convergence check | The answer must not depend on one grid or timestep. |
| Restart check | Segmented and uninterrupted runs must agree. |
| Same-scope check | Simulated observables must be compared to the correct shot/family and diagnostics. |
| Review check | The packet must say exactly what is accepted, candidate, blocked, or not claimed. |

## Current Answer

The current problem can be summarized as:

```text
Given:
  K = partial PF-1000 machine inputs
  S_current = executable 3-D experimental simulator
  R_current = candidate residual and blocker telemetry

Find:
  X = missing source-backed physics, numerical, and review relationships

Such that:
  S_full = S_current + X
  S_full runs a whole DPF shot
  S_full uses first-principles-only authority
  S_full passes physics residuals, numerical residuals, and engineering review
```

Current result:

```text
S_current runs.
X is incomplete.
S_full does not exist yet.
```

Therefore:

```text
The problem is underdetermined.
```

The next solve-ready subproblem is:

```text
Given:
  I(t), Q(t), E(x,t), B(x,t), J(x,t), retained history, and energy ledgers

Find:
  U_DPF(t)

Subject to:
  d(L0 I)/dt = V0 - R0 I - Q/C0 - U_DPF
  U_DPF I = accepted electromagnetic terminal work
  Poynting, J.E, electrode work, and circuit work close within tolerance
  sign convention negative tests pass
  time-centering negative tests pass
  segmented 12 us run is finite and reproducible
```

If that subproblem is solved, the next unknowns become:

```text
startup_BVP
reviewed_geometry_masks
accepted_plasma_closures
neutron_mechanism_model
same_scope_observable_targets
numerical_fidelity_packet
engineering_certificate_packet
```

## Final Statement

The simulator is not a blank page. It is an executable experimental candidate.
But the algebraic word problem is still missing enough independent equations
and constraints that a unique accepted first-principles whole-shot solution
cannot yet be claimed.

The right next move is to keep converting each missing piece of `T` into a
source-backed equation, implementation, residual test, and review packet. The
first piece to solve is the power-port equation, because without accepted
`U_DPF(t)`, the circuit-plasma coupling remains the dominant underdetermined
relationship.
