# First-Principles DPF Simulator Status And Blockers

Date: 2026-05-18

Scope: package-native, first-principles-only DPF simulator path in
`dpf-unified`, with PF-1000/Akel as the current primary demonstrator. This
document describes what the simulator does now, what it is intended to do, and
what still blocks a full first-principles whole-shot tool for engineering
review.

## What The Simulator Does Now

The current simulator is an experimental 3-D hybrid electromagnetic PIC/fluid
runtime. It runs a source-scoped PF-1000/Akel engineering deck through the
package-native first-principles path rather than through reduced Lee/snowplow
authority. It carries an RLC bank/circuit state, applies a magnetic injection
boundary from terminal current, advances a Cartesian 3-D Maxwell field state,
pushes PIC particles, deposits charge/current, applies generalized-Ohm current
and electron-energy candidate closures, tracks ionization state, and emits
fail-closed telemetry packets for conservation, power-port accounting,
dimensionality, closures, limiter activity, neutron authority, waveform
comparison, numerical fidelity, and certificate readiness.

The active long-horizon PF-1000/Akel artifact is:

- `results/experimental_limiter_proof_pf1000_seeded_power_domain_12us_2026_05_18.json`

That run reached `12.000182898446022 us` in `55580` vacuum-CFL steps. It
remained finite, satisfied the requested duration, and reported zero
acceptance-blocking limiter activations. The final terminal current was about
`1.200240 MA`, final circuit charge was `15.95985 C`, final field energy was
`41807.165 J`, and the final particle inventory was `222328`.

The simulator also now exposes an explicit candidate source-sign power-port
mode:

- `lagged_auluck_volume_j_dot_e`: uses the local source relation
  `U_DPF = - integral(J.E)dV / I`
- `lagged_volume_j_dot_e`: conservative existing mode that blocks negative
  `J.E` active-port feedback and falls back to input `U_DPF`

The source-sign smoke artifact is:

- `results/experimental_limiter_proof_pf1000_auluck_power_port_100ns_2026_05_18.json`

That run reached `100.18144773081963 ns` in `464` vacuum-CFL steps, remained
finite, and recorded `active_power = -J.E` explicitly. A direct `12 us`
source-sign comparison was attempted, but it did not produce an artifact within
the practical interactive runtime window and was stopped.

The current simulator is therefore useful for experimental engineering
inspection and blocker reduction. It is not yet an accepted first-principles
DPF shot simulator.

## What We Want It To Do

The goal is a full 3-D first-principles DPF shot simulator that can run an
entire machine discharge from startup through rundown, pinch, post-pinch, and
neutron/diagnostic output without reduced-model authority.

The target tool should:

- Simulate whole-shot startup: gas breakdown, preionization, insulator
  flashover, secondary emission, surface plasma formation, sheath liftoff, and
  transition into rundown.
- Use reviewed machine geometry: PF-1000 rods, hollow anode, insulator
  material surfaces, cathode cage, electrodes, bore, and field/particle
  boundary masks.
- Couple the pulsed-power circuit to the plasma through an accepted
  first-principles power port: terminal current, terminal voltage, named
  Poynting or volume `J.E` domain, electrode-work partition, sign convention,
  time-centering, and residual budget.
- Advance 3-D electromagnetic fields, particles, charge/current, electron and
  ion energy, ionization, radiation, ablation/impurities, collisions/stopping,
  and resistivity from source-backed physics closures.
- Resolve mechanism-separated neutron production: thermonuclear, beam-target,
  beam formation/transport, spectrum, anisotropy, and detector-response
  packets.
- Produce same-scope comparison artifacts: current waveform, phase timing,
  field/density/temperature histories, neutron diagnostics, uncertainty
  quantification, convergence evidence, restart reproducibility, and backend
  parity.
- Emit an engineering review packet that is honest about every candidate
  closure and every acceptance blocker.

## Main Workflow

Current experimental workflow:

```text
source-truth deck
  -> PF-1000/Akel package input deck
  -> startup seeded-layer candidate state
  -> RLC circuit state
  -> magnetic injection boundary
  -> 3-D Maxwell field step
  -> PIC particle push and deposition
  -> generalized-Ohm current domain guard
  -> electron-energy / ionization candidate closures
  -> circuit power-port telemetry
  -> conservation and limiter inventory
  -> whole-shot, fidelity, source, comparator, neutron, certificate packets
  -> JSON artifact for engineering inspection
```

Desired accepted workflow:

```text
reviewed source-truth machine packet
  -> reviewed geometry/material/boundary masks
  -> first-principles startup BVP
  -> time-centered electromagnetic power port
  -> converged 3-D EM/PIC/fluid/kinetic shot
  -> mechanism-separated diagnostics
  -> same-scope comparison and UQ
  -> independent engineering review certificate
```

## Blockers

### 1. Startup BVP

Status: blocked/rejected for first-principles acceptance.

The simulator starts from a seeded engineering layer, not a solved
first-principles startup state. Breakdown, preionization, surface flashover,
secondary emission, sheath liftoff, and the handoff into rundown are still not
accepted as a source-backed BVP.

### 2. Power Port

Status: most urgent physics blocker.

The 12 us fallback artifact completed, but it did so with
`input_sequence_fallback_negative_j_dot_e_active_port_blocked=54503`, meaning
the simulator repeatedly blocked negative `J.E` active-port behavior rather
than accepting a bidirectional terminal-work closure.

The source-sign `U_DPF = - integral(J.E)dV / I` branch now exists and runs to
`100 ns`, but it is still candidate-only. It needs segmented long-run support,
explicit terminal/control-volume domain review, sign review, time-centering,
electrode-work partition, residual tolerance, and negative tests before it can
replace the fail-closed fallback.

### 3. Reviewed 3-D Machine Geometry

Status: candidate.

PF-1000 geometry is represented enough for engineering probes, but rods,
hollow-anode bore, insulator surfaces, material boundaries, and particle/field
masks still need reviewed same-scope geometry packets. The current grid is
coarse and not predictive fidelity.

### 4. Physics Closures

Status: candidate/scaffolded.

Electron energy, two-temperature coupling, conductivity, ionization, heat
flux, radiation, ablation/impurities, anomalous resistance, restrike,
PIC collisions, and stopping are not all accepted as a complete source-backed
closure set. PlasmaPy is useful as an optional community formulary cross-check,
but it does not replace the local source-truth acceptance requirement.

### 5. Neutron Authority

Status: blocked.

The simulator does not yet provide accepted mechanism-separated neutron
authority: thermonuclear versus beam-target production, beam formation and
transport, spectra, anisotropy, and detector-response mapping.

### 6. Same-Scope Data Binding

Status: blocked.

The current packets do not yet bind accepted PF-1000 current waveform, phase
timing, spatial field, density, temperature, and neutron histories in the same
scope as the simulated shot. Without that, engineering comparison remains
incomplete.

### 7. Numerical Fidelity

Status: blocked.

The simulator needs convergence studies, backend parity, limiter-zero proof,
restart reproducibility, segmented long-run telemetry, mesh/time-step
sensitivity, and acceptance of the source-ordered predictor-corrector path.

### 8. Certificate / Engineering Review Packet

Status: blocked.

The current outputs are explicitly `engineering_candidate_not_validation` and
`not_validation`. A real first-principles release needs all required packets
linked into a certificate gate that an engineering team can inspect and
challenge.

## Current Bottom Line

We can run experimental PF-1000/Akel first-principles-path shots to microsecond
duration on a coarse 3-D engineering deck. The 12 us fallback run is finite and
useful. We cannot yet claim a complete first-principles whole-shot DPF
simulation because startup, power-port authority, reviewed geometry, closure
acceptance, neutron authority, same-scope comparison, numerical fidelity, and
certificate review remain open.

The next highest-leverage work is the power port: segment and checkpoint the
source-sign `lagged_auluck_volume_j_dot_e` branch, rerun it toward `12 us`,
then compare its circuit/energy ledger against the conservative fallback
artifact with explicit residuals.
