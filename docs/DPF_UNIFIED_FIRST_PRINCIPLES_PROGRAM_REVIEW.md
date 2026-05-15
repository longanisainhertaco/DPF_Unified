# DPF-Unified First-Principles Program Review

Date: 2026-05-15

This review organizes the whole DPF-Unified program from a first-principles
simulation perspective. It does not use reduced-model success as authority. A
true first-principles path must advance resolved physical state variables,
conserve the right quantities, expose closures and approximations, and compare
only against source-backed same-scope observables.

The companion source index is:

- `docs/FIRST_PRINCIPLES_SOURCE_TRUTH_INDEX.md`
- `docs/FIRST_PRINCIPLES_SOURCE_TRUTH_INDEX.json`

## First-Principles Target

The target simulator is a full 3D dense plasma focus tool with this governing
shape:

1. Source-backed input deck: device geometry, fill gas, circuit, material,
   initial ionization, boundary conditions, and diagnostics.
2. Startup boundary-value problem: breakdown, flashover, sheath liftoff,
   current density, electron/ion temperatures, species, and fields.
3. Fully electromagnetic field solve: Maxwell evolution in plasma and vacuum,
   with constrained divergence control.
4. Kinetic ion advance: PIC particles for ions, current and charge deposition,
   collisions, boundaries, and source-ordered time integration.
5. Fluid electron closure: generalized Ohm law, electron pressure, Hall and
   resistive terms, and separate electron energy where those terms are active.
6. Circuit-field power port: plasma feedback through electromagnetic power
   accounting, not fitted Lee/RADPF inductance authority.
7. Physics closures: EOS, ionization, transport, radiation, material/electrode
   interaction, collisions, and neutron mechanisms.
8. Diagnostics and UQ: current, voltage, density, fields, temperatures, neutron
   mechanisms, detector response, conservation ledgers, uncertainty, and run
   manifests.

## Current Program Organization

### Public Entrypoints

| Surface | Current role | First-principles assessment |
| --- | --- | --- |
| `src/dpf/cli/main.py` | Main `dpf` CLI; routes `simulate`; contains `first-principles-3d` package-native JSON deck glue and the `hybrid-3d-smoke` engineering path. | Useful orchestration layer. The package-native 3D command now exercises the candidate 3D hybrid loop from `src/dpf/fields`, while the older PF-1000/Akel command remains app-backed. |
| `app_mhd.py` | PF-1000/Akel engineering runner, limiter telemetry, first-principles candidate metadata. | Important current probe, but should not remain the scientific execution authority because it is repo-root app code. |
| `src/dpf/server/*` | Async simulation lifecycle, API readiness payloads, field snapshots. | Product/API layer. It should consume the same package-native first-principles runner as CLI, not a separate path. |
| `frontendv2/`, `gui/`, top-level apps | Visualization and interaction surfaces. | Useful for inspection, but no UI path should create scientific authority. |

Concrete current entrypoint split:

- `dpf simulate` loads `SimulationConfig` and runs `SimulationEngine`.
- `dpf first-principles` dynamically loads `app_mhd.run_pf1000_akel_first_principles`.
- server/API simulations create `SimulationManager`, which wraps `SimulationEngine`.
- `dpf first-principles-3d` loads a JSON deck or built-in minimal engineering
  deck and directly exercises the package-native candidate
  `HybridPIC3DSimulator`, writing a compact JSON artifact.
- `dpf hybrid-3d-smoke` remains a direct smoke wrapper around the same candidate
  3D simulator path.

Those are four useful surfaces, but they are not yet one first-principles
execution contract.

### Core State And Execution

| Subsystem | Main files | Current role | First-principles assessment |
| --- | --- | --- | --- |
| Configuration | `src/dpf/config.py` | Pydantic simulation inputs for circuit, geometry, fluid, radiation, boundaries, diagnostics. | Good base, but the target 3D first-principles deck needs explicit startup, source-truth, closure, diagnostic, and acceptance-policy fields. |
| Engine loop | `src/dpf/engine/core.py` | Couples circuit, selected backend, advanced physics modules, diagnostics, and summaries. | General multiphysics engine exists. It is not yet the single target 3D hybrid PIC-fluid first-principles loop. |
| Shared contracts | `src/dpf/core/bases.py` | `StepResult`, `CouplingState`, solver interfaces. | Useful but too MHD/circuit-lumped for the target; 3D field/particle/electron/circuit state contracts should become first-class. |
| Circuit | `src/dpf/circuit/rlc_solver.py`, `src/dpf/circuit/coupler.py` | Lumped RLC and density-weighted MHD feedback scaffold. | RLC base exists. The density-weighted feedback scaffold is not enough for first-principles authority; the target needs a resolved field power port. |
| Fluid/MHD | `src/dpf/fluid/*`, `src/dpf/metal/*`, Athena wrappers | MHD solvers, cylindrical solver, GPU/MLX paths, transport and diffusion support. | Valuable for MHD intervals and numerical methods. Must be bounded or coupled into the hybrid 3D path where MHD assumptions fail. |
| 3D fields/hybrid | `src/dpf/fields/*` | Candidate Maxwell grid, PIC current port, generalized Ohm solver, predictor-corrector, Marder correction, conductivity blend, particle boundaries, electron energy, kinetic yield, circuit boundary, source geometry, hybrid loop. | This is the right first-principles direction. It is still engineering-candidate code and not yet the production run authority. |
| Kinetic/PIC | `src/dpf/experimental/pic/hybrid.py`, `src/dpf/kinetic/*` | Hybrid PIC particle support and kinetic manager. | Needed for the target. The active target path should move out of experimental status or wrap it behind a stable first-principles interface. |
| Atomic/transport/radiation | `src/dpf/atomic/*`, `src/dpf/collision/*`, `src/dpf/radiation/*`, `src/dpf/fluid/two_temperature.py` | Ionization, collisions, radiation, transport, and two-temperature pieces. | Required closure ingredients exist in pieces. They need source-scoped validity regimes, units, sensitivity/UQ, and wiring into the target 3D loop. |
| Diagnostics | `src/dpf/diagnostics/*` | Energy, HDF5, yield, beam, interferometry, x-ray, neutron TOF, evidence manifest, derived fields. | Good surface area. Needs a target 3D diagnostic contract tied to field/particle histories and detector response. |
| Evidence/release | `src/dpf/validation/*`, `src/dpf/release/*`, `docs/*` | Tests, readiness, requirements, source intake, certificates, release guardrails. | Useful for fail-closed outputs, but first-principles development should be driven by equations, closure validity, and source-index traceability rather than document labels alone. |

## Source Spine From The Corpus

The source index shows that the project already has enough material for a
credible first-principles architecture, but not enough for a complete validated
3D DPF machine claim.

| Source group | Best local material | Use in the simulator |
| --- | --- | --- |
| Hybrid electromagnetic PIC-fluid DPF | `KnowledgeReference/fully-electromagnetic-hybrid-pic-fluid-dpf-neutron-yield-acb71fa9.md` and matching `/Users/anthonyzamora/Downloads/2604.09032v1.pdf` | Primary target architecture: kinetic ions, fluid electrons, Maxwell fields in plasma/vacuum, generalized Ohm terms, current predictor-corrector, D-D yield history. |
| Kinetic DPF / beam-target physics | `KnowledgeReference/fully-kinetic-simulations-of-dense-plasma-focus-z-pinch.md`, `KnowledgeReference/comparisons-of-dense-plasma-focus-kinetic-simulations-with-experimental-measurements.md` | Fast-ion distributions, beam-target neutron mechanism, benchmark expectations beyond MHD. |
| Circuit and electromagnetic power | `KnowledgeReference/auluck-2021-dpf-circuit-element.md`, `KnowledgeReference/beresnyak_2022_pulsed_power_ideal_mhd.md` | Poynting/J.E power-port formulation and generator/plasma coupling constraints. |
| MHD and DPF context | ALEGRA, Beresnyak, Esaulov, Stepniewski, HAWK/PF-1000 material | MHD intervals, transport closure context, geometry/boundary requirements, and limitations of MHD-only claims. |
| Neutron and detector physics | `KnowledgeReference/bosch-hale-1992-fusion-reactivity.md`, detector/TOF/tomography/UQ sources | Thermonuclear reactivity, detector response, uncertainty, and mechanism-separated comparison. |

The largest source-backed gaps are startup/electrodes, exact 3D circuit-field
boundary closure, separate electron energy in the hybrid loop, radiation and
material interaction, and same-scope detector/UQ integration.

## Current Workflows

### General Simulation Path

The ordinary package path is:

`config file -> SimulationConfig -> SimulationEngine -> RLCSolver -> selected backend -> diagnostics -> summary/HDF5`

This path supports many physics modules and backends, but it is still centered
on a circuit plus fluid/MHD engine. It does not yet run the target 3D
electromagnetic hybrid PIC-fluid loop as the production first-principles
simulation.

The current package engine step order is approximately:

`compute dt -> ionization/resistivity -> collision/radiation half step -> circuit subcycle -> PIC/kinetic step -> fluid/MHD advance -> post-fluid corrections -> diagnostics/yield -> record/checkpoint`

That order is reasonable for the existing MHD-oriented engine, but the target
3D first-principles loop must put field/particle/current/electron/circuit state
on equal footing instead of treating PIC and circuit feedback as side modules.

### PF-1000/Akel Engineering Probe

The PF-1000/Akel probe is still substantially app-backed:

`CLI/app request -> app_mhd.py runner -> MHD candidate -> limiter/readiness metadata -> result summary`

This is useful for engineering and blocker discovery. It should be demoted to a
caller of package-native first-principles code, not kept as the authoritative
scientific path.

The app-backed branch currently exposes richer first-principles telemetry than
the general `StepResult` path, including field-power-port arrays, limiter
ledgers, readiness metadata, and neutron mechanism summaries. Those channels
need to move into the package-native manifest/diagnostic contract.

### Package-Native 3D Hybrid PIC-Fluid Path

The package-native 3D engineering path is:

`dpf first-principles-3d [--deck deck.json] -> normalized deck -> Maxwell3DGrid -> HybridPIC -> HybridPIC3DLoop -> HybridPIC3DSimulator -> FirstPrinciplesRunManifest -> JSON artifact`

This is the closest path to the target first-principles architecture. It
already exercises field-particle-current coupling, generalized Ohm components,
electron energy telemetry, kinetic yield telemetry, and candidate circuit
boundary coupling. It is intentionally still a small engineering smoke, not a
full DPF-machine production simulator. It is explicitly outside the older
validation workflow and remains `not_validation` until reviewed source-backed
acceptance artifacts are attached.

## Capability Matrix

| First-principles capability | Existing material | Main code surface | Status | What is still required |
| --- | --- | --- | --- | --- |
| Full Maxwell fields in plasma/vacuum | Strong source support from the 2604.09032v1 hybrid PIC-fluid paper and numerical EM sources. | `src/dpf/fields/maxwell_3d.py` | Candidate | Production integration, convergence tests, divergence control budget, boundary validation. |
| Kinetic ion PIC advance and deposition | Strong source support from hybrid PIC-fluid and particle simulation sources. | `src/dpf/experimental/pic/hybrid.py`, `src/dpf/fields/pic_coupling.py`, `src/dpf/fields/hybrid_loop.py` | Candidate | Stable public interface, source-ordered long-run tests, deposition conservation, collision/operator validation. |
| Fluid electron generalized Ohm law | Strong source support from 2604.09032v1 and plasma formulary material. | `src/dpf/fields/ohm_solver.py`, `src/dpf/fields/predictor_corrector.py` | Candidate | Accepted pressure/Hall/resistive term authority, Te coupling, numerical stability and energy consistency. |
| Circuit-field power port | Source support from Auluck circuit-element material and pulsed-power MHD sources. | `src/dpf/circuit/*`, `src/dpf/fields/circuit_boundary.py`, `app_mhd.py` | Blocked/candidate | Replace placeholder or scaffold feedback with resolved `U_DPF`/Poynting or `J.E` closure tied to geometry and energy residuals. |
| Startup/breakdown BVP | Source material exists but not enough implementation authority. | `app_mhd.py`, `src/dpf/experimental/civ_breakdown.py`, config scaffolds | Blocked | Source-backed breakdown/flashover/liftoff initial state generator with units and boundary checks. |
| Electrode, conductor, vacuum, and particle boundaries | Partial source support and candidate boundary code. | `src/dpf/fields/particle_boundaries.py`, `src/dpf/fields/source_geometry.py`, MHD boundary modules | Candidate/blocker | True machine geometry, injection port, PML/conductor semantics, electrode work, material interaction. |
| Plasma-vacuum conductivity blending | Source support from hybrid EM/vacuum treatment and transport sources. | `src/dpf/fields/conductivity.py` | Candidate | Weak-activity tests, sensitivity, and proof that blending is not creating the physical result. |
| Separate electron energy | Source support from hybrid/fluid and two-temperature sources. | `src/dpf/fields/electron_energy.py`, `src/dpf/fluid/two_temperature.py` | Candidate/blocker | Coupled Te equation with pressure/Hall authority, heat flux, collisional coupling, radiation, UQ. |
| Ion collisions and stopping | Source support from formulary, collision, and PIC sources. | `src/dpf/collision/*`, `src/dpf/fields/hybrid_loop.py`, `src/dpf/experimental/pic/hybrid.py` | Partial | Target-regime collision operator, timestep coupling, beam-target stopping validation. |
| Thermonuclear neutron history | Source support from D-D reactivity and DPF neutron papers. | `src/dpf/diagnostics/neutron_yield.py`, `src/dpf/fields/kinetic_yield.py` | Candidate | Field-history integral with uncertainty, same-scope target comparison, detector response. |
| Beam-target neutron mechanism | Source support from neutron dynamics and anisotropy sources. | `src/dpf/diagnostics/beam_target.py`, `src/dpf/fields/kinetic_yield.py` | Blocked/candidate | Kinetic ion distributions, target stopping, angular/spectral response, mechanism separation. |
| Diagnostics and UQ | Broad code surface exists. | `src/dpf/diagnostics/*`, `src/dpf/validation/*` | Partial | One output schema for current, fields, temperatures, density, neutron mechanisms, detector response, uncertainty, and conservation ledgers. |
| Same-scope validation | Source corpus contains many possible targets but not a complete accepted 3D packet. | Source index, `src/dpf/validation/*`, docs | Blocked | One machine/shot packet with all required comparable observables and uncertainty. |
| Production scaling | CPU/GPU/backend options exist. | Python, Athena/AthenaK, Metal/MLX, server/CLI | Partial | Target 3D runtime model, memory budget, restart, backend authority, reproducible manifests. |

## Main Blockers

1. **No single production first-principles runner yet.** The new
   `dpf first-principles-3d` command provides package-native CLI/deck glue for
   the candidate 3D loop, but the general engine, app-backed PF-1000 probe, and
   3D hybrid path are still not one coherent production execution contract.
2. **Startup is not a solved first-principles boundary-value problem.** The
   code can seed or estimate early state, but accepted breakdown/liftoff
   initialization is still missing.
3. **The circuit-field power port is not closed.** Existing RLC, density-weighted
   coupling, and candidate magnetic boundary pieces are not yet a resolved
   `U_DPF`/Poynting/J.E closure with geometry, sign, time-centering, and energy
   residual authority.
4. **The target 3D hybrid loop is candidate-only.** The right components exist
   under `src/dpf/fields`, but they need long-run integration, production
   manifests, numerical tests, and conservation diagnostics.
5. **Electron-energy and neutron authority remain incomplete.** Hall/pressure
   terms and neutron yield cannot be authoritative without separate Te,
   mechanism-separated yield history, detector response, and UQ.
6. **Same-scope validation is still the scientific bottleneck.** The corpus has
   strong method material, but a full 3D machine packet with all comparable
   observables is not yet assembled.
7. **Reduced and surrogate paths are still mixed into the product surface.**
   Lee, snowplow, scaling-law, and AI/surrogate modules are useful baselines or
   acceleration aids, but they must stay out of predictive authority for the
   first-principles claim.

## Recommended Organization

1. Make `src/dpf/fields` the nucleus of the true 3D first-principles solver.
   Keep the Maxwell grid, PIC coupling, Ohm solve, predictor-corrector, Marder
   correction, conductivity blend, electron energy, kinetic yield, and circuit
   boundary in one explicit target loop.
2. Introduce a package-native first-principles run contract:
   `FirstPrinciplesInputDeck -> FirstPrinciplesRun -> FirstPrinciplesManifest`.
   The CLI, server, app, and tests should call that same contract.
3. Keep `SimulationEngine` as the general multiphysics engine until the target
   loop replaces or wraps it. Do not force the 3D hybrid path through the old
   MHD-only state shape if that hides field/particle/electron/circuit state.
4. Split code authority labels:
   - `first_principles_core`: resolved field/particle/electron/circuit path.
   - `mhd_interval`: bounded MHD-only physics.
   - `baseline_model`: Lee, snowplow, scaling laws.
   - `preview_backend`: MLX/Metal/surrogate until proven for a specific scope.
   - `diagnostic_or_ui`: output, visualization, and inspection only.
5. Wire the source index into development:
   every new closure, boundary, or diagnostic module should declare source-index
   IDs and the capability tags it claims to implement.
6. Treat the final simulator as a conservation-ledger machine:
   every step should emit mass, charge, field energy, particle energy, electron
   energy, circuit energy, radiation, boundary work, and residuals.

## Finish-Line Execution Order

1. Extend the new package-native first-principles runner around the existing
   3D hybrid loop from engineering candidate into the whole-shot production
   contract.
2. Add the source-backed input deck and startup BVP contract.
3. Implement the resolved circuit-field power port and remove placeholder
   `U_DPF` behavior from acceptance paths.
4. Promote candidate field/PIC/Ohm/current/boundary/electron-energy/yield
   pieces by adding conservation tests and long-run telemetry.
5. Add mechanism-separated neutron and detector/UQ output schemas.
6. Assemble one same-scope 3D machine validation packet from the source corpus
   or explicitly choose a better-diagnosed first demonstration scope.
7. Keep reduced, app-only, and surrogate paths as comparison layers until they
   can prove their own bounded authority.
