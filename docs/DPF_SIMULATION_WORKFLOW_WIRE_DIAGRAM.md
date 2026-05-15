# DPF Simulation Workflow Wire Diagram

Date: 2026-05-15

This document explains the DPF-Unified simulation wiring from a first-principles
perspective. It separates the current program flow from the target full 3D
first-principles flow.

## Current Program Flow

```mermaid
flowchart TD
    A[CLI dpf simulate] --> B[SimulationConfig]
    A2[Server/API SimulationManager] --> B
    A3[App/UI general run] --> B
    B --> C[SimulationEngine]
    C --> D[RLCSolver]
    C --> E{Backend selector}
    E --> F[Python fluid/MHD solver]
    E --> G[Athena/AthenaK wrappers]
    E --> H[Metal/MLX preview solvers]
    E --> I[Hybrid Athena/WALRUS path]
    D --> J[CouplingState: I, V, Lp, dL/dt, R_plasma]
    F --> J
    G --> J
    H --> J
    I --> J
    C --> K[Atomic, collision, radiation, sheath, turbulence modules]
    C --> L[KineticManager / PIC side path]
    C --> M[Diagnostics: energy, yield, HDF5, derived fields]
    M --> N[StepResult, HDF5, API snapshots, summaries]
```

The current ordinary path is a circuit plus MHD-oriented engine. It can run
useful physics probes, but it is not yet the complete 3D hybrid PIC-fluid
first-principles simulator.

## Current App-Backed First-Principles Branch

```mermaid
flowchart TD
    A[dpf first-principles or app first_principles_mhd] --> B[Dynamic load app_mhd.py]
    B --> C[run_pf1000_akel_first_principles]
    C --> D[run_mhd_simulation]
    D --> E[_run_python_mhd field_coupled_candidate]
    E --> F[Field-power-port arrays]
    E --> G[Limiter ledger]
    E --> H[Neutron mechanism summary]
    E --> I[Readiness/result metadata]
    F --> J[Compact JSON artifact or app result]
    G --> J
    H --> J
    I --> J
```

This branch currently carries richer first-principles telemetry than the
ordinary `StepResult` path. The target architecture should move those channels
into the package-native first-principles manifest rather than leaving them in a
repo-root app.

## Current Package-Native 3D Engineering Flow

```mermaid
flowchart TD
    A[dpf first-principles-3d] --> A1[Built-in or JSON input deck]
    A1 --> B[HybridPICSourceGeometry smoke grid]
    A2[dpf hybrid-3d-smoke compatibility smoke] --> B
    B --> C[Maxwell3DGrid and Maxwell3DState]
    B --> D[HybridPIC ion particles]
    C --> E[HybridPIC3DLoop]
    D --> E
    E --> F[Deposit ion charge/current]
    F --> G[GeneralizedOhmSolver]
    G --> H[CurrentPredictorCorrector]
    H --> I[Maxwell3DFieldCore]
    I --> J[Marder/Gauss control]
    J --> K[Particle push and boundaries]
    K --> L[ElectronEnergyClosure telemetry]
    K --> M[KineticIonYieldHistory telemetry]
    L --> N[HybridPIC3DSimulator result]
    M --> N
    N --> O[FirstPrinciplesRunManifest]
    O --> P[JSON artifact: not_validation engineering candidate]
```

This path has the right architectural spine, but it is still a compact
engineering candidate. It does not yet carry production geometry, startup,
resolved circuit power closure, validated electron energy, or same-scope
diagnostic targets.

`dpf first-principles-3d --deck deck.json --output run.json` is the current
package-native CLI glue for this path. Without `--deck`, it runs a minimal
built-in engineering deck; without `--output`, it prints the JSON artifact to
stdout. The older `dpf first-principles` PF-1000/Akel command remains unchanged
and app-backed.

The package-native command now routes through
`src/dpf/first_principles/runner.py`, not the older validation workflow. Its
artifact includes a source-index manifest, conservation telemetry, candidate
evidence keys, and explicit `can_support_first_principles_acceptance = false`
until a real engineer-reviewed source-backed acceptance packet exists.

## Target Full 3D First-Principles Flow

```mermaid
flowchart TD
    A[Source-truth input deck] --> B[Device geometry and materials]
    A --> C[Circuit bank and switches]
    A --> D[Fill gas, species, pressure]
    A --> E[Diagnostic target definitions]
    B --> F[Startup boundary-value problem]
    C --> F
    D --> F
    F --> G[Initial 3D fields, particles, electron state, circuit state]
    G --> H[Source-ordered timestep loop]
    H --> I[Ion PIC push and charge/current deposition]
    I --> J[Electron fluid: generalized Ohm law]
    J --> K[Separate electron energy and pressure]
    K --> L[Full Maxwell update in plasma and vacuum]
    L --> M[Divergence/Gauss and boundary control]
    M --> N[Resolved circuit-field power port]
    N --> O[Closures: collisions, ionization, EOS, radiation, material interaction]
    O --> P[Mechanism-separated neutron production]
    P --> Q[Synthetic diagnostics and detector response]
    Q --> R[Conservation, UQ, and manifest]
    R --> S[Comparison to same-scope source targets]
```

The target path removes reduced-model authority from the predictive chain. Lee,
snowplow, scaling-law, and surrogate outputs may remain comparison or preview
layers, but they cannot drive the accepted first-principles result.

## Target Timestep Loop

```mermaid
flowchart LR
    A[State n: E, B, particles, ne, Te, circuit] --> B[Advance ion positions/velocities]
    B --> C[Apply conductor/PML/particle boundaries]
    C --> D[Deposit charge and ion current]
    D --> E[Rebuild quasi-neutral electron density]
    E --> F[Solve generalized Ohm law]
    F --> G[Predict/correct total current]
    G --> H[Advance Maxwell fields]
    H --> I[Apply Gauss/divergence control]
    I --> J[Update electron energy and closures]
    J --> K[Update circuit through Poynting or J.E power port]
    K --> L[Accumulate diagnostics, UQ, conservation residuals]
    L --> M[State n+1]
```

Current `SimulationEngine.step()` order for the general path is different:

```mermaid
flowchart LR
    A[Compute dt] --> B[Ionization, resistivity, R_plasma, L_plasma]
    B --> C[Collision/radiation half step]
    C --> D[Circuit subcycle]
    D --> E[Kinetic/PIC side step]
    E --> F[Fluid/MHD advance]
    F --> G[Post-fluid corrections]
    G --> H[Diagnostics and neutron yield]
    H --> I[Record, checkpoint, StepResult]
```

The general path is useful, but the target loop must make field, particle,
electron, and circuit state co-equal rather than side-loading kinetic and
field-power behavior around a fluid-centered step.

Required residual ledgers per step:

- charge conservation and Gauss-law residual;
- magnetic divergence residual;
- field energy, particle kinetic energy, electron internal energy, circuit
  energy, radiation loss, boundary work, and total residual;
- current deposition and predictor-corrector residual;
- limiter/floor/cap/repair activation count;
- neutron mechanism increments and detector-response increments.

## Source-Truth To Code Traceability

```mermaid
flowchart TD
    A[KnowledgeReference plus 2604.09032v1 PDF] --> B[Source truth index JSON]
    B --> C[Capability tags: Maxwell, PIC, Ohm, circuit, startup, neutron, UQ]
    C --> D[Implementation requirement]
    D --> E[Code module or closure packet]
    E --> F[Unit and conservation tests]
    F --> G[Run manifest]
    G --> H[Diagnostic/UQ packet]
    H --> I[Same-scope comparison]
    I --> J{Acceptance decision}
    J -->|pass| K[First-principles claim for declared scope]
    J -->|fail or missing| L[Engineering candidate only]
```

The index is not itself a validation certificate. It is the map that prevents
coding from drifting away from the source-truth corpus.

## First-Principles Source Spine

```mermaid
flowchart TD
    A[2604.09032v1 hybrid EM PIC-fluid DPF] --> B[Target 3D hybrid architecture]
    C[Auluck DPF circuit element] --> D[Resolved electromagnetic power port]
    E[ALEGRA, Beresnyak, Esaulov, Stepniewski] --> F[MHD, circuit, transport, and DPF context]
    G[Schmidt/LSP kinetic DPF papers] --> H[Kinetic ion and beam-target neutron physics]
    I[Bosch-Hale and DD reactivity sources] --> J[Thermonuclear neutron rate]
    K[Detector, TOF, tomography, UQ sources] --> L[Synthetic diagnostics and uncertainty]
    B --> M[Full 3D first-principles simulator]
    D --> M
    F --> M
    H --> M
    J --> M
    L --> M
```

The strongest corpus result is that we have a credible source spine for the
architecture. The weakest parts are not the high-level concept; they are
startup/electrodes, exact 3D circuit-field boundary closure, separate electron
energy, radiation/material coupling, beam-target detector response, and a
single same-scope validation packet.

## Reduced-Model Boundary

```mermaid
flowchart TD
    A[Lee/RADPF] --> B[Baseline comparison only]
    C[Snowplow] --> B
    D[Scaling laws] --> B
    E[AI/surrogate] --> F[Acceleration or preview only]
    B --> G[Can compare against first-principles output]
    F --> G
    G --> H[Cannot supply accepted predictive authority]
```

Reduced models remain valuable for regression, speed, intuition, and comparison.
They must not be used as closure factors inside the accepted first-principles
chain.

## Final Workflow Contract

The final simulator should expose one package-native contract:

```text
FirstPrinciplesInputDeck
  -> StartupBoundaryValueProblem
  -> HybridEMPicFluidRun
  -> FirstPrinciplesRunManifest
  -> DiagnosticAndUQPacket
  -> SameScopeComparisonPacket
```

All user surfaces should call this same contract:

```mermaid
flowchart LR
    A[CLI] --> D[Package-native first-principles runner]
    B[Server/API] --> D
    C[App/GUI] --> D
    D --> E[Run manifest and diagnostics]
    E --> F[Source-traceable review output]
```
