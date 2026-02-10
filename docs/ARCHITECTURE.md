# DPF Unified — Architecture & Data Flow

## System Architecture

```mermaid
graph TB
    subgraph Frontend["Frontend Layer"]
        CLI["CLI<br/>(Click)"]
        Server["Server<br/>(FastAPI)"]
        GUI["GUI<br/>(Unity)"]
        AI_API["AI REST/WS<br/>/api/ai/*"]
    end

    subgraph Core["Core Orchestration"]
        Engine["SimulationEngine<br/>(engine.py)<br/>Config -> Backend Select<br/>State dict management<br/>Strang splitting<br/>Circuit coupling"]
        AI_ML["AI/ML Layer<br/>surrogate.py (WALRUS 1.3B)<br/>inverse_design.py (Bayesian)<br/>hybrid_engine.py (handoff)<br/>confidence.py (ensemble)<br/>instability_detector.py<br/>realtime_server.py"]
    end

    subgraph Backends["Backend Selection<br/>auto-priority: athenak > athena > metal > python"]
        AthenaK["AthenaK<br/>Kokkos C++<br/>OpenMP/CUDA/HIP/SYCL<br/>VTK I/O"]
        Athena["Athena++<br/>pybind11 C++<br/>Circuit / Spitzer<br/>Braginskii / Bremss.<br/>Two-Temp / HDF5 I/O"]
        Metal["Metal GPU<br/>WENO5-Z + HLLD<br/>SSP-RK3 + CT<br/>Float32/64<br/>Resistive MHD<br/>PyTorch MPS"]
        Python["Python Engine<br/>WENO-Z + HLLD<br/>SSP-RK3<br/>Braginskii<br/>Powell + Dedner<br/>Numba / NumPy"]
    end

    Base["PlasmaSolverBase (bases.py)<br/>State dict: {rho, velocity, pressure, B, Te, Ti, psi}"]

    CLI --> Engine
    Server --> Engine
    GUI --> Engine
    AI_API --> AI_ML
    Engine --> AthenaK
    Engine --> Athena
    Engine --> Metal
    Engine --> Python
    AthenaK --> Base
    Athena --> Base
    Metal --> Base
    Python --> Base

    style Frontend fill:#e1f5fe,stroke:#0288d1
    style Core fill:#fff3e0,stroke:#f57c00
    style Backends fill:#e8f5e9,stroke:#388e3c
    style Base fill:#f3e5f5,stroke:#7b1fa2
```

**Fidelity Grade: 8.9/10** | 1,475 tests (1,353 non-slow + 122 slow) | 0 failures | Phases A-P complete

---

## Metal GPU Physics Stack (Phase P — 8.9/10)

```mermaid
graph TB
    subgraph MetalSolver["MetalMHDSolver (metal_solver.py)"]
        subgraph TimeInt["Time Integration"]
            RK3["SSP-RK3<br/>3-stage, O(3)<br/>Shu-Osher 1988"]
            RK2["SSP-RK2<br/>2-stage, O(2)<br/>Gottlieb et al."]
        end

        subgraph Recon["Spatial Reconstruction (metal_riemann.py)"]
            WENO["WENO5-Z<br/>5th-order<br/>Borges 2008<br/>FD point-value<br/>tau5 indicator"]
            PLM["PLM<br/>2nd-order<br/>minmod/MC limiter"]
        end

        subgraph Riemann["Riemann Solvers (metal_riemann.py)"]
            HLLD["HLLD<br/>8-component<br/>4 intermediate states<br/>Miyoshi & Kusano 2005<br/>NaN -> HLL fallback"]
            HLL["HLL<br/>2-wave<br/>fully vectorized"]
        end

        subgraph DivB["Divergence Control"]
            CT["Constrained Transport<br/>(metal_stencil.py)<br/>div(B) = 0 to machine prec."]
        end

        subgraph Resistive["Resistive MHD (Phase P)"]
            ResMHD["Operator-Split Diffusion<br/>Explicit with CFL sub-cycling<br/>dt_diff = dx^2 * mu0 / (2*eta)<br/>Spitzer / anomalous eta"]
        end

        Precision["Precision: float32 (MPS GPU) | float64 (CPU fallback)"]
    end

    TimeInt --> Recon --> Riemann --> DivB --> Resistive --> Precision

    style MetalSolver fill:#e8f5e9,stroke:#2e7d32
    style TimeInt fill:#c8e6c9,stroke:#388e3c
    style Recon fill:#c8e6c9,stroke:#388e3c
    style Riemann fill:#c8e6c9,stroke:#388e3c
    style DivB fill:#c8e6c9,stroke:#388e3c
    style Resistive fill:#a5d6a7,stroke:#1b5e20
```

---

## Python Engine Physics Stack (Phase P — 8.9/10)

```mermaid
graph TB
    subgraph PySolver["MHDSolver / CylindricalMHDSolver (mhd_solver.py)"]
        subgraph PyTime["Time Integration"]
            PyRK3["SSP-RK3 (default)<br/>3-stage, O(3)<br/>Shu-Osher 1988"]
            PyRK2["SSP-RK2<br/>2-stage, O(2)"]
        end

        subgraph PyRecon["Flux Computation"]
            PyWENO["WENO-Z Weights<br/>Borges 2008<br/>FV Jiang-Shu polynomials<br/>Interior cells [2, N-3]"]
            PyGrad["np.gradient<br/>2nd-order central<br/>All cells (fallback)"]
        end

        subgraph PyRiemann["Riemann Solvers"]
            PyHLLD["HLLD (default)<br/>Miyoshi & Kusano 2005<br/>Full 8-component"]
            PyHLL["HLL<br/>2-wave fallback"]
        end

        subgraph PyDivB["Divergence Control"]
            Powell["Powell Source Terms"]
            Dedner["Dedner Hyperbolic Cleaning"]
        end

        subgraph PyPhysics["DPF Physics"]
            Circuit["Circuit RLC Coupling"]
            Spitzer["Spitzer Resistivity"]
            Brag["Braginskii Transport"]
            TwoTemp["Two-Temperature e/i"]
            Bremss["Bremsstrahlung"]
            Nernst["Nernst Effect"]
        end

        VelClamp["Velocity Clamping<br/>10x fast magnetosonic<br/>(WENO5 boundary safety)"]
    end

    PyTime --> PyRecon --> PyRiemann --> PyDivB --> PyPhysics --> VelClamp

    style PySolver fill:#e3f2fd,stroke:#1565c0
    style PyTime fill:#bbdefb,stroke:#1976d2
    style PyRecon fill:#bbdefb,stroke:#1976d2
    style PyRiemann fill:#bbdefb,stroke:#1976d2
    style PyDivB fill:#bbdefb,stroke:#1976d2
    style PyPhysics fill:#90caf9,stroke:#0d47a1
```

> **Note**: Python WENO-Z is a *hybrid* scheme — WENO5 flux divergence updates density/momentum
> only in interior cells `[2, N-3]`, while other terms use `np.gradient` on all cells. This
> boundary mismatch limits stability for dynamic problems with sound waves. For full-fidelity
> WENO5+HLLD+SSP-RK3, use the **Metal engine** which has a fully conservative formulation.

---

## Iterative Accuracy Workflow

```mermaid
graph LR
    CREATE["CREATE<br/>Implement physics<br/>using reference lit."]
    TEST["TEST<br/>Unit + shock +<br/>conservation +<br/>convergence"]
    RATE["RATE<br/>Fidelity 1-10<br/>Sandia=8<br/>open-source=6-7"]
    RESEARCH["RESEARCH<br/>opus agents +<br/>literature +<br/>cost/benefit"]
    IMPROVE["IMPROVE<br/>Highest-impact fix<br/>correctness > accuracy<br/>> performance"]

    CREATE --> TEST --> RATE --> RESEARCH --> IMPROVE --> CREATE

    style CREATE fill:#c8e6c9,stroke:#2e7d32
    style TEST fill:#bbdefb,stroke:#1565c0
    style RATE fill:#fff9c4,stroke:#f9a825
    style RESEARCH fill:#f8bbd0,stroke:#c2185b
    style IMPROVE fill:#d1c4e9,stroke:#512da8
```

### Accuracy Milestones

```mermaid
graph LR
    M65["6.5/10<br/>PLM + HLL<br/>SSP-RK2<br/>float32 + CT"]
    M80["8.0/10<br/>WENO5 + HLLD<br/>SSP-RK3<br/>float32"]
    M87["8.7/10<br/>WENO5-Z + HLLD<br/>SSP-RK3 + float64<br/>CT + MC limiter"]
    M89["**8.9/10**<br/>+ Python WENO-Z<br/>+ SSP-RK3 + HLLD<br/>+ Metal resistive MHD"]
    M90["9.0/10<br/>+ characteristic<br/>decomposition"]
    M95["9.5/10<br/>+ AMR + higher-order<br/>CT + HPC scaling"]

    M65 -->|"Phase M/N"| M80 -->|"Phase O interim"| M87 -->|"Phase O"| M89 -->|"Phase P ★"| M90 -->|"Future"| M95

    style M89 fill:#a5d6a7,stroke:#1b5e20,stroke-width:3px
    style M95 fill:#e0e0e0,stroke:#9e9e9e
    style M90 fill:#e0e0e0,stroke:#9e9e9e
```

---

## AI/ML Data Pipeline

```mermaid
graph LR
    subgraph Training["Training Pipeline"]
        DPF["DPF Engine<br/>(full sim)<br/>min-hours"]
        Well["Well HDF5 Export<br/>float32 / cartesian<br/>[x, y, z] axis order"]
        FT["WALRUS 1.3B<br/>Fine-Tuning<br/>torch==2.5.1"]
    end

    subgraph Inference["Inference Pipeline (~100ms/step)"]
        State["DPF State History<br/>(tau steps)<br/>rho, B, Te, Ti, v, p"]
        RevIN["RevIN Normalize<br/>(RMS, sample-wise)"]
        Model["IsotropicModel<br/>Encoder-Processor-Decoder<br/>768-1408 hidden dims"]
        Delta["Delta Predict<br/>u(t+1) = u(t) + delta"]
        DeNorm["Denormalize<br/>+ Residual Connection"]
    end

    DPF --> Well --> FT
    State --> RevIN --> Model --> Delta --> DeNorm

    style Training fill:#fff3e0,stroke:#f57c00
    style Inference fill:#e1f5fe,stroke:#0288d1
```

### WALRUS Architecture Detail

```mermaid
graph TB
    subgraph WALRUS["IsotropicModel (1.3B params)"]
        Enc["Encoder<br/>SpaceBagAdaptiveDVstride<br/>Strided conv + modulation"]
        Proc["Processor<br/>12-40 SpaceTimeSplitBlocks<br/>Axial or full attention"]
        Dec["Decoder<br/>AdaptiveDVstride<br/>Transposed conv"]
    end

    subgraph Hardware["Apple Silicon (M3 Pro, 36GB)"]
        FP16["Float16 inference: ~2.6 GB"]
        MPS["MPS: 1.57x speedup"]
        MLX["MLX: zero-copy memory"]
    end

    Enc --> Proc --> Dec
    Dec --> Hardware

    style WALRUS fill:#f3e5f5,stroke:#7b1fa2
    style Hardware fill:#e8eaf6,stroke:#283593
```

---

## Verification & Validation Tiers

```mermaid
graph TB
    T5["Tier 5: Metal GPU + Engine Accuracy (68 tests)<br/>HLLD stability, WENO5-Z convergence, SSP-RK3,<br/>float64, resistive MHD, WENO-Z weights"]
    T4["Tier 4: WALRUS Surrogate<br/>Single-step, rollout, sweep, ensemble, hybrid fallback"]
    T3["Tier 3: System (DPF Device)<br/>Lee Model (PF-1000, NX2), scaling laws, energy balance"]
    T2["Tier 2: Integration<br/>Orszag-Tang, Sedov, MHD convergence, anisotropic cond."]
    T1["Tier 1: Unit<br/>Sod, Brio-Wu, Spitzer, bremsstrahlung, fusion, circuit"]

    T5 --> T4 --> T3 --> T2 --> T1

    style T5 fill:#c8e6c9,stroke:#1b5e20,stroke-width:2px
    style T4 fill:#dcedc8,stroke:#33691e
    style T3 fill:#fff9c4,stroke:#f9a825
    style T2 fill:#ffe0b2,stroke:#e65100
    style T1 fill:#ffccbc,stroke:#bf360c
```

---

## Backend Accuracy Comparison (Phase P)

| Feature | Python Engine | Athena++ | AthenaK | Metal GPU |
|---------|--------------|----------|---------|-----------|
| Reconstruction | WENO-Z (hybrid) | PLM/PPM | PLM/PPM | WENO5-Z (full) |
| Riemann Solver | **HLLD** (default) | HLLD/HLLC | HLLD/HLLC | **HLLD** + HLL fallback |
| Time Integration | **SSP-RK3** | VL2 | VL2 | **SSP-RK3** |
| Precision | float64 | float64 | float64 | float32/64 |
| div(B) Control | Powell + Dedner | CT | CT | CT |
| Resistive MHD | Spitzer (Strang split) | Full (C++) | Basic | **Operator-split + CFL sub-cycling** |
| DPF Physics | Full | Full (C++) | Basic | Hydro + MHD + Resistive |
| Circuit Coupling | Yes | Yes (C++) | No | No |
| Braginskii | Yes | Yes (C++) | No | No |

### Maximum Accuracy Configuration

```python
# Metal engine (recommended for highest fidelity)
MetalMHDSolver(
    reconstruction="weno5",      # 5th-order WENO-Z (Borges et al. 2008)
    riemann_solver="hlld",       # Miyoshi & Kusano (2005) 4-wave solver
    time_integrator="ssp_rk3",   # Shu-Osher (1988) 3rd-order SSP
    precision="float64",         # CPU float64 for maximum accuracy
    use_ct=True,                 # Constrained transport for div(B)=0
    limiter="mc",                # Monotonized Central slope limiter
)

# Python engine (full DPF physics, HLLD + SSP-RK3 defaults)
MHDSolver(
    gamma=5/3,
    use_weno5=False,             # np.gradient for stability on dynamic problems
    riemann_solver="hlld",       # HLLD default (Phase P)
    time_integrator="ssp_rk3",   # SSP-RK3 default (Phase P)
)
```

---

## Engine Step Flow (SimulationEngine.run)

```mermaid
sequenceDiagram
    participant User
    participant Engine as SimulationEngine
    participant Config as SimulationConfig
    participant Backend as Backend Solver
    participant Circuit as RLC Circuit

    User->>Engine: run(config)
    Engine->>Config: Validate + select backend
    Config-->>Engine: backend priority<br/>(athenak > athena > metal > python)

    loop Each timestep
        Engine->>Backend: compute_dt()
        Backend-->>Engine: dt (CFL-limited)
        Engine->>Circuit: advance(dt, R_plasma, L_plasma)
        Circuit-->>Engine: I(t), V(t)
        Engine->>Backend: step(state, dt, current, voltage)

        Note over Backend: 1. Reconstruction (WENO5-Z / PLM)<br/>2. Riemann solve (HLLD / HLL)<br/>3. Flux divergence<br/>4. Source terms<br/>5. CT / div(B) cleaning<br/>6. Resistive diffusion (if eta > 0)

        Backend-->>Engine: updated state dict
        Engine->>Engine: Strang splitting<br/>(physics source terms)
    end

    Engine-->>User: results dict<br/>{final_current_A, final_voltage_V,<br/>energy_conservation, steps, sim_time}
```

---

## Phase History

```mermaid
gantt
    title DPF Unified Development Phases
    dateFormat X
    axisFormat %s

    section Foundation
    A - Documentation           :done, a, 0, 1
    B - Wire Physics            :done, b, 1, 2
    C - V&V Framework           :done, c, 2, 3
    D - Braginskii Transport    :done, d, 3, 4
    E - Apple Silicon           :done, e, 4, 5

    section C++ Integration
    F - Athena++ Integration    :done, f, 5, 6
    G - Athena++ DPF Physics    :done, g, 6, 7

    section AI/ML
    H - WALRUS Pipeline         :done, h, 7, 8
    I - AI Features             :done, i, 8, 9

    section Advanced Backends
    J.1 - AthenaK Integration   :done, j1, 9, 10

    section GPU & Accuracy
    M - Metal GPU               :done, m, 10, 11
    N - Cross-Backend V&V       :done, n, 11, 12
    O - Physics Accuracy        :done, o, 12, 13
    P - Engine Accuracy         :done, p, 13, 14

    section Future
    J.2+ - Backlog              :j2, 14, 15
```
