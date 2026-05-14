# DPF-Unified Software Requirements Specification Draft

Document ID: `DPF-UNIFIED-SRS-001`
Version: `0.1`
Status: SRS development draft, not a requirements baseline
Prepared date: 2026-05-09
Template source: `/Users/anthonyzamora/Downloads/DPF-Simulator_Actual_SRS_v1.0.docx`
Project source root: `/Users/anthonyzamora/dpf-unified`

## 0. Document Control

| Field | Value |
| --- | --- |
| Document type | Software Requirements Specification draft with work-status deep dive |
| System | DPF-Unified dense plasma focus simulation workbench |
| Scope basis | Live repository, `README.md`, `CortexFindings.md`, `CodexFindings.md`, `docs/SCIENTIFIC_CLOSURE_SOURCE_QUEUE.md`, `docs/SCOPE.md`, backend docs, validation code, and the supplied SRS template |
| Scientific source rule | Local `KnowledgeReference/` records only |
| Safety/use boundary | Exploratory scientific software. The current repository shall not be represented as a validated end-to-end predictive DPF simulator. |
| Change control recommendation | After review, convert this draft into a baseline SRS with stable requirement IDs, owner, status, verification method, test mapping, and issue links. |
| Candidate baseline | `docs/DPF_REQUIREMENTS_BASELINE.md` contains the first stable-ID P0/P1 requirements table; `docs/SRS_TRACEABILITY_MATRIX.json` and `docs/SRS_TRACEABILITY_MATRIX.csv` are staged RTM exports before Doorstop import. Doorstop is installed as `Doorstop v3.1`, but the tree is not initialized. |

### 0.1 Evidence Inputs Reviewed

| Evidence source | Use in this draft |
| --- | --- |
| `README.md` | Product description, implemented physics, backend summary, current validation boundary |
| `pyproject.toml` | Python package metadata, optional dependencies, pytest markers |
| `src/dpf/config.py` | Current configuration schema and validated inputs |
| `src/dpf/engine/core.py` | Current engine orchestration and backend dispatch |
| `src/dpf/cli/main.py` | Current user-visible CLI behavior and backend mismatch |
| `docs/SCOPE.md` | Claim boundaries, known limitations, not-yet-validated areas |
| `docs/ADR_COMPUTE_AUTHORITY.md` | Accepted compute-authority and result-classification decision |
| `docs/BACKEND_PARITY.md` | Backend physics parity and unsupported-feature warning gap |
| `docs/METAL_V2_DOD.md` | Current PF-1000/Akel MLX gate status and S1/S2 blockers |
| `docs/METAL_V2_SPEC.md` | Current Metal/MLX architecture status and source-scope caveats |
| `docs/SPRINT4_VALIDATION_REVIEW.md` | Historical MLX review plus superseded-status addendum |
| `docs/SCIENTIFIC_CLOSURE_SOURCE_QUEUE.md` | Source acquisition and Akel digitization queue |
| `docs/FIRST_PRINCIPLES_FINISH_LINE_PLAN.md` | Active first-principles finish-line roadmap from PF-1000/Akel engineering probe to accepted simulation |
| `CortexFindings.md` | Detailed plan and execution log |
| `CodexFindings.md` | Running findings, verdict, completed ratchets, remaining scientific plan |
| `docs/todo_audit.md` | Current 2026-05-08 TODO/FIXME/XXX audit; stale `src/dpf/engine.py` entries are retired |

## 1. Introduction

### 1.1 Purpose

This document converts the current DPF-Unified state into an SRS-oriented development draft. It has two purposes:

1. Deep-dive the work already completed, the work explicitly planned, and the work that is missing from the current plan.
2. Provide a requirements skeleton that can be developed into a formal SRS using the supplied docx template.

This draft is intentionally conservative. It treats validation gates, tests, and local evidence as current capabilities, but does not promote engineering runs, draft digitization packets, or source-acquisition candidates into scientific validation.

### 1.2 Product Scope

DPF-Unified is currently a Lee/snowplow plus MHD simulation workbench with source-gated validation infrastructure. The live README describes it as containing Lee/snowplow, resistive-MHD, circuit-coupling, diagnostics, and validation infrastructure, but not yet an end-to-end predictive DPF simulator.

Current in-scope capability:

- Circuit-coupled dense plasma focus simulation workflows.
- Lee/snowplow axial, radial, reflected-shock, and post-pinch modeling paths.
- Conservative MHD and reduced MHD solvers across Python, MLX/Metal, PyTorch Metal, Athena++, and AthenaK-oriented wrappers.
- Diagnostics for current, voltage, energy, radiation, neutron estimates, yield tracking, and HDF5 output.
- Source-gated validation helpers tied to local `KnowledgeReference/` records.
- Scientific-accuracy gap reporting and predictive-readiness blocking.
- Local Akel 2021 figure digitization workflow with draft Fig. 1 current-waveform packet.

Current out-of-scope or not yet validated:

- End-to-end predictive neutron-yield validation.
- Same-scope spatial DPF validation across density, magnetic/EM, and temperature.
- Accepted same-scope Akel waveform NRMSE/current-dip validation for S1/S2.
- Full high-fidelity physics closure for EOS, ionization, two-temperature physics, radiation transport/opacities, impurity/ablation, Hall/FLR/kinetic/PIC, 3D instabilities, flashover/startup, restrike, anomalous resistance, and beam-target coupling.
- Formal product SRS governance, validation certificates, result labels, security controls, and export acceptance as described in the supplied SRS template.

### 1.3 Requirement Status Terms

| Status | Meaning |
| --- | --- |
| Completed | Implemented or documented in current repo and backed by test/log evidence in the findings docs. |
| Partial | Implemented partly, but missing evidence, integration, acceptance tests, or source scope. |
| Planned | Explicitly listed in `CortexFindings.md`, `CodexFindings.md`, or `docs/SCIENTIFIC_CLOSURE_SOURCE_QUEUE.md`. |
| Unplanned | Needed for an SRS-grade product but not currently represented as an owned plan item. |
| Blocked | Known needed work cannot support a validation claim until stated evidence or review is complete. |
| Deferred | Acknowledged future capability outside current validation scope. |

## 2. Overall Description

### 2.1 Product Perspective

The current system is a Python package and local workbench, not yet a formally baselined product. It has:

- A Pydantic configuration model in `src/dpf/config.py`.
- A CLI entrypoint `dpf` in `src/dpf/cli/main.py`.
- A central simulation engine in `src/dpf/engine/core.py`.
- Backends for Python, Athena++, AthenaK, PyTorch Metal, and MLX paths.
- A FastAPI/WebSocket server path.
- HDF5 diagnostics and Well-format export support.
- A large pytest suite with source-gated validation, physics-fidelity, uncertainty, and digitization tests.

The supplied SRS template assumes an explicit T0/T2 architecture where a T0 float64 reference backend is authoritative and T2 MLX/Metal is preview-only. DPF-Unified now records this as an explicit compute-authority decision in `docs/ADR_COMPUTE_AUTHORITY.md`: Python, Athena, and AthenaK are reference candidates, while Metal, MLX, hybrid, and unresolved auto outputs default to Preview unless future accepted same-scope validation rules promote a specific result scope. Engine run summaries, sidecar manifests, FastAPI `SimulationInfo`, GUI wire types, and the TopBar Preview/Reference badge now emit these labels.

### 2.2 Current Subsystems

| Subsystem | Current status |
| --- | --- |
| Configuration | Partial. Pydantic validates many physics/config fields, and result classification/manifests are now formalized. Product-level project schema remains open. |
| Circuit | Partial/completed for core RLC and Lee/snowplow coupling workflows. Full field-derived Poynting/back-EMF coupling evidence remains open. |
| Snowplow/Lee model | Partial/completed. Current-factor circuit-loading fix is landed and Akel M2/M6 source-scoped probes are current. Consolidation into one documented reference implementation remains a review finding. |
| MHD solvers | Partial/completed for scheduled Tier-3 code verification. The local packet now includes finite-volume, cylindrical convergence, circuit-energy, resistive-diffusion, backend-parity, restart, convergence, and scope-limit evidence. DPF spatial validation and broader scientific validation remain open. |
| MLX/Metal | Partial/completed for key stability ratchets. Standalone Akel source-scoped no-crowbar probes reach 12 us; S1/S2 waveform acceptance remains blocked. |
| Diagnostics | Partial. HDF5 diagnostics, neutron/yield/regime tools, and validation helpers exist. Calibrated detector response and same-scope neutron outputs remain open. |
| Digitization | Partial. Gate and Akel Fig. 1 draft packet exist; independent review acceptance is missing. |
| Validation governance | Partial/completed for scientific gates, candidate SRS baseline, manifests, labels, and fail-closed validation certificate schema. Doorstop import and formal release matrix remain open. |
| Export bridge | Partial. v1 scope now accepts HDF5 diagnostics and Well HDF5, and defers VTK/VTU, CGNS/HDF5, OpenFOAM, and Ansys/PyMAPDL until writer/readability/license-aware tests exist. |
| UI/server | Partial. API/GUI wire surfacing now exposes authority labels, readiness blockers, digitization status, and units/dimensions metadata; UI mode requirements remain open. |

## 3. Deep-Dive Work Inventory

### 3.1 Completed Items

These items have current evidence in the repo or findings docs.

| Area | Completed item | Evidence and notes |
| --- | --- | --- |
| Claim hygiene | README and major docs now frame the project as a workbench, not a validated predictive simulator. | `README.md`, `docs/SCOPE.md`, `CodexFindings.md`. |
| KR source authority | Local source authority is enforced through manifest/source audits and source-gated evidence helpers. | `src/dpf/validation/kr_targets.py`, `src/dpf/validation/quality_assessment.py`, tests. |
| Corpus review | DPF-relevant local markdown source review reached closure. | Later `CortexFindings.md` entries supersede the older top-of-file "Current Execution Position"; current status says 96/96 DPF-relevant markdown files are review-closed. |
| Predictive readiness gate | Predictive readiness blocks unless source authority and the required evidence tiers pass. | `predictive_readiness_report()` and tests. |
| High-fidelity gap report | Scientific-accuracy gaps are machine-readable and exported in app results. | `scientific_accuracy_gap_report()`, app-level post-processing. |
| Tier evidence hardening | Placeholder/unsourced dictionaries no longer support circuit, snowplow, spatial, neutron, physics-fidelity, coupling, or UQ tiers. | `CodexFindings.md` completed ratchets and tests. |
| MLX safe import | Broken/sandboxed MLX native imports no longer abort full pytest collection. | Safe child-process MLX detection and test hardening are logged in findings. |
| MLX PF-1000 probe stability | Earlier nonfinite windows were narrowed and patched; standalone probe reached longer targets. | `CodexFindings.md` PF-1000 MLX probe sections. |
| Akel source-scoped no-crowbar M2/M6 | Current-factor-corrected `pf1000_akel` no-crowbar run reached 12 us with `peak_I_MA=1.150507`, inside shot-12581 M2 band. | `docs/METAL_V2_DOD.md`, `docs/METAL_V2_SPEC.md`, `CortexFindings.md`. |
| Lee current-factor loading | CPU `SnowplowModel` circuit-facing `Lp/dLdt` now applies current-factor scaling while preserving unscaled geometry coefficient. | `src/dpf/fluid/snowplow.py`, `tests/test_snowplow_consolidated.py`, findings. |
| Akel Fig. 1 figure artifact | Fig. 1 crop exists in `KnowledgeReference/figures/` with SHA-256 provenance. | `docs/SCIENTIFIC_CLOSURE_SOURCE_QUEUE.md`. |
| Akel Fig. 1 draft arrays | Draft measured/computed current arrays exist in `KnowledgeReference/digitization/`. | 294 measured-current candidate points and 34 computed-current candidate points. |
| Akel overlay residual | Internal vector round-trip residual is below verifier RMS threshold. | RMS `0.213455189 px` over 328 points. |
| Digitization status helper | `pf1000_16kv_current_waveform_digitization_candidate_evidence()` reports draft waveform status as `blocked_by_review`, not missing data. | `src/dpf/validation/kr_targets.py`, tests. |
| Configuration schema | Pydantic config validates many numerical/physics fields and dimensions. | `src/dpf/config.py`. |
| CLI baseline | `dpf simulate`, `dpf verify`, backend listing, server, sweep, and Well export commands exist. | `src/dpf/cli/main.py`. |
| HDF5 diagnostics/checkpointing | HDF5 output and checkpoint/restart helpers exist; scheduled Tier-3 restart reproducibility evidence now passes for the deterministic fixture. | `src/dpf/diagnostics/hdf5_writer.py`, `src/dpf/engine/state_management.py`, `src/dpf/diagnostics/checkpoint.py`, `results/mhd_restart_reproducibility_evidence.json`. |

### 3.2 Explicitly Planned Items

These are already planned in current project artifacts.

| Plan area | Planned work | Current driver |
| --- | --- | --- |
| Akel review gate | Complete independent review of the Akel Fig. 1 digitization packet and set review status only if accepted. | `docs/SCIENTIFIC_CLOSURE_SOURCE_QUEUE.md`, `CortexFindings.md`. |
| S1/S2 waveform closure | Use accepted same-scope digitized Akel current waveform and current-dip evidence with uncertainty to support S1/S2. | `docs/METAL_V2_DOD.md`, `CodexFindings.md`. |
| Akel remaining figures | Digitize Fig. 2-4 current waveforms and Fig. 5-6 neutron-yield plots as priority 1/2 queue items. | `scientific_closure_digitization_queue()`. |
| Source acquisition | Acquire exact local source documents for missing PF-1000/neutron detector/spectrum/anisotropy needs before KR ingestion. | `docs/SCIENTIFIC_CLOSURE_SOURCE_QUEUE.md`. |
| Same-scope validation packet | Promote one same-scope validation packet only when KR-backed circuit, phase, spatial, neutron, detector, field coupling, physics-fidelity, and uncertainty evidence share one scope. | `CortexFindings.md`, `CodexFindings.md`. |
| Tier 2 phase validation | Build source-backed axial, radial, pinch/stagnation timing targets for production runs. | Remaining high-fidelity plan. |
| Tier 3 numerical fidelity | Keep the scheduled complete code-verification packet wired into readiness/release reporting without promoting it beyond Tier 3. | `results/mhd_tier3_numerical_packet.json`; higher scientific validation remains separate. |
| Tier 4 spatial validation | Ingest same-scope density, magnetic/EM, and temperature diagnostics. | Remaining high-fidelity plan. |
| Tier 5 neutron validation | Add mechanism-separated neutron histories, spectrum, anisotropy, detector/activation response, and yield uncertainty. | Remaining high-fidelity plan. |
| Physics-fidelity closure | Mark or validate EOS, ionization, two-temperature, radiation transport, impurity/ablation, kinetic/Hall/FLR, 3D, startup, restrike, anomalous resistance, and beam-target effects. | Remaining high-fidelity plan. |
| Circuit-field coupling fidelity | Define evidence for inductance, dL/dt, back-EMF, Poynting flux, circuit energy, and snowplow-to-MHD transition timing. | Remaining high-fidelity plan. |
| UQ propagation | Extend UQ from circuit waveform tools into phase, spatial, neutron, numerical, model-form, and shot-to-shot evidence. | Remaining high-fidelity plan. |
| First-principles finish-line plan | Execute the phased PF-1000/Akel path from no-hidden-limiter numerics through source-backed startup, validated field-circuit coupling, physics-fidelity closure, same-scope evidence, neutron authority, certificate gate, and second-scope generalization. | `docs/FIRST_PRINCIPLES_FINISH_LINE_PLAN.md`. |
| Long PF-1000 gate handling | Decide whether to convert long xfailed PF-1000 fixture into opt-in endurance/regression path with adequate step cap. | Latest findings entries. |
| Findings doc maintenance | Update stale top-of-file `CortexFindings.md` current-position text so it reflects later source-review closure. | Current review observation. |

### 3.3 Missing or Insufficiently Planned Items

These are important for an SRS-grade system but are not yet adequately owned in current plans.

| Gap | Why it matters | Suggested SRS owner |
| --- | --- | --- |
| Doorstop import and traceability matrix | A candidate requirements baseline and staged JSON/CSV RTM export exist, but accepted rows are not yet imported into a Doorstop tree. | Product/V&V |
| Validation certificate workflow integration | The certificate schema/writer exists, but no accepted scientific certificate can be produced until evidence gates pass. | V&V |
| Project lifecycle UI/API integration | Local project create/load/duplicate/archive helpers exist with provenance-preserving manifests; FastAPI now exposes bounded lifecycle endpoints under `DPF_PROJECTS_ROOT`, and GUI client wire types are available. | Product/UI/API |
| Backend parity campaign | Unsupported-feature diagnostics exist; broader parity evidence for production DPF observables remains open. | Engine/UX |
| UI modes and status UX | Pedagogical/advanced modes, warning labels, validation status display, and preview/reference distinctions are not baselined. | Frontend/Product |
| Local-first/security audit | Local bind/CORS/share defaults, hardware-driver import scans, runtime-AI mutation scans, and manifest classification metadata are now implemented. Accepted HDF5/Well v1 export paths now carry fail-closed classification/provenance labels; deferred external bridge schemas still need classification propagation before acceptance. | Security/Product |
| Air-gap artifact evidence | A fail-closed air-gap gate and runbook now exist, but the current repo is not release-ready until the wheelhouse, hash manifest, and offline logs are produced. | Release/QA |
| TODO backlog routing | `docs/todo_audit.md` is refreshed; it now routes the live Athena++ circuit-source marker and MLX AMR overlay marker without carrying stale `src/dpf/engine.py` blockers. | Engineering |
| Product release envelope | Current scientific plan is high-fidelity-first; product release levels and acceptance criteria are not defined. | Product/Architecture |

## 4. Functional Requirements Draft

The following requirements adapt the supplied SRS structure to the current repository. `Status` is the current implementation/planning state, not a pass/fail certification.

### 4.1 Governance and Source Authority

| ID | Priority | Requirement | Status | Verification approach |
| --- | --- | --- | --- | --- |
| GOV-001 | P0 | The system shall treat local `KnowledgeReference/` records as the only scientific source authority for validation claims. | Completed | Source-authority tests and KR manifest audits. |
| GOV-002 | P0 | The system shall fail closed when evidence lacks local source path, line range, source hash, or required KR status. | Completed/partial | Validation helper tests; expand to all evidence types. |
| GOV-003 | P0 | The system shall keep predictive and high-fidelity readiness blocked until same-scope evidence passes all required tiers. | Completed/partial | `predictive_readiness_report()` and high-fidelity gap tests. |
| GOV-004 | P1 | The SRS shall define allowed user-facing claim labels and prohibit promotion of engineering tests into scientific validation. | Completed | Compute-authority ADR, result classification schema, UI/API surfacing, and negative promotion tests. |
| GOV-005 | P1 | Every major finding or validation change shall update `CodexFindings.md` and, when plan status changes, `CortexFindings.md`. | Partial | Inspection of findings docs during release reviews. |

### 4.2 Project, Configuration, and Runtime Management

| ID | Priority | Requirement | Status | Verification approach |
| --- | --- | --- | --- | --- |
| SYS-001 | P0 | The system shall load and validate simulation configuration from JSON/YAML-like project files. | Partial/completed | Existing Pydantic config tests; add product-level schema tests. |
| SYS-002 | P0 | The system shall validate units, dimensional bounds, supported geometry, supported backend, and unsupported physics options before launch. | Partial | Config validators and backend capability diagnostics exist; broader project preflight remains open. |
| SYS-003 | P0 | The system shall produce a run manifest for every solver execution. | Completed | Engine sidecar manifest tests and failed-run manifest tests. |
| SYS-004 | P0 | The system shall classify every result as Reference, Preview, Derived Diagnostic, Exploratory, Superseded, or Invalid. | Completed | Output manifest inspection and negative promotion tests. |
| SYS-005 | P1 | The CLI shall expose every supported backend that the config/engine can run. | Completed | `mlx` is accepted by `dpf simulate --backend`; backend listing includes MLX. |
| SYS-006 | P1 | The system shall support project create/load/duplicate/archive operations with preserved provenance. | Completed/partial | Local project lifecycle helpers and tests exist; UI/API integration remains optional product-surface work. |

### 4.3 Backend and Precision Requirements

| ID | Priority | Requirement | Status | Verification approach |
| --- | --- | --- | --- | --- |
| CMP-001 | P0 | The SRS shall designate an authoritative reference backend or reference workflow for validation claims. | Completed | `docs/ADR_COMPUTE_AUTHORITY.md` and validation tests. |
| CMP-002 | P0 | MLX/Metal float32 outputs shall be labeled according to their scientific status and shall not satisfy validation claims without source-gated evidence. | Completed | Result classification tests keep MLX Preview/non-certifying. |
| CMP-003 | P0 | T0/T2 or equivalent backend authority boundaries shall be visible in UI, API, logs, and exported artifacts. | Completed | Run summaries/manifests, FastAPI `SimulationInfo`, GUI wire types, and TopBar authority badges. |
| CMP-004 | P1 | The system shall compute projected memory demand before solver start and refuse unsafe runs above a defined threshold. | Completed | Memory preflight tests. |
| CMP-005 | P1 | The system shall record runtime memory telemetry for long or GPU runs. | Completed | Runtime peak RSS telemetry and optional MLX backend telemetry are attached to summaries/manifests. |
| CMP-006 | P1 | Backend parity tests shall fail or warn when selected physics is unsupported by the chosen backend. | Completed | Backend capability diagnostics and MLX flag pass-through tests. |

### 4.4 Physics and Solver Requirements

| ID | Priority | Requirement | Status | Verification approach |
| --- | --- | --- | --- | --- |
| PHY-001 | P0 | The system shall solve the coupled circuit and plasma workflow using validated inputs and SI units. | Partial | Existing config/engine/circuit tests; expand unit audits. |
| PHY-002 | P0 | Lee/snowplow circuit-facing `Lp` and `dL/dt` shall preserve current-factor scaling without corrupting unscaled geometry coefficients. | Completed | Snowplow tests and Akel probe evidence. |
| PHY-003 | P0 | PF-1000/Akel scientific gates shall not mix Akel 16 kV shot-12581 scope with Scholz/Gribkov full-energy PF-1000 scope. | Completed/partial | Source-scope tests and docs. |
| PHY-004 | P0 | S1/S2 waveform acceptance shall require accepted same-scope digitized current waveform and uncertainty evidence. | Planned/blocked | Digitization review gate, comparator, and waveform tests. |
| PHY-005 | P1 | The project shall define one documented Lee/RADPF reference implementation and require backend parity against it. | Planned | Code consolidation and parity tests. |
| PHY-006 | P1 | MHD-mode field/circuit coupling claims shall require validated field-derived inductance, dL/dt/back-EMF, Poynting power, and energy balance evidence. | Planned | `field_coupling_validation` evidence record. |
| PHY-007 | P1 | Physics-fidelity gaps shall be explicit for each run: EOS, ionization, two-temperature, radiation transport, impurity, kinetic/Hall/FLR, 3D, startup, restrike, anomalous resistance, and beam-target coupling. | Partial/planned | Current gap report plus per-run evidence expansion. |
| PHY-008 | P0 | First-principles mode shall drive circuit feedback from resolved field power and conservation ledgers rather than Lee/RADPF closure factors. | Partial | Field-coupled engineering probe exists; accepted field-coupling evidence remains blocked. |
| PHY-009 | P0 | Accepted first-principles runs shall not depend on hidden engineering limiters or unreported state repair. | Planned/blocked | Limiter telemetry and replacement with verified numerical controls. |
| PHY-010 | P0 | First-principles startup shall be source-backed for breakdown, preionization, electrode boundary, initial plasma, and sheath evidence. | Planned/blocked | Source-backed startup state generator, evidence packet, and comparator tests. |
| PHY-011 | P0 | Total neutron-yield authority shall require resolved thermonuclear history plus accepted kinetic/hybrid beam-target production and same-scope neutron UQ. | Planned/blocked | Mechanism-separated neutron evidence and first-principles neutron authority tests. |
| PHY-012 | P0 | First-principles acceptance shall define dimensionality and any MHD-to-kinetic handoff for the claimed interval and observables. | Planned/blocked | Dimensionality/handoff packet plus tests that reject out-of-scope observables. |
| PHY-013 | P0 | First-principles numerical-fidelity packets shall define named tests, norms, mesh families, tolerances, precision/backend scope, and limiter-zero acceptance. | Planned/blocked | Numerical-fidelity packet, limiter-zero evidence, and reference-workflow tests. |
| PHY-014 | P0 | The active first-principles circuit power port shall pass Poynting or `J.E`, electrode-work, time-centering, sign, and residual tests without clipped back-EMF for acceptance. | Planned/blocked | Power-port component tests and integrated energy-ledger tests. |
| PHY-015 | P0 | First-principles startup shall be generated as a source-backed boundary-value problem with current-density, field, ionization, temperature, and sheath-liftoff consistency checks. | Planned/blocked | Startup BVP packet, source evidence, and consistency tests. |
| PHY-016 | P0 | Every active or bounded-out physical closure shall have a packet with source equations, symbol map, units, validity regime, verification, sensitivity/UQ, and claim impact. | Planned/blocked | Closure packet registry and physics-fidelity tests. |
| PHY-017 | P1 | Accepted first-principles execution shall run through one package-native `src/dpf` path shared by CLI, API, config, and app surfaces. | Planned | Package-native runner consolidation and app-only rejection test. |

### 4.5 Digitization and Validation Evidence

| ID | Priority | Requirement | Status | Verification approach |
| --- | --- | --- | --- | --- |
| DIG-001 | P0 | Digitized figure/table data shall pass one-for-one provenance verification before it supports validation. | Completed/partial | `digitization_verification_evidence()` tests. |
| DIG-002 | P0 | Akel Fig. 1 draft data shall remain non-accepting until independent review count and `review_status="accepted"` pass. | Completed/blocked | Current helper returns `blocked_by_review`. |
| DIG-003 | P0 | Digitization packets shall include source path/hash, local figure path/hash, page/figure ID, axis calibration, units, series arrays, overlay residuals, and reviewer metadata. | Completed/partial | Digitization schema/gate tests. |
| DIG-004 | P1 | Akel Fig. 2-4 current waveforms and Fig. 5-6 yield plots shall be tracked through the same queue/status workflow. | Planned | Queue tests and future packet tests. |
| VAL-001 | P0 | Tier 2 shall pass only from same-device KR-backed phase targets attached to production results. | Planned/blocked | Phase target records and production comparison tests. |
| VAL-002 | P0 | Tier 4 shall require same-scope density, magnetic/EM, and temperature evidence. | Planned/blocked | Spatial evidence combiner tests plus real KR packets. |
| VAL-003 | P0 | Tier 5 shall require same-scope neutron timing, spectrum, anisotropy, detector/activation response, scalar yield, and uncertainty. | Planned/blocked | Neutron validation outputs and evidence tests. |
| VAL-004 | P1 | The system shall produce a validation certificate artifact only when all linked gates pass. | Completed | Validation certificate schema/writer rejects blocked, failed, draft, and cross-scope evidence. |
| VAL-005 | P0 | First-principles PF-1000/Akel acceptance shall require same-scope waveform, phase, spatial, neutron, detector, coupling, physics-fidelity, numerical-fidelity, and UQ evidence in one certificate path. | Planned/blocked | Finish-line plan and certificate gate. |

### 4.6 Diagnostics, Data, and Export Requirements

| ID | Priority | Requirement | Status | Verification approach |
| --- | --- | --- | --- | --- |
| DAT-001 | P0 | The system shall write time-series diagnostics with units and a consistent time base. | Completed | HDF5 diagnostics unit/time-base tests. |
| DAT-002 | P0 | HDF5 outputs shall include provenance, backend, solver mode, validation status, and source/readiness metadata. | Completed | HDF5 schema/time-base attributes plus embedded backend, solver mode, validation status, result label, source authority, and classification JSON. |
| DAT-003 | P1 | Checkpoint/restart shall preserve state sufficiently for deterministic restart comparisons. | Completed | Deterministic restart evidence builder and regression test; `results/mhd_restart_reproducibility_evidence.json`. |
| EXP-001 | P1 | Well-format HDF5 export for training data shall be schema-tested and provenance-tagged. | Completed | Well schema/unit tests, engine adapter metadata tests, normal-run flush regression, circuit scalar forwarding, and cylindrical grid metadata. |
| EXP-002 | P2 | VTK/VTU, CGNS/HDF5, OpenFOAM, and Ansys/PyMAPDL export requirements shall be explicitly accepted, deferred, or rejected for v1.0. | Completed | `docs/EXPORT_SCOPE_V1.md`; `export_scope_decisions()` tests. |

### 4.7 UI, API, and User Experience Requirements

| ID | Priority | Requirement | Status | Verification approach |
| --- | --- | --- | --- | --- |
| UI-001 | P1 | User-facing outputs shall display predictive-readiness and high-fidelity-readiness blockers. | Completed | FastAPI `SimulationInfo`, GUI wire type, TopBar blocker count, and server readiness tests. |
| UI-002 | P1 | Preview/non-certifying outputs shall be visibly labeled and blocked from Reference promotion. | Completed | API result classification plus TopBar Preview/Reference badge; negative promotion remains enforced in artifact tests. |
| UI-003 | P1 | The system shall offer beginner/pedagogical and advanced engineering workflows without changing physical results implicitly. | Planned | UI mode requirements and workflow tests remain to be designed. |
| API-001 | P1 | Backend API schemas shall expose units, dimensions, backend mode, validation status, and source authority. | Completed | `SimulationInfo` plus `/api/metadata/units` endpoint and GUI wire type. |

### 4.8 Security, Local-First, and Release Requirements

| ID | Priority | Requirement | Status | Verification approach |
| --- | --- | --- | --- | --- |
| SEC-001 | P0 | The system shall not control physical pulsed-power hardware or lab equipment in the current release. | Implemented | `local_first_security_audit()` plus `tests/test_local_first_security.py`. |
| SEC-002 | P0 | The system shall run local-first and shall not transmit project data externally by default. | Implemented | Local UI bind defaults, explicit share opt-in, localhost CORS defaults, and wildcard-CORS opt-in tests. |
| SEC-003 | P1 | Runtime AI agents shall not modify solver code, active config, or active simulation state during execution. | Implemented | Runtime AI boundary scan in `dpf.security.local_first`; mutation findings fail the local-first audit. |
| SEC-004 | P1 | Project and export artifacts shall support owner-supplied classification/distribution metadata. | Implemented | Project manifests and run manifests carry owner-supplied classification/distribution metadata; accepted HDF5 outputs embed artifact classification JSON. |
| REL-001 | P1 | Releases shall have an offline/air-gap-capable build and test path where licensing permits. | Partial | `docs/AIR_GAP_RELEASE_GATE.md` and `airgap_release_gate()` define and test the fail-closed gate; wheelhouse/hash artifacts and offline logs are still missing. |
| REL-002 | P0 | Every P0 requirement in the baseline SRS shall map to at least one test, inspection, analysis, or demonstration. | Partial | Candidate baseline maps verification methods; Doorstop import remains open. |

## 5. V&V Requirements Draft

| ID | Priority | Requirement | Status |
| --- | --- | --- | --- |
| VNV-001 | P0 | Maintain KR-only scientific source authority and source-line audits for validation targets. | Completed/partial |
| VNV-002 | P0 | Keep predictive readiness blocked when evidence tiers are absent, partial, cross-scope, unsourced, or malformed. | Completed/partial |
| VNV-003 | P0 | Distinguish numerical verification from DPF experimental validation in reports and UI. | Partial |
| VNV-004 | P0 | Require same-scope circuit, phase, spatial, neutron, detector, coupling, physics-fidelity, and UQ support before high-fidelity claims. | Planned/blocked |
| VNV-005 | P0 | Require accepted digitization packets before waveform/yield plot arrays can support validation. | Completed/blocked for Akel Fig. 1 |
| VNV-006 | P1 | Define and maintain a formal validation certificate schema. | Completed |
| VNV-007 | P1 | Refresh the historical TODO/placeholder audit against current files before adding it to SRS backlog. | Completed |
| VNV-008 | P1 | Add acceptance tests for product-level manifests, result labels, memory preflight, local-first behavior, and export bridges if in scope. | Partial; manifests, labels, memory preflight, local-first behavior, and v1 export scope tests exist; remaining external bridge tests are deferred by scope decision. |

## 6. Traceability Snapshot

| Stakeholder need | Current support | Remaining traceability gap |
| --- | --- | --- |
| Explore DPF physics locally | CLI, app paths, config, engine, backends, diagnostics. | Product workflow requirements and UI acceptance tests. |
| Avoid overclaiming scientific validity | KR-only rule, readiness gates, source-authority helpers, findings docs. | Formal result labels, validation certificates, UI claim controls. |
| Compare to local scientific sources | KR target manifest, source audits, digitization gate, Akel draft packet. | Accepted waveform packets, same-scope spatial/neutron/UQ data. |
| Run MLX/Metal PF-1000 probes | Standalone probe path and current Akel 12 us evidence. | S1/S2 acceptance and long fixture policy. |
| Generate diagnostics | HDF5 diagnostics, energy/yield/regime helpers. | Unit/provenance manifest and calibrated detector response. |
| Export downstream artifacts | HDF5 diagnostics and Well HDF5 are accepted for v1; VTK/VTU, CGNS, OpenFOAM, and Ansys/PyMAPDL are explicitly deferred. | Embedded HDF5 provenance and future external bridge smoke tests. |
| Reproduce and certify runs | Many pytest slices, run manifests, certificate schema, findings logs, and a fail-closed air-gap gate exist. | Doorstop traceability import, accepted certificates, and real offline CI logs. |

## 7. Open Items and Controlled TBDs

| TBD ID | Open item | Required decision |
| --- | --- | --- |
| TBD-001 | Should the SRS adopt the template's T0/T2 architecture exactly? | Decide whether Python/Athena++ become T0 reference authority and MLX becomes T2 preview, or define a different authority model. |
| TBD-002 | What is v1.0 product scope? | Decide whether v1.0 is a science-closure release, engineering workbench release, or formal product release with UI/export/security controls. |
| TBD-003 | What result labels are mandatory? | Adopt labels such as Reference, Preview, Derived Diagnostic, Exploratory, Superseded, Invalid. |
| TBD-004 | What export bridges are in v1.0? | Resolved for candidate v1 scope: accept HDF5 diagnostics and Well HDF5; defer VTK/VTU, CGNS/HDF5, OpenFOAM, and Ansys/PyMAPDL. |
| TBD-005 | What is the SRS memory safety rule? | Decide threshold, formula, telemetry, and failure behavior. |
| TBD-006 | What residual security posture is required? | Local-first defaults, runtime AI mutation boundary, and project/manifest/HDF5 classification metadata are implemented; decide audit-log depth. |
| TBD-007 | Who can perform independent digitization review? | Define reviewer role, evidence format, acceptance criteria, and whether self-review is disallowed. |
| TBD-008 | How should stale findings headers be handled? | Update `CortexFindings.md` top status so it no longer contradicts later corpus-review closure. |
| TBD-009 | What is the accepted long PF-1000 fixture policy? | Keep xfailed scientific gate, convert to opt-in endurance regression, or define a separate release test. |
| TBD-010 | How should future TODO audits be refreshed? | Re-run the scoped `rg` commands in `docs/todo_audit.md` and supersede stale entries with dated addenda. |

## 8. Recommended Next Actions

1. Review `docs/DPF_REQUIREMENTS_BASELINE.md` and the staged RTM exports, then import accepted rows into a Doorstop requirements tree.
2. Continue the science ratchet from the current active point: Akel Fig. 1 independent review and S1/S2 comparator path.
3. Implement the remaining product controls: audit-log depth and real offline release artifacts/logs.
4. Use `docs/todo_audit.md` as the current TODO backlog source and refresh it with dated addenda when source markers change.

## Appendix A. Current Akel Fig. 1 Digitization State

| Field | Current value |
| --- | --- |
| Figure artifact | `KnowledgeReference/figures/akel-2021-fig1-current-waveform-shot-12581.png` |
| Draft packet | `KnowledgeReference/digitization/akel-2021-fig1-current-waveform-shot-12581-draft-packet.json` |
| Source SVG | `KnowledgeReference/digitization/akel-2021-page3.svg` |
| Measured-current candidate points | 294 |
| Computed-current candidate points | 34 |
| Combined overlay RMS | `0.213455189 px` |
| Current readiness status | `blocked_by_review` |
| Missing checks | `independent_review_missing`, `review_status_not_accepted` |
| Scientific boundary | Does not close S1/S2 or tier-1 waveform validation. |

## Appendix B. Completed vs Planned vs Unplanned Summary

| Category | Counted summary |
| --- | --- |
| Completed | Source-gated validation infrastructure, claim cleanup, corpus review closure, readiness blockers, MLX safe import, PF-1000/Akel no-crowbar M2/M6 evidence, digitization workflow, Akel Fig. 1 draft arrays and overlay residual, formal SRS candidate baseline, compute-authority ADR, run manifests, result classification, validation certificate schema/writer, S1/S2 comparator scaffold, CLI MLX backend consistency, backend unsupported-feature diagnostics, launch memory preflight, runtime peak memory telemetry, project lifecycle helpers, UI/API authority/readiness surfacing, API units/dimensions schema, v1 export bridge scope, embedded HDF5 run metadata, fail-closed air-gap gate, current TODO audit, and local-first/security controls. |
| Planned | Akel independent review, remaining Akel figures, same-scope evidence packet, phase/spatial/neutron/UQ/physics/coupling closure, source acquisition, project lifecycle UI/API integration, audit-log depth, and real offline release artifacts/logs. |
| Unplanned | No P0/P1 product controls from the draft-SRS review remain completely unplanned; several remain planned or partial in `docs/DPF_REQUIREMENTS_BASELINE.md`. |
