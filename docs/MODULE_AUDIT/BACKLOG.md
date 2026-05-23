# Module Audit Backlog

Status date: 2026-05-11

This backlog tracks module-specific work discovered during the audit. It is not
an implementation plan by itself; each item still needs current-code review,
source review, and task sizing before implementation.

## Validation Module

| ID | Status | Task |
| --- | --- | --- |
| VAL-001 | Blocked | Split validation package roles into clearer layers: source/KR authority, target extraction, digitization, result classification, numerical verification, experimental comparison, and legacy calibration. |
| VAL-002 | Blocked | Make the validation package fail closed by default for every public API path, not only the newest SRS/artifact helpers. |
| VAL-003 | Blocked | Re-audit all `ExperimentalDevice` registry values against `KnowledgeReference/` line references and mark every device/field as accepted, reconstructed, reference-only, or unverified. |
| VAL-004 | Blocked | Remove or quarantine validation reliance on reconstructed waveforms until accepted digitized traces and uncertainty packets exist. |
| VAL-005 | Blocked | Complete Akel Fig. 1 independent review and Figs. 2-6 digitization/review packets before using Akel plots for waveform/yield validation. |
| VAL-006 | Blocked | Convert remaining partial KR target groups into same-scope target packets: circuit waveform, phase timing, spatial temperature, neutron yield, detector response, spectrum, anisotropy, and uncertainty. |
| VAL-007 | Complete 2026-05-09 | Add explicit provenance classes to calibration results so optimized parameters cannot be confused with experimental validation. |
| VAL-008 | Blocked | Review analytic helper modules such as Bennett, z-pinch, Sedov, Riemann, magnetized Noh, and pinch diagnostics as verification tools only, with claim limits. |
| VAL-009 | Blocked | Review source-line semantic audit strength; marker hits are useful guards but not human validation of extracted values. |
| VAL-010 | Complete 2026-05-09 | Ensure quality/readiness reports are surfaced consistently through engine, API, CLI, GUI, exports, manifests, and certificates. Certificates carry result classification, artifact classification, readiness summaries, and blockers with fail-closed accepted-certificate rules; HDF5 metadata embeds compact readiness/source-blocker evidence when summaries provide it. |
| VAL-011 | Complete 2026-05-11 | Add an Akel Fig. 1 source-integrity verifier before independent review. `scripts/verify_akel_digitization_source_integrity.py` checks local markdown/PDF/JSON parity, PDF/hash matches, Fig. 1 crop hash, page-3 SVG hash, draft packet hash, source caption line window, measured/computed series point counts, and non-review digitization failures. It passes today only as a pre-review guardrail with `accepted_for_validation=false`; Akel review and S1/S2 remain blocked under `VAL-005`. |
| VAL-012 | Complete 2026-05-11 | Promote local `downloaded_books_papers/Research Papers` intake PDFs into KR text-parity records and remove exact intake duplicates. `scripts/promote_research_papers_to_kr.py --apply` promoted 54 unique PDFs into `KnowledgeReference/`, skipped 7 already represented source-level records, and deleted 16 byte-for-byte duplicate intake files. The generated records are `text_parity_extracted_review_needed`; typed target extraction and validation acceptance remain blocked under `VAL-006`. |
| VAL-013 | Complete 2026-05-11 | Correct the electron-ion Coulomb-log convention in `src/dpf/validation/pinch_physics.py::coulomb_mean_free_path()` during the formulary audit. The helper no longer defaults to the NRL electron-electron expression for an electron-ion mfp calculation, and focused formulary transport tests cover the branch behavior. |
| VAL-014 | Complete 2026-05-23 | Add the SS19 UQ/comparator/certificate pipeline evaluator. `build_ss19_certificate_pipeline(...)` now checks comparator mapping, uncertainty-budget completeness, run/source hashes, upstream blockers, negative controls, and review status; it refuses incomplete and complete-production stacks while accepting only a synthetic complete fixture for wiring, with all runtime/first-principles acceptance flags false. |

## Engine/Core Module

| ID | Status | Task |
| --- | --- | --- |
| ENG-001 | Complete 2026-05-09 | Make `app_engine.run_mhd_simulation_core` honor `n_steps` through `engine.run(max_steps=n_steps)` or remove/rename the parameter so bounded UI requests cannot silently run full duration. |
| ENG-002 | Complete 2026-05-09 | Replace broad silent Lee fallback with explicit failed-MHD status unless the caller opts into fallback behavior. |
| ENG-003 | Complete 2026-05-09 | Audit MLX/Metal operator ownership for Nernst, diffusion, radiation, and transport; add tests proving each requested feature is applied exactly once or explicitly skipped. GPU-owned Nernst/diffusion are no longer applied by the Python operator path, and backend diagnostics report backend-owned, fallback, or Python-owned operator behavior. |
| ENG-004 | Complete 2026-05-09 | Wire `BreakdownConfig` into `SimulationEngine` or label it config-only/experimental in engine summaries and readiness output. |
| ENG-005 | Complete 2026-05-10 | Add KR line-reference metadata or explicit empirical/unverified status for every preset value currently supported only by narrative comments. `src/dpf/presets.py` now exposes `preset_value_authority()` / `preset_authority_manifest()` records for every preset config leaf; records fail closed as `not_validation_evidence` unless future KR packets promote them. |
| ENG-006 | Complete 2026-05-09 | Add tests for `src/dpf/constants.py` and clarify whether constants are standards-scoped implementation constants or KR-scoped scientific inputs. |
| ENG-007 | Complete 2026-05-09 | Keep backend `"production"` labels separate from validation/readiness labels in CLI, API, GUI, exports, and manifests. |
| ENG-008 | Complete 2026-05-09 | Preserve first-failure evidence around `_sanitize_state` by adding fail-fast or first-nonfinite telemetry paths for audit/probe runs. |
| ENG-009 | Complete 2026-05-11 | Correct SI conservative-MHD energy flux in `src/dpf/fluid/mhd_solver.py` and `src/dpf/fluid/cylindrical_mhd.py` so the magnetic flux term is `B(v dot B)/mu_0`. This is a formula-correctness fix only, not experimental validation evidence. |

## Metal/MLX Module

| ID | Status | Task |
| --- | --- | --- |
| MLX-001 | Complete 2026-05-09 | Replace or downgrade unsupported authority comments in MLX coupling/solver surfaces unless direct local KR traceability is added. MLX coupling methods now expose fail-closed authority metadata, engine output carries `mhd_coupling_authority`, and claim-guard tests prevent reintroducing "correct"/"first-principles" coupling language. |
| MLX-002 | Complete 2026-05-09 | Audit `mlx_timestepper._apply_floors()` against the no-density-injection invariant with full timestep tests, not only direct helper tests. |
| MLX-003 | Complete 2026-05-09 | Fix or justify the no-op radial-coordinate expression in `compute_upf_voltage_flux()`. |
| MLX-004 | Blocked | Add trace links or quarantine labels for PF-1000 Akel/shot constants until accepted KR evidence exists. |
| MLX-005 | Complete 2026-05-09 | Keep PF-1000 endurance/probe tests as engineering regression only; do not promote them into acceptance gates without source closure. |
| MLX-006 | Complete 2026-05-09 | Audit MLX circuit back-EMF ownership; current engine path passes `back_emf=0.0`, so voltage-feedback claims need explicit status. |
| MLX-007 | Complete 2026-05-09 | Surface reduced MLX phase-model limits in API/GUI/export readiness so axial/radial/pinch-only runs are not presented as full Lee five-phase coverage. |
| MLX-008 | Complete 2026-05-09 | Define trust gates for MHD-derived coupling that require more than finite/positive fields before scientific claims use MHD-derived `Lp`, `dLdt`, or resistance. `evaluate_mhd_coupling_gate()` now requires phase eligibility, finite/positive/comparable `Lp`, finite `dLdt`, and finite/nonnegative resistance for engineering blend eligibility, while `mhd_coupling_gate` keeps scientific claims blocked by missing same-scope validation evidence. |
| MLX-009 | Complete 2026-05-11 | Correct cylindrical conservative source-term handling in Metal/MLX MHD paths. Radial momentum now uses the local-KR r-weighted form with `p_total` and inward toroidal hoop stress, theta momentum sign is corrected, and MLX source application no longer density-multiplies conserved source arrays a second time. |
| MLX-010 | Complete 2026-05-11 | Replace the fixed cross-field conduction ratio in `src/dpf/metal/mlx_transport.py::apply_thermal_conduction()`. When field components are available, MLX now computes an NRL electron-ion Coulomb-log and the Braginskii high-field perpendicular conductivity with coefficient `4.7`, caps it at `kappa_parallel`, and preserves isotropic fallback only when field components are absent. |

## Circuit/Snowplow Module

| ID | Status | Task |
| --- | --- | --- |
| CIR-001 | Complete 2026-05-09 | Add source-status notes and tests for density-weighted `CircuitCoupler` behavior; treat it as engineering scaffolding until local KR support exists. `CircuitCoupler` now exposes fail-closed authority metadata, engine summaries include `circuit_coupler_authority`, and trust-status records remain `not_validation_evidence` with scientific claims disabled. |
| CIR-002 | Complete 2026-05-09 | Reconcile or explicitly document CPU `r_s` versus MLX `r_p` radial inductance conventions. CPU and MLX snowplows now expose `radius_convention` metadata that identifies CPU shock-front-radius loading versus reduced-MLX piston-radius loading and marks them as not cross-backend equivalent validation evidence. |
| CIR-003 | Complete 2026-05-09 | Normalize or scope-separate CPU `r_pinch_min = 0.17a` versus MLX `0.13a`. CPU metadata labels `0.17a` as PF-1000/0.14-0.17-band scoped, while MLX metadata labels `0.13a` as a reduced deuterium gross boundary without reflected-shock coverage. |
| CIR-004 | Complete 2026-05-09 | Update stale snowplow docstrings/comments so circuit-facing current-factor scaling matches implementation. |
| CIR-005 | Blocked | Replace placeholder/xfailed waveform-current tests with KR-gated Akel digitization tests after review acceptance. |
| CIR-006 | Complete 2026-05-09 | Audit post-pinch resistance multipliers and assign source/provenance status before they support any validation claim. CPU snowplow now exposes `post_pinch_resistance_authority` metadata that labels the multipliers as empirical engineering continuity knobs with `source_status="multiplier_source_missing"` and `validation_status="not_validation_evidence"`. |
| CIR-007 | Complete 2026-05-09 | Strengthen `auto` MHD-coupler trust gating beyond any nonzero density field. |
| CIR-008 | Complete 2026-05-09 | Preserve the geometry/loading boundary: `L_coeff` remains unscaled, while circuit-facing helpers apply `fc` and `fcr_eff`. |
| CIR-009 | Complete 2026-05-11 | Fix `CircuitCoupler` inductive-EMF ownership. The coupler now clamps `dLp_dt` by equivalent `I*dLp/dt` voltage but returns `back_emf=0.0`, because `RLCSolver` already contains the inductive `-I*dLp/dt` term. |
| CIR-010 | Complete 2026-05-11 | Apply Lee radial `fcr` circuit-loading in `src/dpf/validation/lee_model_comparison.py`. The validation Lee model now carries `radial_current_fraction`, applies `lee_fcr` device overrides, uses `fcr` for radial inductance, radial `dLp/dt`, radial/reflected force, and frozen/post-crowbar radial inductance, and reports `fcr` in metadata. |
| CIR-011 | Complete 2026-05-23 | Close the SS15 power-port evidence-bound slice without promoting acceptance. `field_power_diagnostics_from_cylindrical_state()` now records the axisymmetric `J·E` volume-integral method, load-positive sign convention, cell-centered time-centering, and Poynting/`J·E` residual metrics; Tier-3 circuit-energy evidence validates interval labels; Phase 4-B power-port packets require reviewed residual metadata before the review gate is present. |

## Collision/Transport Module

| ID | Status | Task |
| --- | --- | --- |
| COL-001 | Complete 2026-05-11 | Correct Braginskii perpendicular conductivity against the NRL formulary high-field coefficient. `src/dpf/collision/spitzer.py` now preserves the unmagnetized limit while matching the `4.7` high-field coefficient, and direct high-field helpers in CPU/Metal transport use `4.7` instead of `4.66`. |
| COL-002 | Blocked | Define the public `nu_ee` convention before editing `src/dpf/collision/spitzer.py::nu_ee()`. The local formulary has multiple electron-electron collision/relaxation rows, while current tests expect `sqrt(2) * nu_ei`; changing it without an API convention would create a different ambiguity. |

## Diagnostics Module

| ID | Status | Task |
| --- | --- | --- |
| DIA-001 | Complete 2026-05-09 | Fix or quarantine the beam-tracker energy-to-voltage path before using `BeamTracker` yield estimates. |
| DIA-002 | Complete 2026-05-09 | Define a real `div_B` diagnostic with geometry and grid-spacing treatment, or mark current HDF5 `max_div_B` as a rough array metric. |
| DIA-003 | Blocked | Add local KR support and accepted formulas for Thomson, nTOF spectrum, x-ray filter/emissivity, regime, instability, plasmoid, shear, and runaway helpers. |
| DIA-004 | Blocked | Review anisotropy and beam-target dwell/transit assumptions beyond the Lee/Saw per-shot yield formula. |
| DIA-005 | Complete 2026-05-10 | Build a diagnostics evidence manifest that classifies every formula/output as accepted, blocked-by-review, missing, engineering-probe, or synthetic-only. `src/dpf/diagnostics/evidence_manifest.py` now covers every diagnostics module/public symbol and fails closed with no accepted validation entries. |
| DIA-006 | Complete 2026-05-10 | Split engineering smoke tests from source-backed physics validation tests. `src/dpf/diagnostics/test_lanes.py` and pytest collection markers now classify diagnostics tests as engineering-smoke, source-component-check, source-blocked, or synthetic-only; no diagnostics test is currently in the source-backed validation lane. |
| DIA-007 | Complete 2026-05-09 | Update stale diagnostics troubleshooting notes after the audit is complete. |
| DIA-008 | Blocked; packet scaffold added 2026-05-23 | Add same-scope diagnostic validation packets for neutron yield, timing, spectrum, anisotropy, detector response, and uncertainty. SS18 now adds a mechanism-separated fail-closed neutron diagnostic packet and validator for yield, timing, spectrum, anisotropy, detector/activation response, diagnostic mapping, and uncertainty blockers, but it deliberately remains non-accepting until spectrum, response matrix, UQ, comparator, and review certificate close. |
| DIA-009 | Complete 2026-05-11 | Correct NRL electron-ion Coulomb-log and Spitzer-resistivity usage in regime diagnostics. `magnetic_reynolds_number()` now uses centralized corrected resistivity, and `classify_regime()` uses the NRL electron-ion Coulomb-log branches rather than an electron-electron-like expression. |

## Radiation/Atomic/Neutrons Module

| ID | Status | Task |
| --- | --- | --- |
| RAD-001 | Blocked | Source-close line cooling coefficients or replace them with a KR-backed opacity/EOS/radiation packet. |
| RAD-002 | Blocked | Add branch-specific Bosch-Hale DD neutron/proton handling and tests tied to local tables and neutron-yield semantics. |
| RAD-003 | Blocked; packet scaffold added 2026-05-23 | Build a neutron validation packet covering yield, timing, spectrum, anisotropy, detector response, and uncertainty. SS18 now provides a fail-closed mechanism-separated neutron diagnostic packet plus validator, but no spectrum, detector-response matrix, uncertainty budget, review certificate, or acceptance claim is promoted. |
| RAD-004 | Blocked | Add p-B11 reactivity/yield source packets or keep p-B11 outputs permanently marked non-predictive. |
| RAD-005 | Complete 2026-05-10 | Add a QMF suppression derivation/source packet or quarantine QMF as diagnostic-only heuristic. QMF remains source-missing, but `qmf_model_metadata()` and `QMFDiag` now quarantine outputs as heuristic diagnostics with `validation_status="not_validation_evidence"` and no validation/design-claim support. |
| RAD-006 | Complete 2026-05-09 | Add tests that enforce conservative metadata/status labels for all unverified radiation and yield paths. |
| RAD-007 | Blocked | Source-close ionization/ablation constants and rates field by field. |
| RAD-008 | Complete 2026-05-09 | Reconcile CPU and MLX line-radiation provenance language so both surfaces expose the same uncertainty status. |
| RAD-009 | Complete 2026-05-11 | Correct NRL formulary radiation/atomic mismatches: Eq. 30 bremsstrahlung unit and `Z_eff` handling in coronal radiation, Eq. 33 recombination-radiation coefficient, Eq. 34 cyclotron sign invariance, and Eq. 13 radiative recombination bracket. |
| RAD-010 | Blocked; guardrail added 2026-05-11 | Source-close opacity/FLD/Kramers/Rosseland radiation-transport behavior before presenting it as formulary-backed. `radiation_transport_model_metadata()` now marks the current FLD/Rosseland/Kramers path as `rosseland_kramers_fld_source_packet_missing` and `not_validation_evidence`; the physics remains blocked until a local source packet exists. |
| RAD-011 | Complete 2026-05-11 | Add fail-closed source-status metadata for p-B11 diagnostics. `pb11_model_metadata()` now separates local-NRL-supported reaction/Q-value bookkeeping from missing reactivity-table source support and reports `validation_status="not_validation_evidence"`. |

## IO/Export Module

| ID | Status | Task |
| --- | --- | --- |
| IO-001 | Complete 2026-05-09 | Close/flush Well exporter inside engine normal and failure paths; add a regression where CLI/engine run emits a Well file without manual `engine.close()`. |
| IO-002 | Blocked | Ingest/review The Well schema into `KnowledgeReference/`, or relabel Well output as a local experimental adapter. |
| IO-003 | Complete 2026-05-09 | Add strict Well validator checks for scalar finiteness, geometry consistency, required attributes, energy evidence, and all-zero field detection. Current strict mode checks scalar histories, monotonic time, root provenance/classification, sanitized/non-finite labels, saturation-scale values, energy evidence, and all-zero magnetic fields. |
| IO-004 | Complete 2026-05-09 | Wire artifact classification through config/CLI/API into HDF5, Well output, manifests, and dataset manifests. Well HDF5 persists owner/distribution classification metadata; `dpf export-well` accepts classification flags; run config/API payloads drive engine HDF5, engine Well, run-manifest, batch Well, checkpoint HDF5, and batch dataset-manifest classification metadata. Certificate readiness/context propagation remains tracked under `VAL-010`. |
| IO-005 | Blocked | Quarantine or regenerate current local training HDF5; keep existing files only for negative/schema-regression tests. |
| IO-006 | Complete 2026-05-09 | Forward circuit scalars through the `src/dpf/io/well_exporter.py` adapter or explicitly document that the adapter exports field state only. |
| IO-007 | Complete 2026-05-09 | Fix or label cylindrical `grid_type` metadata in the full AI Well exporter. |
| IO-008 | Complete 2026-05-09 | Reconcile export baseline docs with the SRS draft where non-manifest export classification propagation remains incomplete. |

## AI/WALRUS Module

| ID | Status | Task |
| --- | --- | --- |
| AI-001 | Blocked | Ingest WALRUS/The Well/CATS/model-card/license/dataset records into `KnowledgeReference/` with hashes, source status, and review state. |
| AI-002 | Blocked | Keep current local WALRUS/HDF5 data out of validation and publication claims until defects are resolved and source/provenance packets exist. |
| AI-003 | Complete 2026-05-09 | Add strict validator mode for scalar finite checks, energy requirements, geometry/root consistency, all-zero B detection, saturation thresholds, monotonic time, and provenance manifests. |
| AI-004 | Complete 2026-05-09 | Make exporter fail or explicitly label non-finite data instead of silently zeroing fields; fix `grid_type`; write source/provenance/classification metadata. |
| AI-005 | Blocked | Claim real WALRUS inference only after checkpoint hash, version, license, source, and local inference behavior are recorded. |
| AI-006 | Complete 2026-05-09 | Split model reporting into `placeholder_loaded`, `real_model_loaded`, and `source_backed_model_loaded`. |
| AI-007 | Dependency blocked | Verify `_build_walrus_batch` channel/metadata ordering against the real WALRUS formatter; full check needs actual WALRUS dependencies/checkpoint. |
| AI-008 | Complete 2026-05-09 | Update stale AI/WALRUS docs and scripts that claim HDF5 generation or placeholder identity behavior contrary to current code. |

## Server/GUI/CLI Module

| ID | Status | Task |
| --- | --- | --- |
| SGC-001 | Complete 2026-05-09 | Align GUI/server/CLI/backend contracts for `mlx` and `hybrid`. |
| SGC-002 | Complete 2026-05-09 | Fix GUI TopBar time formatting so seconds from API/store are not displayed as nanoseconds. |
| SGC-003 | Complete 2026-05-09 | Make CLI/UI validation display source-authority status, not only PASS/FAIR/POOR from peak-current deviation. |
| SGC-004 | Complete 2026-05-09 | Decide whether API readiness is global project readiness or per-run/device readiness; current generic Akel blocker may confuse unrelated runs. |
| SGC-005 | Complete 2026-05-09 | Replace Gradio "validated", "publication-grade", "WORKING", and "97x demonstrated" claims with KR-supported labels or explicit Preview wording. |
| SGC-006 | Complete 2026-05-09 | Extend local-first audit to renderer HTML, CSP, and external assets. |
| SGC-007 | Complete 2026-05-09 | Align visible UI version/package version semantics. |
| SGC-008 | Complete 2026-05-09 | Label PF-1000 defaults and presets by source scope so broad PF-1000 values are not confused with Akel shot-12581 values. |
| SGC-009 | Complete 2026-05-23 | Add SS21 product-claim/release-posture guardrails. `docs/SS21_PRODUCT_CLAIM_SURFACE_RELEASE_DECISION_2026_05_23.md` records HONEST-BLOCKED / SOURCE-GATED PREVIEW; `README.md` exposes the fail-closed flags; `tests/test_ss21_product_claim_surface.py` prevents public-copy drift into accepted production first-principles/full-3D claims. Post-review fix/reverify recorded independent focused review PASS as approval of the honest-blocked wording only; all acceptance flags remain false. |

## Supplemental Physics Helpers

| ID | Status | Task |
| --- | --- | --- |
| PHX-001 | Complete 2026-05-11 | Add fail-closed metadata to uncovered physics helpers. `ablation_model_metadata()`, `two_temperature_model_metadata()`, `braginskii_viscosity_model_metadata()`, `nernst_model_metadata()`, `sheath_model_metadata()`, `anomalous_resistivity_model_metadata()`, and `civ_breakdown_model_metadata()` now mark their current role as engineering/scaffolded and `not_validation_evidence`. |
| PHX-002 | Blocked | Source-close electrode ablation efficiencies, pulse/fluence ranges, shielding, droplet/ejection assumptions, and impurity-mixing limits before using ablation for predictive high-Z or electrode-erosion claims. |
| PHX-003 | Blocked | Source-audit two-temperature relaxation and Braginskii ion-viscosity collision-time conventions against local NRL rows, including electron-ion equilibration, ion-ion Coulomb log, and `tau_i` units. |
| PHX-004 | Blocked | Promote/review Nernst and Ettingshausen coefficient sources before presenting thermomagnetic transport as source-closed DPF physics. |
| PHX-005 | Blocked; SS16 packet added 2026-05-23 | Promote/review anomalous-resistivity, LHDI/Buneman, CIV, Paschen, gas-coefficient, and startup/sheath source packets before using those helpers for restrike, flashover, or anomalous-resistance validation claims. `docs/SS16_STARTUP_BVP_EVIDENCE_PACKET_MATRIX_2026_05_23.json` now line-cites PF-1000 startup candidates and explicit blockers, but startup payload, preionization, UQ, and review remain non-promoting. |
| PHX-006 | Blocked | Keep Sedov/Auluck/GV/verification helpers in method-support lanes unless source-specific numerical constants, validation scopes, and same-scope evidence are accepted. |

## Project Process / Agent Operations

| ID | Status | Task |
| --- | --- | --- |
| AGT-001 | Complete 2026-05-11 | Add root `AGENTS.md` as the project-level operating contract for future agent work. It records required first reads, `KnowledgeReference/`-only science source rules, evidence-state vocabulary, non-promotion lanes, hard blockers, verification command matrix, module routing, delegation rules, and maintenance triggers. This is process scaffolding only and does not promote any scientific evidence. |
| AGT-002 | Complete 2026-05-23 | Add SS22 research/ops packaging for sustained work. `docs/SS22_RESEARCH_OPS_RUNBOOK_2026_05_23.md`, `docs/SS22_EVIDENCE_INDEX_2026_05_23.md`, `docs/SS22_LONG_RUN_RESEARCH_ROADMAP_2026_05_23.md`, `docs/SS22_FUTURE_SPRINT_QUEUE_2026_05_23.md`, and `docs/SS22_RESEARCH_OPS_PACKAGING_STATUS_2026_05_23.md` package the honest-blocked release posture, board cleanup plan, resource/scope/claim guards, and future sprint queue; `tests/test_ss22_research_ops_packaging.py` keeps those docs linkable and non-promoting. Post-review fix/reverify consumed independent review PASS, recorded the closed implementation/review lanes, and preserved the fail-closed claim posture. |
