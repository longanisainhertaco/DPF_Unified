# SS14 PF-1000 Same-Scope Source Packet Expansion (2026-05-23)
Authority: local `KnowledgeReference/` line-cited text only. `/Users/anthonyzamora/Desktop/heliosmatrix_kb` was used as retrieval/extraction support only, not as source authority.
Acceptance boundary: `accepted_runtime_claim=false`, `can_support_first_principles_acceptance=false`, `promotes_acceptance=false`.
Machine-readable matrix: `docs/SS14_PF1000_SAME_SCOPE_SOURCE_PACKET_MATRIX_2026_05_23.json`.
## Channel status summary
- `geometry`: `candidate` / `same_scope_candidate` — Geometry is line-cited across PF-1000 sources, but full runtime 3-D mask, uncertainty, and independent review remain open.
- `bank_circuit`: `candidate` / `same_scope_candidate` — Bank/circuit values are candidate targets; no full circuit waveform uncertainty, power-port evidence, or review certificate.
- `gas_fill`: `candidate` / `same_scope_candidate` — Gas-fill evidence is candidate; shot conditioning, pressure uncertainty, and scope-specific review remain missing.
- `current_waveform`: `candidate` / `same_scope_candidate` — Current evidence includes diagnostics and figure candidates but no digitized waveform, uncertainty, or accepted comparator.
- `startup`: `candidate` / `same_scope_candidate` — Startup/sheath-liftoff evidence is qualitative/candidate; no boundary-value initial condition packet or reviewed handoff.
- `density_history`: `candidate` / `same_scope_candidate` — Density history has PF-1000 diagnostic candidates; numeric density fields and reviewed digitization are missing.
- `em_field_history`: `candidate` / `same_scope_candidate` — Magnetic/current-sheath evidence is candidate and partly lower-energy; no full EM field history with uncertainty/review.
- `temperature_or_distribution_history`: `candidate` / `same_scope_candidate` — Temperature/distribution evidence remains mechanism/context level; no reviewed Te/Ti/distribution time history.
- `neutron_scalar_yield`: `candidate` / `same_scope_candidate` — Scalar yield candidates exist, but detector calibration, uncertainty propagation, and review are not closed.
- `neutron_timing`: `candidate` / `same_scope_candidate` — Timing evidence is candidate; raw traces/digitization, detector response, and uncertainty remain missing.
- `neutron_spectrum`: `candidate` / `same_scope_candidate` — No same-scope measured spectrum packet exists; 2.45 MeV and future ToF language are non-promoting candidates only.
- `neutron_anisotropy`: `candidate` / `same_scope_candidate` — Anisotropy evidence is candidate; angular detector response and uncertainty are not reviewed.
- `detector_response`: `candidate` / `same_scope_candidate` — Detector response has calibration/context candidates but no reviewed response matrix/uncertainty packet.
- `uncertainty_budget`: `blocked` / `same_scope_candidate` — No complete same-scope uncertainty budget covering source extraction, digitization, detector response, comparator, and model uncertainty exists.
- `review_certificate`: `blocked` / `same_scope_candidate` — No independent review certificate exists for SS14 matrix rows; all rows stay non-accepted.

## Transfer rows
- `pf1000_pf400_lee_model_current_fit`: `cross_scope_candidate`; promotes_acceptance=`false` — PF1000/PF400 Lee-model comparison includes PF-400 and computed quantities; useful transfer/method context but cannot promote PF-1000 same-scope acceptance.
- `lower_energy_pcs_current_scaling`: `candidate`; promotes_acceptance=`false` — PF-1000 source but lower capacitor-bank energy (250-500 kJ), below SS14 full-energy scope; non-promoting candidate for later transfer rule.
- `future_tof_spectrum_measurement`: `rejected`; promotes_acceptance=`false` — The cited source says future experiments will move probes for ToF spectra; it is not an existing same-scope neutron-spectrum measurement.

## Evaluate / Learn / Continue
- **Evaluate**: Matrix expands PF-1000 full/upper-energy candidates across 15 required channels, exact line-window quote validation, KnowledgeReference-only source-ref enforcement, broad-window rejection, transfer-row non-promotion, and fail-closed acceptance boundary. Post-review reverify passed: validator 0 issues, SS14 tests 9/9, source-truth exhaustion 0 open issues, strict module-vetting passed, JSON acceptance scan 0 true hits.
- **Learn**: Local KR has useful PF-1000 geometry, bank, gas, current, density, sheath/magnetic-probe, neutron yield/timing/anisotropy, and detector context, but no complete uncertainty budget or independent review certificate; neutron spectrum remains especially blocked. Reviewer requested no functional fixes; the validator was tightened to reject non-KR repo docs and overwide source windows so source authority remains fail-closed.
- **Continue**: SS15-SS18 child packets should target digitization/uncertainty closure by channel priority; keep acceptance flags false until a complete reviewed certificate stack exists.
