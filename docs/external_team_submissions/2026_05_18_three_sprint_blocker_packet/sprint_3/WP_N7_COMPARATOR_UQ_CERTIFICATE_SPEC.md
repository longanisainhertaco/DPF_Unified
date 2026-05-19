# WP-N7 Comparator / UQ / Certificate Design Specification

**Packet:** Sprint 3 parallel lane 6
**Output path:** `docs/external_team_submissions/2026_05_18_three_sprint_blocker_packet/sprint_3/WP_N7_COMPARATOR_UQ_CERTIFICATE_SPEC.md`
**Branch:** `codex/corpus`
**Audited HEAD:** `07fe76a`
**Date:** 2026-05-19
**Status:** `design_spec_not_implementation` — READ-ONLY research packet.

---

## 1. Purpose

This packet defines exactly what an external engineering review bundle must
contain for WP-N7 (comparator / UQ / certificate). It does not implement, does
not promote validation, and does not mark any channel `accepted`. Every physics
claim below cites a local `KnowledgeReference/` file (path + section/line).
Every codebase claim cites the source file.

The certificate gate is fail-closed by construction: see section 5.

---

## 2. Source-Backed Findings — What PF-1000 / Akel Comparator Data Exists

### 2.1 Akel 2021 (Radiation Physics and Chemistry 188, 109633)

**File:** `KnowledgeReference/radiation-physics-and-chemistry-188-2021-109633.md`
**Ingestion status:** text extracted; figures/tables are NOT target-extracted.
**KR validation status:** not declared (no `KR ingestion status` header found in
this file — it uses the older table-of-contents format without the standard
header block).

**What is present in the text:**

- Device geometry (text): cathode 12 × 8 cm SS tubes, anode tube 231 mm
  diameter, electrodes 480 mm long. (lines 111–114)
- Bank and operational parameters for the Akel shot set: C₀ = 1332 µF, 16 kV,
  170.5 kJ, p₀ = 1.05 and 1.20 Torr D₂. (lines 115–117)
- Lee model fitted parameters for shot 12581 explicitly: fm = 0.17, fc = 0.70,
  fmr = 0.25 (corrected to 0.26 in Table 1), fcr = 0.75. (lines 271–274)
- Scalar I_peak (text only): ~1165 kA (shot 12581). (line 280)
- Pinch geometry (Lee model output, not measured): r_p = 2.40 cm, z_p = 18.2 cm.
  (Table 2 header, lines 585–607)
- Maximum pinch ion density (Lee model output): n_i = 1.7 × 10²³ m⁻³ for shot
  12581. (Table 2, lines 598–607)
- Scalar neutron yield (measured): (6.1 ± 0.2) × 10⁹ n/sh for shot 12581.
  (lines 287–288)
- Shot-series yield ranges (measured): 3 × 10⁸ to 6.1 × 10⁹ n/sh at 1.2 Torr;
  1.7 × 10⁸ to 1.11 × 10¹⁰ n/sh at 1.05 Torr. (lines 870–876)
- Series average yields (measured): (1.75 ± 0.2) × 10⁹ n/sh at 1.2 Torr (8
  shots); (2.29 ± 0.2) × 10⁹ n/sh at 1.05 Torr (16 shots). (lines 866–873)
- Timing: breakdown to derivative dip ~7 µs; constriction starts 50–100 ns after
  dip; secondary plasmoids 100–200 ns after dip. (lines 133–136)
- Neutron yield measurement uncertainty declared: ±0.2 × 10⁹ (systematic, silver
  activation counters calibrated with Am-Be). (lines 127–132)
- Timing channel uncertainty: ~3–5 ns single-discharge. (lines 136–138)
- Detector geometry (text only): three scintillators at 7 m, 0°/90°/180°.
  (lines 119–122)

**What is NOT present or is cross-scope:**

- No digitized time-series current waveform (figures not target-extracted).
- No spatially resolved density history.
- No electron temperature T_e history.
- No ion temperature T_i history measured directly (text explicitly states T_i
  was not measured in this campaign, line 816 of scholz-2007-pf1000-part2-jphysd.md).
- No magnetic field (B_θ) spatial or time history.
- No neutron spectrum (TOF spectrum not in this paper).
- No neutron anisotropy measurement for the 16 kV shot set.
- No startup / breakdown / preionization data for the 16 kV shots.
- Pinch density and geometry are Lee model outputs, not independent measurements.

### 2.2 Scholz 2006 (Nukleonika 51(1):79–84)

**File:** `KnowledgeReference/scholz-2006-pf1000-mega-joule.md`
**Ingestion status:** no `KR ingestion status` header (older format). No
`source_available_not_target_extracted` declaration.

**What is present:**

- Qualitative current-waveform comparison figures (Fig. 7 referenced in text):
  discrepancy noted between computed and experimental current traces, attributed
  to circuit parameters or model limits. (lines 328–332)
- High-speed frame camera images of radial collapse at p₀ = 4 hPa, U₀ = 33 kV,
  I_max = 1.7 MA (Fig. 5). This is a DIFFERENT operating point — not Akel 16 kV.
- Neutron yield (different scope): regular 10¹⁰–10¹¹ n/shot; maximum ~3.5 × 10¹¹
  at MJ-scale operation. (lines 456–465)
- Streak camera images distinguishing high/low yield shots (Fig. 6).
- Neutron timing correlation: first pulse 20–30 ns before X-ray peak; FWHM
  50–70 ns. (lines 407–418)

**Scope mismatch:** Scholz 2006 describes 0.5–1 MJ operation (33–40 kV), not the
Akel 16 kV / 170.5 kJ shots. Data from this source CANNOT serve as a same-scope
comparator for the Akel 16 kV shot set without a reviewed transfer rule.

### 2.3 Scholz / Gribkov 2007 (J. Phys. D 40, 3592 — Part II)

**File:** `KnowledgeReference/scholz-2007-pf1000-part2-jphysd.md`
**Ingestion status:** no `KR ingestion status` header (older format).

**What is present:**

- Ion density (estimated from pinch radius): n_i ≈ 0.8 × 10¹⁹ cm⁻³ at first
  compression. (line 786)
- Ion temperature (inferred, not directly measured): T_i ≈ 1.3 keV at first
  compression; T_i_eff ≥ 4 keV required for second pulse yield. (lines 786–814)
  Text explicitly states: "temperatures which were not measured here directly (and
  have not been reliably measured in all previous experiments)." (lines 815–817)
- Pinch dimensions: R_p ≈ 0.45 cm, h_p ≈ 10 cm, τ ≈ 150 ns (first compression).
  (line 787)
- Scope: This paper describes PF-1000 at full energy (≈1 MJ, not 16 kV).

**Scope mismatch:** Gribkov/Scholz 2007 is NOT the Akel 16 kV shot set. A
reviewed transfer rule is required before any of these numbers serve as a
same-scope comparator.

### 2.4 Sixteen-Frame Interferometer (Zielinska 2011)

**File:** `KnowledgeReference/sixteenframe-interferometer-for-a-study-of-a-pinch-dynamics-in-pf1000-device-f8dc9d1b.md`
**KR ingestion status:** `text_parity_extracted_review_needed`
**Validation status:** `source_available_not_target_extracted`

**What is present:**

- Demonstration of 16-frame interferometric density imaging on PF-1000; one shot
  at 2.6 hPa with Y_n = 1.3 × 10¹¹. (lines 129–132)
- Confirms electron density is accessible via Mach-Zehnder interferometry.
- Delay range 0–220 ns, 16 frames per shot. (line 86)

**Scope mismatch:** The shot reported (2.6 hPa, high yield) is NOT the Akel 16 kV
1.05–1.2 Torr shot set. Density target values are NOT target-extracted.

### 2.5 Characteristics of Closed Currents and Magnetic Fields (Kubes 2020)

**File:** `KnowledgeReference/characteristics-of-closed-currents-and-magnetic-fields-outside-the-dense-pinch-column-in-a-40d59f2d.md`
**KR ingestion status:** `text_parity_extracted_review_needed`
**Validation status:** `source_available_not_target_extracted`

**What is present:**

- Quantitative magnetic field and current measurements at PF-1000 using a
  Rogowski coil and Hall probes (scope not confirmed as 16 kV / Akel shot set).
- Figures referenced but NOT target-extracted.

**Scope mismatch:** Cannot confirm this is the Akel 16 kV shot set without
further review. NOT accepted as a same-scope magnetic field comparator.

### 2.6 Experimental Study of Plasma Current Sheath Structure (Krauz 2012)

**File:** `KnowledgeReference/experimental-study-of-the-structure-of-the-plasma-current-sheath-on-the-pf-1000-facility-705bcc83.md`
**KR ingestion status:** `text_parity_extracted_review_needed`
**Validation status:** `source_available_not_target_extracted`

**What is present:** Detailed current-sheath structure measurements on PF-1000.
Figures NOT target-extracted. Scope not confirmed as Akel 16 kV shots.

### 2.7 Summary of KR Evidence State

The `same_scope.py` module explicitly codifies what the codebase already knows
about same-scope channel availability. Citations are:
`src/dpf/first_principles/same_scope.py` — `BLOCKING_SAME_SCOPE_CHANNELS`,
`PF1000_AKEL_TEXT_SUPPORTED_CHANNELS`, `OTHER_SCOPE_SOURCE_GROUPS`.

The `comparator_uq.py` module codifies the same for the comparator/UQ side:
`src/dpf/first_principles/comparator_uq.py` — `PF1000_AKEL_TEXT_SUPPORTED_CHANNELS`,
`OTHER_SCOPE_SOURCE_GROUPS`.

---

## 3. Supported / Candidate / Blocked Table

Per comparator channel and per certificate component.

### 3.1 Comparator Channels

| Channel | Status | Evidence source | Notes |
|---|---|---|---|
| `current_waveform` (scalar I_peak) | **candidate** | `KnowledgeReference/radiation-physics-and-chemistry-188-2021-109633.md` lines 280, 344–353 | Text only; figures not target-extracted; shot 12581 |
| `current_waveform` (digitized time-series) | **blocked** | Same KR | Figures 1–4 present but not target-extracted; no digitized CSV |
| `current_dip` (timing, text) | **candidate** | Same KR lines 133–136 | ~7 µs to dip; text only |
| `phase_timing` (text) | **candidate** | Same KR lines 133–138 | 50–100 ns constriction onset; 3–5 ns uncertainty |
| `spatial_density` (history) | **blocked** | `KnowledgeReference/sixteenframe-interferometer-for-a-study-of-a-pinch-dynamics-in-pf1000-device-f8dc9d1b.md` | Source available; wrong scope; not target-extracted |
| `magnetic_em_field` (B_θ history) | **blocked** | `KnowledgeReference/characteristics-of-closed-currents-and-magnetic-fields-outside-the-dense-pinch-column-in-a-40d59f2d.md` | Scope unconfirmed; not target-extracted |
| `temperature` (T_e) | **blocked** | No KR source for Akel 16 kV T_e measurement | Not measured in that campaign |
| `temperature` (T_i) | **blocked** | `KnowledgeReference/scholz-2007-pf1000-part2-jphysd.md` lines 815–817 | Explicitly not measured; inferred only; wrong scope |
| `neutron_scalar_yield` | **candidate** | `KnowledgeReference/radiation-physics-and-chemistry-188-2021-109633.md` lines 287–288, 866–876 | (6.1 ± 0.2) × 10⁹ n/sh (shot 12581); series ranges present; ±0.2 × 10⁹ uncertainty declared |
| `neutron_timing` (history) | **blocked** | `KnowledgeReference/scholz-2006-pf1000-mega-joule.md` lines 407–418 | Wrong scope (MJ-scale); not Akel 16 kV |
| `neutron_spectrum` | **blocked** | No same-scope KR source | Not present for Akel 16 kV shots |
| `neutron_anisotropy` | **blocked** | `KnowledgeReference/scholz-2006-pf1000-mega-joule.md` lines 436–444 | Wrong scope |
| `detector_activation_response` (text geometry) | **candidate** | `KnowledgeReference/radiation-physics-and-chemistry-188-2021-109633.md` lines 119–132 | Scintillators at 7 m, 0°/90°/180°; Am-Be calibration described; not digitized |
| `numerical_fidelity` | **candidate** | `src/dpf/first_principles/numerical_fidelity.py` | Code exists; requires production run and review |
| `physics_fidelity` | **candidate** | `src/dpf/first_principles/comparator_uq.py` | Code exists; requires production run |

### 3.2 Certificate Components

The canonical list of required certificate channels comes from
`src/dpf/first_principles/certificate_gate.py`, `REQUIRED_CERTIFICATE_CHANNELS`.

| Certificate Channel | Status | Blocking reason |
|---|---|---|
| `run_manifest_hash` | **candidate** | Manifest schema exists (`manifest.py`); requires production run with complete provenance |
| `evidence_packet_hashes` | **blocked** | No accepted evidence packets exist |
| `validation_scope_and_source_scope` | **blocked** | Same-scope packet remains `blocked_same_scope_source_packet_not_available` |
| `package_native_execution_proof` | **candidate** | Runner exists (`runner.py`); requires production run |
| `same_scope_source_packet_accepted` | **blocked** | `build_same_scope_source_packet()` always returns `blocked_same_scope_source_packet_not_available` |
| `waveform_phase_packet_accepted` | **blocked** | No digitized same-scope current waveform |
| `spatial_field_temperature_packet_accepted` | **blocked** | No same-scope density / T_e / T_i / B_θ time histories |
| `neutron_authority_packet_accepted` | **blocked** | WP-N6 not yet delivered |
| `comparator_uq_packet_accepted` | **blocked** | `build_comparator_uq_packet()` returns `blocked_comparator_uq_matrix_not_available` |
| `numerical_fidelity_packet_accepted` | **candidate** | Code exists; requires convergence run and review |
| `physics_closure_packet_accepted` | **blocked** | WP-N5: research_packet_delivered; runtime_packet_not_delivered; accepted_packet_not_delivered |
| `limiter_zero_or_physical_bounds_packet` | **candidate** | `limiter_readiness.py` exists; requires review |
| `power_port_packet_accepted` | **blocked** | WP-N1B terms II/IV/V/VI blocked pending reviewed Sigma_p geometry (WP-N3 runtime_packet_not_delivered) |
| `startup_packet_accepted` | **blocked** | WP-N2: research_packet_delivered; runtime_packet_not_delivered; accepted_packet_not_delivered |
| `dimensionality_handoff_packet_accepted` | **blocked** | Handoff packet requires review |
| `reduced_model_rejection_proof` | **blocked** | No test proves rejection of Lee-model or snowplow-only runs |
| `reviewer_metadata` | **blocked** | No external reviewer assigned |
| `accepted_review_status` | **blocked** | No review completed |
| `comparator_metrics_and_uq_ids` | **blocked** | No accepted same-scope targets |
| `requirement_links` | **candidate** | `DPF_REQUIREMENTS_BASELINE.md` exists; not linked to accepted evidence |
| `commands_and_versions` | **candidate** | `manifest.py` `REQUIRED_PROVENANCE_FIELDS` includes `command_argv`; requires production run |
| `release_label` | **blocked** | Hard-coded `engineering_candidate_not_releasable_for_first_principles_claim` until all channels pass |
| `release_decision` | **blocked** | Hard-coded `do_not_release_first_principles_claim` |
| `negative_test_draft_evidence` | **blocked** | Not yet implemented |
| `negative_test_blocked_evidence` | **blocked** | Not yet implemented |
| `negative_test_cross_scope_evidence` | **blocked** | Not yet implemented |
| `negative_test_missing_uq` | **blocked** | Not yet implemented |
| `negative_test_missing_review` | **blocked** | Not yet implemented |
| `negative_test_hidden_limiter` | **blocked** | Not yet implemented |
| `negative_test_app_only_or_reduced_model_fallback` | **blocked** | Not yet implemented |
| `certificate_artifact_hash` | **blocked** | Cannot hash a non-existent certificate |

**Per-status counts (certificate channels):** supported/candidate = 7, blocked = 22.

**Per-status counts (comparator channels):** candidate = 6, blocked = 9.

---

## 4. Runtime Fields Required

The review bundle must include a manifest serialized by
`src/dpf/first_principles/manifest.py`. Every field listed in
`REQUIRED_PROVENANCE_FIELDS` must be non-empty for the manifest to be
certificate-eligible (`has_complete_provenance()` must return `True`).

### 4.1 Required Provenance Fields (from `manifest.py`)

```
command_argv               — exact sys.argv used to produce the run
git_commit                 — HEAD SHA at run time
source_truth_index_sha256  — SHA-256 of the KnowledgeReference index
source_packet_hashes       — dict[source_id, sha256] for each cited KR file
input_deck_sha256          — SHA-256 of the SimulationConfig / input deck JSON
artifact_schema_version    — must be "first_principles_artifact_v1"
artifact_generation_commit — HEAD SHA at artifact generation time
```

### 4.2 Required Additional Runtime Fields for the Bundle

The following fields are not yet fully wired in the runtime. They are required
by the certificate gate but have no current production path.

| Field | Where it goes | Current status |
|---|---|---|
| `dirty_worktree` | `manifest.dirty_worktree` | `git_provenance()` in `manifest.py` — wired, soft-fail |
| `stored_magnetic_energy_delta_J` | `conservation.power_port` telemetry | Wired (WP-N1B, Sprint 2) |
| `stored_electric_energy_delta_J` | `conservation.power_port` telemetry | Wired (WP-N1B, Sprint 2) |
| `Sigma_p` face packet (B, E, J, v, eta, dS) | `power_port` terms II/IV/V/VI | Blocked — WP-N3 / Sprint 3 |
| `comparator_scope_id` | Certificate `validation_scope_and_source_scope` | Not emitted by runtime |
| `accepted_same_scope_target_hashes` | `evidence_packet_hashes` | No accepted targets |
| `uq_budget` (measurement / model / numerical / closure / detector) | `comparator_uq` packet | Not computed |
| `negative_test_run_ids` | All seven `negative_test_*` certificate channels | Not implemented |
| `reviewer_id` and `review_certificate_sha256` | `reviewer_metadata`, `accepted_review_status` | No external reviewer |

### 4.3 UQ Budget Components Required

The review bundle must carry an uncertainty budget for every observable that
reaches an accepted same-scope target. The `comparator_uq.py` module defines the
required channels: `measurement_uncertainty_by_observable`,
`model_uncertainty_by_observable`, `numerical_uncertainty_by_observable`,
`closure_sensitivity_uncertainty`, `detector_response_uncertainty`,
`shot_to_shot_uncertainty_or_scope_rule`, `uq_propagation_method`.

For the Akel 16 kV shot set, the only declared measurement uncertainty available
from KR is the neutron yield ±0.2 × 10⁹ (absolute) and timing ±3–5 ns. All
other UQ components are blocked by missing same-scope data.

---

## 5. Missing Parameters — Comparator Targets and UQ Inputs with No Source

The following are required but have NO usable same-scope source in the current
`KnowledgeReference/` corpus:

| Missing parameter | Impact | Nearest available source | Scope gap |
|---|---|---|---|
| Digitized current waveform (time series, shot 12581) | Blocks `waveform_phase_packet_accepted` and `accepted_digitized_current_waveform` | Figs. 1–4 in `radiation-physics-and-chemistry-188-2021-109633.md` | Figures not target-extracted |
| Spatially resolved electron density (n_e(r,z,t)) | Blocks `density_spatial_history`, `spatial_field_temperature_packet_accepted` | Interferometer KR present but wrong scope and not target-extracted | Scope mismatch |
| Electron temperature T_e(r,z,t) | Blocks `electron_temperature_history` | No PF-1000 Akel 16 kV T_e measurement in KR | Not measured |
| Ion temperature T_i (measured directly) | Blocks `ion_temperature_or_distribution_history` | Gribkov/Scholz 2007 inferred T_i at wrong scope | Not measured for 16 kV set |
| Magnetic field B_θ(r,z,t) | Blocks `em_field_history` | Kubes 2020 KR present; scope unconfirmed; not target-extracted | Scope unconfirmed |
| Neutron timing history (shot-specific pulse shape) | Blocks `neutron_timing_history` | Scholz 2006 timing only (wrong scope) | Scope mismatch |
| Neutron energy spectrum (TOF) | Blocks `neutron_spectrum` | No same-scope KR entry | Not present |
| Neutron anisotropy (0°/90°/180° ratio for 16 kV shots) | Blocks `neutron_anisotropy` | Scholz 2006 anisotropy (wrong scope) | Scope mismatch |
| Startup / preionization data | Blocks `startup_breakdown_preionization` | No KR entry for Akel 16 kV breakdown | Not present |
| Independent density measurement (not Lee output) | Required for `density_spatial_history` | Table 2 in Akel 2021 gives Lee model n_i, not measured | Lee output only |
| Pinch T_e, T_i from spectroscopy | Required for temperature comparator | None in KR for this shot set | Not present |
| Shot-to-shot yield spread model | Required for `shot_to_shot_uncertainty_or_scope_rule` UQ channel | Series ranges in Akel 2021 text support candidate; no model | Partial text only |

---

## 6. Proposed Tests and Fail-Closed Negative Controls

### 6.1 Certificate Fail-Closed Invariants (from `certificate_gate.py`)

The certificate gate (`build_first_principles_certificate_gate_packet()`) is
already fail-closed by construction:

- `can_write_accepted_certificate: False` always (hard-coded).
- `can_release_first_principles_claim: False` always.
- `can_support_first_principles_acceptance: False` always.
- Any upstream packet with status starting with `blocked`, `candidate`, or
  `rejected` is a `BLOCKING_UPSTREAM_STATUSES` entry.

These invariants must be tested by dedicated negative tests. The following tests
are proposed and do not yet exist:

| Test ID | Scenario | Expected behavior |
|---|---|---|
| N7-NEG-01 | Certificate submitted with a `blocked` same-scope packet | Gate returns `blocked_first_principles_certificate_not_available`; `can_write_accepted_certificate` is `False` |
| N7-NEG-02 | Certificate submitted with a `candidate` comparator-UQ packet | Same gate rejection; `comparator_uq_packet_accepted` channel listed in `missing_acceptance_channels` |
| N7-NEG-03 | Certificate submitted with no `reviewer_metadata` | Gate rejects; `reviewer_metadata` in `missing_acceptance_channels` |
| N7-NEG-04 | Certificate submitted with cross-scope evidence and no transfer rule | Gate rejects; `validation_scope_and_source_scope` missing |
| N7-NEG-05 | Certificate submitted with missing `run_manifest_hash` | Gate rejects; channel in `missing_acceptance_channels` |
| N7-NEG-06 | Certificate submitted with any upstream packet having `can_support_first_principles_acceptance: True` | Gate must reject — this is currently impossible by `manifest.py` `__post_init__` enforcement, but the test must confirm it |
| N7-NEG-07 | Comparator UQ packet built with a text-only yield scalar promoted to `accepted_same_scope_target` | `build_comparator_uq_packet()` must leave `accepted_same_scope_target_registry` in `missing_acceptance_channels` |
| N7-NEG-08 | Certificate submitted when `dirty_worktree: True` | Artifact linter check C8 must flag it; certificate gate must note it in `reviewer_metadata` |
| N7-NEG-09 | Same-scope packet submitted with only `PF1000_AKEL_TEXT_SUPPORTED_CHANNELS` accepted | `build_same_scope_source_packet()` must still return `blocked_same_scope_source_packet_not_available`; none of `BLOCKING_SAME_SCOPE_CHANNELS` can flip to accepted by text alone |
| N7-NEG-10 | Lee model scalar outputs (n_i from Table 2 of Akel 2021) promoted as `density_spatial_history` | Must be rejected; Lee model outputs are not independent measurements |

### 6.2 Artifact Linter Negative Controls (from `audit_first_principles_artifacts.py`)

The existing linter implements checks C1–C8. The following linter behaviors must
be verified:

- C6: any artifact containing `can_support_first_principles_acceptance: true`
  fails the linter (exit non-zero).
- C7: artifact with `manifest.provenance_complete: false` fails C7.
- C8: artifact generated from a dirty worktree fails C8.

These are tested in `tests/test_first_principles_artifact_linter.py`. No new
linter checks are proposed in this packet; the existing checks are sufficient for
the certificate pathway described here.

---

## 7. Exact Implementation Recommendations

### 7.1 Review Bundle Contents

An external engineering review bundle for the WP-N7 gate MUST contain the
following files, all generated by a single production run from a clean worktree
at a tagged HEAD:

```
review_bundle/
  manifest.json            — FirstPrinciplesRunManifest.to_dict(); provenance_complete: true
  conservation_ledger.json — FirstPrinciplesConservationLedger from the run
  same_scope_packet.json   — build_same_scope_source_packet() output
  comparator_uq_packet.json — build_comparator_uq_packet() output
  certificate_gate.json    — build_first_principles_certificate_gate_packet() output
  waveform_comparator.json — build_waveform_phase_packet() output (when implemented)
  spatial_field_temp.json  — spatial_field_temperature_packet output (when implemented)
  neutron_authority.json   — WP-N6 neutron authority packet (when delivered)
  numerical_fidelity.json  — numerical_fidelity packet
  negative_tests/
    N7-NEG-01.json ... N7-NEG-10.json  — per negative-test result records
  artifact_linter_report.txt — stdout of audit_first_principles_artifacts.py over all .json
  README.md                — contents, SHA-256 of every file, review checklist
```

Every `.json` file in the bundle must have its SHA-256 recorded in
`certificate_gate.json` under `evidence_packet_hashes`.

### 7.2 Certificate Gate Contract

The certificate gate defined in `src/dpf/first_principles/certificate_gate.py`
is the authoritative pass/fail arbiter. Its contract:

1. Accepts a `declared_scope` string — must exactly identify the shot set (e.g.
   `"pf1000_akel_16kv_1p2torr_deuterium_shot_12581"`).
2. Accepts `accepted_channels` — only channels for which accepted artifacts with
   matching scope exist.
3. Accepts `upstream_packets` — all sub-packets indexed by canonical name.
4. Returns a packet where:
   - `status` is always `blocked_first_principles_certificate_not_available` until
     every channel in `REQUIRED_CERTIFICATE_CHANNELS` is accepted.
   - `can_write_accepted_certificate` is always `False` unless all 29 channels
     pass.
   - `can_release_first_principles_claim` is always `False`.
   - `upstream_certificate_blockers` lists every packet that blocks the gate.

5. A gate packet is NOT accepted unless:
   - All 29 `REQUIRED_CERTIFICATE_CHANNELS` appear in `accepted_channels`.
   - All 11 `REQUIRED_UPSTREAM_PACKET_CHANNELS` have upstream status that passes
     `_status_is_accepted_for_certificate()`.
   - All 7 `REQUIRED_NEGATIVE_TEST_CHANNELS` appear in `accepted_channels`.
   - An external reviewer is named in `reviewer_metadata` and
     `accepted_review_status` is `"accepted"`.
   - `same_scope_source_packet_accepted` is accepted — which requires all
     `BLOCKING_SAME_SCOPE_CHANNELS` to be non-blocked.

### 7.3 Manifest Provenance Requirements

Implement a production-run wrapper that calls `git_provenance()`, collects
`sys.argv`, hashes the input deck with `sha256_of_file()`, hashes the
KnowledgeReference index, and calls
`source_packet_hashes_from_references()` for all cited sources. The wrapper
must call `mx.eval()` before reading any MLX arrays for hashing. The resulting
manifest must have `has_complete_provenance()` return `True` before the bundle
is submitted.

### 7.4 UQ Propagation Method Required

The `uq_propagation_method` channel requires a documented propagation approach.
The recommended method for the comparator/UQ packet is:

- Measurement uncertainty: quote directly from KR source (Akel 2021 ±0.2 × 10⁹
  for neutron yield; ±3–5 ns for timing). Any other channel requires
  target-extracted data from the same-scope source.
- Model uncertainty: parametric sensitivity sweep over fc/fm within the Optuna
  calibration bounds — requires production run output.
- Numerical uncertainty: grid-independence study result from CVG-03 (verified at
  0.03% between 32×64 and 64×128) is a candidate; requires same-scope geometry
  (WP-N3) to confirm the production grid is resolved.
- Closure sensitivity: blocked until WP-N5 closure registry is delivered.
- Detector response: blocked until neutron detector forward model is implemented
  (WP-N6).

No propagated combined UQ result can be reported while any component is blocked.

---

## 8. Explicit "Do Not Promote" Notes

1. **Validation is blocked.** The `same_scope.py` module returns
   `blocked_same_scope_source_packet_not_available` for every call. The
   `comparator_uq.py` module returns `blocked_comparator_uq_matrix_not_available`.
   The `certificate_gate.py` module returns
   `blocked_first_principles_certificate_not_available`. None of these statuses
   may be overridden by this design spec.

2. **Text-supported scalars are not acceptance evidence.** The channels in
   `PF1000_AKEL_TEXT_SUPPORTED_CHANNELS` (in both `same_scope.py` and
   `comparator_uq.py`) — including scalar I_peak, scalar neutron yield, Lee model
   fit parameters, detector geometry text — are reference material only. They
   cannot flip any `BLOCKING_SAME_SCOPE_CHANNELS` channel to accepted.

3. **Lee model outputs are not independent measurements.** Pinch density n_i and
   pinch dimensions from Table 2 of Akel 2021 are Lee model outputs, not
   interferometric or spectroscopic measurements. They cannot satisfy
   `density_spatial_history` or `temperature` comparator channels.

4. **Cross-scope sources require a reviewed transfer rule.** All PF-1000 sources
   that are NOT the Akel 16 kV, 1.05–1.2 Torr, Sept–Oct 2018 shot set are
   classified in `OTHER_SCOPE_SOURCE_GROUPS` in both `same_scope.py` and
   `comparator_uq.py`. They are usable only for requirements and schema
   definition. A reviewed transfer rule with all `TRANSFER_RULE_REQUIRED_CHANNELS`
   satisfied is required before cross-scope data can contribute to acceptance.

5. **`can_support_first_principles_acceptance` must remain `False`.** The
   `manifest.py` `__post_init__` raises `ValueError` if this flag is `True`. The
   artifact linter (C6) rejects any artifact where this flag appears `True`
   anywhere in the JSON tree. No parallel-lane deliverable in this packet changes
   that constraint.

6. **Sprint 2.2 files are frozen.** This packet does not touch any file listed in
   the "Sprint 2.2 files" section of
   `docs/FIRST_PRINCIPLES_CODEX_AUDIT_SPRINT2_FOLLOWUP_2026_05_19.md` §
   "Parallel Work Allowed Outside Sprint 2.2".

---

## 9. Blocker Summary

| Blocker | Owner sprint | Gate unblocked by |
|---|---|---|
| Digitized same-scope current waveform | Sprint 3 / WP-N3 comparator data | Target extraction from Akel 2021 Figs. 1–4 |
| Spatial density history | Sprint 3 + new measurement campaign or KR source | Same-scope interferometry target extraction |
| T_e, T_i histories | New measurement campaign or new KR source | No existing same-scope source |
| B_θ field history | KR review of Kubes 2020 scope + target extraction | Scope confirmation + extraction |
| Neutron spectrum | New measurement campaign or new KR source | No same-scope source |
| Neutron anisotropy (16 kV set) | New KR source or transfer rule from Scholz 2006 | Reviewed transfer rule |
| Startup / preionization | WP-N2 | WP-N2 delivery |
| Power port terms II/IV/V/VI | Sprint 2.2 (WP-N1B) → Sprint 3 (WP-N3 Σ_p) | Σ_p runtime interface + geometry packet |
| Closure registry | WP-N5 | WP-N5 delivery |
| Neutron authority | WP-N6 | WP-N6 delivery |
| External reviewer | External assignment | External reviewer named + review completed |
| Negative tests N7-NEG-01 through N7-NEG-10 | Sprint 6 (certificate) | Implemented and passing |

**Certificate total:** 7 candidate / 22 blocked / 0 accepted.
**Comparator channels:** 6 candidate / 9 blocked / 0 accepted.
**Validation promotion:** blocked.
