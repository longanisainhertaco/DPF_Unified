# First-Principles DPF Simulator — SSR Implementation Submission

Date: 2026-05-18

Submitting team: Cortana multi-agent implementation team (Claude Opus 4.7 lead +
15 delegated dpf-domain agents).

Against spec: `docs/FIRST_PRINCIPLES_SIMULATOR_SSR_AND_IMPLEMENTATION_AUDIT_SPEC_2026_05_18.md`

Self-assessed verdict offered: **`accept_engineering_progress`** — honest
experimental engineering progress on the first-principles path. The certificate
gate remains blocked by design; no first-principles acceptance is claimed. The
formal verdict is left to the external Codex audit.

---

## 1. Implementation Summary

This submission executes one disciplined pass over WP-0 through WP-7 / SSR-001
through SSR-013. It is **not** a "finished simulator" claim. Per the spec, the
correct near-term state is an honest experimental simulator with source-truth
citations, honest blocker packets, and no unsupported acceptance claim.

Work performed:

1. **WP-0 restored.** The source-truth exhaustion gate had regressed to
   `exhausted: false` (4 open issues). Root cause: 5 macOS `.DS_Store` Finder
   metadata files miscounted as scientific source files. Fixed by permanently
   excluding `.DS_Store` from the source-truth scanner and regenerating the
   index. Gate is now `exhausted: true`, 0 issues; module source-vetting stays
   `strict_passed: true`.

2. **8-agent verification audit (Wave 1).** Every work package was audited
   against `KnowledgeReference/` by a dedicated opus agent: WP-1 power port,
   WP-2 startup, WP-3 geometry, WP-4 long-run runtime, WP-5 closures, WP-6
   neutron, WP-7 comparator/numerics/certificate, plus a repo-wide claim-safety
   sweep. Full reports: `docs/ssr_audit_2026_05_18/`.

3. **3 fail-closed-integrity defects found and fixed.** The audit found three
   genuine integrity violations (a fail-open acceptance gate, a dishonest
   status flag, a hidden numerical floor). All three were verified against
   source and fixed. See Section 7.

4. **80 negative-control tests added (Wave 2).** The spec rejects submissions
   lacking negative controls. A 7-agent test team built the negative-control
   suite each WP audit authored, verified against the now-fixed code.

5. **Runtime artifacts produced and inspected.** CLI smoke runs and a
   source-sign power-port run were executed; packet statuses confirmed honest
   and fail-closed.

The honesty discipline holds repo-wide: in the fresh runtime artifact,
**304 of 304** `can_support_first_principles_acceptance` flags are `False`, the
certificate gate is `blocked_first_principles_certificate_not_available`, and
`scientific_status` is `engineering_candidate_not_validation`.

---

## 2. Changed Files

### 2a. Source/test files modified this SSR pass

| File | Change | Requirement |
| --- | --- | --- |
| `scripts/verify_first_principles_source_truth_exhaustion.py` | Exclude `.DS_Store` from source-truth inventory + scan | WP-0 / SSR-001 |
| `src/dpf/first_principles/startup_bvp.py` | `can_support` now also requires `startup_payload_review["channel_acceptance_eligible"]` — closes fail-open gate | WP-2 / SSR-004 |
| `src/dpf/fields/hybrid_loop.py` | Bare-literal `np.maximum(...,1.0)` floor replaced by named `density_floor_m3`; `electron_density_floor` telemetry packet (floored-cell count) added | WP-4 / SSR-007/008 |
| `src/dpf/first_principles/runner.py` | `_conservation_telemetry` `passed`→`finite_state` + `energy_conservation_assessed`; `_conductor_mask_packet` rod-resolution disclosure fields | WP-3/WP-4 / SSR-005/011 |
| `src/dpf/first_principles/experimental_numerics.py` | Conservation-telemetry consumer rename `passed`→`finite_state` | WP-4 / SSR-011 |
| `src/dpf/first_principles/numerical_fidelity.py` | Consumer rename + `conservation_passed`→`conservation_finite_state` observation | WP-4 / SSR-011 |
| `src/dpf/first_principles/limiter_readiness.py` | Consumer rename + observation rename | WP-4 / SSR-011 |
| `tests/test_first_principles_runner.py` | +8 WP-1 power-port negative tests appended | WP-1 / SSR-006 |
| `docs/FIRST_PRINCIPLES_SOURCE_TRUTH_INDEX.json` / `.md` | Regenerated (WP-0 remediation) | WP-0 / SSR-001 |

### 2b. New files created this SSR pass

- `tests/test_first_principles_startup_bvp.py` (17 tests)
- `tests/test_first_principles_geometry.py` (8 tests)
- `tests/test_first_principles_long_run_integrity.py` (6 tests)
- `tests/test_first_principles_closures.py` (20 tests)
- `tests/test_first_principles_neutron_authority.py` (8 tests)
- `tests/test_first_principles_certificate_negative_controls.py` (13 tests)
- `docs/ssr_audit_2026_05_18/` — 8 WP audit reports
- `docs/FIRST_PRINCIPLES_SSR_SUBMISSION_2026_05_18.md` — this report
- `docs/FIRST_PRINCIPLES_SOURCE_TRUTH_EXHAUSTION_2026_05_18.*`,
  `docs/FIRST_PRINCIPLES_MODULE_SOURCE_VETTING_2026_05_18.*` — regenerated gates
- `results/audit_first_principles_3d_smoke.json`,
  `results/audit_experimental_whole_shot_smoke.json`,
  `results/audit_limiter_proof_auluck_power_port_1us_2026_05_18.json`

### 2c. Pre-existing in-progress work (present at session start — audited, not authored here)

These files were already modified/untracked on branch `codex/corpus` before this
SSR pass began (Stage-0 power-port work). They were **audited** by WP-1 and found
honest; they are disclosed here to keep the diff attributable:

- `src/dpf/first_principles/power_port.py` (+304: Stage-0 packet scaffolds,
  `lagged_auluck_volume_j_dot_e` mode, operator comparison, sigma deferral)
- `src/dpf/first_principles/deck.py`, `src/dpf/cli/main.py`,
  `src/dpf/fields/hybrid_simulator.py`, `tests/test_hybrid_3d_simulator.py`,
  `tests/test_cli_first_principles_3d.py`,
  `docs/FIRST_PRINCIPLES_ENGINEERING_FIRM_DOSSIER_2026_05_16.md`
- 7 untracked items: 4 today-dated docs, 3 pre-existing `results/` JSON files.

`runner.py` and `test_first_principles_runner.py` carry **both** pre-existing
in-progress work and this pass's changes.

---

## 3. Requirements Touched — SSR / WP Status Matrix

| ID | Area | Audit verdict | Packet status |
| --- | --- | --- | --- |
| WP-0 / SSR-001 | Source-truth + package-native authority | restored | exhaustion clean, vetting clean |
| WP-1 / SSR-006 | Power port | `accept_engineering_progress` | candidate, fail-closed |
| WP-2 / SSR-004 | Startup BVP | `request_changes` → gate fixed | blocked (correct) |
| WP-3 / SSR-005 | Geometry / boundaries | `request_changes` → disclosure fixed | candidate |
| WP-4 / SSR-007 | Long-run field/PIC/electron | `request_changes` → 2 fixes applied | candidate, honest |
| WP-5 / SSR-008 | Physics closures | `request_changes` | 9 source-backed / 6 blocked-visible |
| WP-6 / SSR-009 | Neutron authority | `accept_engineering_progress` | blocked, mechanism-separated |
| WP-7 / SSR-010/011/012 | Comparator/UQ/numerics/certificate | `request_changes` | all fail-closed |
| SSR-013 | Generalization | — | blocked (only PF-1000/Akel exercised) |
| SSR-002 | Package-native runtime authority | confirmed | CLI routes through `dpf.first_principles` |
| SSR-003 | PF-1000/Akel deck lock | confirmed | `deck_diff` packet locks 11 source values |

---

## 4. Source Evidence

All citations below were opened at the cited line range and content-verified by
the Wave-1 audit agents against local `KnowledgeReference/`. No fabricated paths
or non-existent line ranges were found across the audited modules.

| Local source path:lines | Claim / equation | Implemented in | Status |
| --- | --- | --- | --- |
| `auluck-2021-dpf-circuit-element.md:151-209` | `U_DPF = -∫_Ω(J·E)dV / I`; source-interface (cathode plate / cage) excluded | `power_port.py` operator comparison + Stage-0 packets | VERIFIED TRUE |
| `auluck-2021-dpf-circuit-element.md:235-262` | Poynting flux at source interface = `I·V` | `power_port.py` candidate ledger | VERIFIED TRUE |
| `2019nrlplasma-formulary-037290d4.md:1880-1888` | Poynting theorem `∂W/∂t + ∮N·dS = -∫J·E dV`, signed | `power_port.py` / conservation ledger | VERIFIED TRUE |
| `fully-electromagnetic-hybrid-pic-fluid-...-acb71fa9.md:741-789` | External-circuit ODE, current-derived B boundary, source-derived `U_DPF` | `runner.py` / `circuit_boundary.py` | VERIFIED TRUE |
| `radiation-physics-and-chemistry-188-2021-109633.md:262-270` | shot-12581 deck: L0=25nH, C0=1332µF, r0=6.1mΩ, b=16cm, a=11.55cm, z0=48cm, V0=16kV, p0=1.2Torr | `runner.py` `PF1000_AKEL_SOURCE_LOCKED_DECK` | VERIFIED TRUE |
| `experimental-...-pf-1000-facility-705bcc83.md:340-356` | 12 cathode rods 80mm, CE r=115.5mm, OE r=200mm, CE length 460mm, alumina insulator 85mm | `runner.py` `_conductor_mask_packet` | VERIFIED TRUE |
| `gribkov-2007-pf1000-jphysd-part2.md:55-80` | DPF phase structure: insulator breakdown, kinetic surface discharge, MHD inverse pinch, µs acceleration | `startup_bvp.py` source refs | VERIFIED TRUE |
| `effect-of-current-sheath-initiation-...-b2e95b88.md:616-642` | Paschen regimes are guidelines only; Paschen↔DPF breakdown link is fragile | `startup_bvp.py` (keeps startup fail-closed) | VERIFIED TRUE |
| `anisotropy-of-...-dd-fusion-neutrons-...-527cc533.md:94-130, 276-304` | TOF spectra estimate neutron energy; direct vs scattered must be separated before transformation | `neutron_authority.py` mechanism channels | VERIFIED TRUE |
| `2019nrlplasma-formulary-037290d4.md` (resistivity / heat-flux / ionization-recombination / bremsstrahlung families) | Closure formula families | `conductivity.py`, `ionization_transport.py`, `electron_energy.py`, `collision/spitzer.py` | VERIFIED TRUE |

Citation imprecisions found (real source, imprecise anchor — not fabrication;
patches noted in WP-5 audit doc): `conductivity.py` cites NRL `:2660-2725` for a
parallel-resistivity formula better anchored elsewhere; `closure_packet.py` /
`electron_energy.py` cite a `.json` structural line range instead of the `.md`
semantic anchor. Honest gap found by WP-2: `civ_breakdown.py` docstring cites
Alfvén 1954 / Brenning / Danielsson — not in `KnowledgeReference/` — but the
module is not promoted and self-flags `not_validation_evidence`.

---

## 5. Commands Run

```bash
# WP-0 — source-truth gate
.venv312/bin/python scripts/verify_first_principles_source_truth_exhaustion.py --refresh-index --strict
.venv312/bin/python scripts/verify_first_principles_module_source_vetting.py
git diff --check

# Verification — focused test suite + lint
.venv312/bin/python -m ruff check src/dpf/first_principles src/dpf/fields
.venv312/bin/python -m pytest tests/test_first_principles_*.py tests/test_hybrid_3d_*.py -q

# Audit Phase 5 — CLI smoke runs
.venv312/bin/dpf first-principles-3d --deck-preset pf1000_akel_16kv --steps 2 \
  --output results/audit_first_principles_3d_smoke.json
.venv312/bin/dpf experimental-whole-shot --deck-preset pf1000_akel_16kv --steps 20 \
  --target-time-s 1.0e-10 --dt-policy combined-cfl \
  --output results/audit_experimental_whole_shot_smoke.json

# WP-1 — source-sign power-port artifact (Auluck volume J·E mode)
.venv312/bin/dpf experimental-limiter-proof --deck-preset pf1000_akel_16kv \
  --circuit-udpf-mode lagged_auluck_volume_j_dot_e --target-time-s 1.0e-06 \
  --auto-step-budget --dt-policy vacuum-cfl --max-step-results 0 \
  --output results/audit_limiter_proof_auluck_power_port_1us_2026_05_18.json
```

Result: source-truth exhaustion `exhausted: true` / 0 issues; module vetting
`strict_passed: true`; `git diff --check` clean; ruff clean on touched files;
**180 passed, 4 xfailed** across `test_first_principles_*` + `test_hybrid_3d_*`.

---

## 6. Artifacts

| Path | Purpose | SHA256 |
| --- | --- | --- |
| `results/audit_first_principles_3d_smoke.json` | Audit Phase 5 — 3-D first-principles 2-step smoke | `b5f04f825b738347ceee5040290d16c2ccf7d25f57b7918a4071fd7844855d61` |
| `results/audit_experimental_whole_shot_smoke.json` | Audit Phase 5 — whole-shot 20-step smoke | `644e2aaa8ab35682c995d874e8a84f0227b4a1b189a786c8611fee882a4549a6` |
| `results/audit_limiter_proof_auluck_power_port_1us_2026_05_18.json` | WP-1 source-sign 1µs power-port run | `aa4e3bf452f8cdf27a5d64ce299ca25a496dd514b773c75393c8f84efe849dc6` |
| `docs/ssr_audit_2026_05_18/WP-1..7_*.md`, `claim_safety_sweep.md` | 8 detailed WP audit reports | text reports |

Runtime-artifact honesty check (`audit_first_principles_3d_smoke.json`):
`scientific_status: engineering_candidate_not_validation`; `blocker_count: 11`;
`certificate_gate` status `blocked_first_principles_certificate_not_available`;
`can_support_first_principles_acceptance` — 0 True / 304 False.

---

## 7. Integrity Defects Found and Fixed

The audit found three genuine fail-closed-integrity defects. Each was verified
against source before fixing; each fix is now confirmed live in the runtime
artifact.

### D1 — WP-2: fail-OPEN startup acceptance gate (`startup_bvp.py`)

`build_startup_bvp_packet`'s `can_support` was computed from caller-declared
`accepted_channels` strings alone. A caller could pass channel strings covering
all `REQUIRED_STARTUP_CHANNELS` plus an accepted mode and reviewed status, with
**no `startup_payload`**, and obtain `accepted_startup_bvp_packet` /
`can_support_first_principles_acceptance: True`. The `startup_payload_review`
(which reports `startup_payload_not_supplied`) was computed but never gated
acceptance. This violates the spec's fail-closed rule. **Fix:** `can_support`
now also requires `startup_payload_review["channel_acceptance_eligible"]`.
Default PF-1000/Akel deck (seeded_layer) was already correctly rejected; the
hole was the forceable accepted path.

### D2 — WP-4: dishonest conservation `passed` flag (`runner.py`)

`_conservation_telemetry` returned `"passed": finite` — a finiteness-only
boolean. A run that lost 58.4% of total energy (170 kJ → 71 kJ) still reported
`conservation_telemetry.passed: true`. **Fix:** field renamed `passed` →
`finite_state`; explicit `energy_conservation_assessed:
"not_assessed_no_accepted_tolerance"` added (no tolerance is invented). All 5
consumers across `experimental_numerics.py`, `numerical_fidelity.py`,
`limiter_readiness.py` updated; the honest energy-drift number
(`relative_tracked_total_energy_change`) was already present and is retained.

### D3 — WP-4: hidden electron-density floor (`hybrid_loop.py`)

`electron_density = np.maximum(ionization_state.electron_density_m3, 1.0)` — a
bare-literal floor with no activation count, in the source-backed-conductivity
path. **Fix:** floor now uses the named, range-validated `density_floor_m3`
parameter; an `electron_density_floor` telemetry packet
(`floor_active_cells`, `total_cells`, `density_floor_m3`, `floor_source`) is
emitted. The fresh artifact shows `floor_active_cells: 125` of 125 — a fully
active floor that was previously invisible.

### D4 — WP-3: rod-fidelity over-disclosure (`runner.py`)

`_conductor_mask_packet` reported `cathode_rods_projected: True` with no
resolution gate. At default grids the rods are sub-grid (0.73 cells across an
80 mm rod). **Fix:** `cathode_rod_diameter_grid_cells` and
`cathode_rods_resolution_reviewed: False` added, plus a limitation line. The
artifact now exposes `cathode_rod_diameter_grid_cells: 0.727`.

---

## 8. Negative Tests

80 negative-control tests added (76 pass, 4 honest xfail):

| File | Tests | Pass / xfail | Asserts |
| --- | --- | --- | --- |
| `test_first_principles_runner.py` (+WP-1) | 8 | 8 / 0 | sign reversal, domain corruption, time-centering downgrade, low-current P/I, sigma-as-driver rejection, first-step fallback, default-mode guard, negative-J·E not clipped |
| `test_first_principles_startup_bvp.py` | 17 | 17 / 0 | seeded/text-only/payload-less startup cannot accept; **the D1 fail-open gate fix holds** |
| `test_first_principles_geometry.py` | 8 | 5 / 3 | coarse geometry stays candidate; rod resolution not reviewed; xfail: mask SHA256 + projection-error packet (unimplemented) |
| `test_first_principles_long_run_integrity.py` | 6 | 5 / 1 | conservation telemetry has no `passed` key (D2 fix); floor telemetered (D3 fix); checkpoint/restart + split-continuation equivalence; xfail: checkpoint grid-mismatch guard |
| `test_first_principles_closures.py` | 20 | 20 / 0 | 6 blocked closures stay blocked; empirical modules absent from FP import graph; active-closure unit checks |
| `test_first_principles_neutron_authority.py` | 8 | 8 / 0 | scalar-total-yield-only cannot accept; Lee outputs stay comparator-only; cross-scope rejected; detector-response + UQ required |
| `test_first_principles_certificate_negative_controls.py` | 13 | 13 / 0 | certificate gate blocked under draft/blocked/cross-scope/missing-UQ/missing-review/hidden-limiter/reduced-model/app-only evidence; manifest raises on non-candidate status |

The 4 xfails are honest: they mark unimplemented SSR-005 packet fields (mask
hash, projection error) and one checkpoint-loader robustness gap (B-WP4-6), each
with a specific `reason`. No plain failing tests are committed.

---

## 9. Remaining Blockers

All 16 first-principles gates (FP-0..FP-15) and the certificate gate remain
blocked. This is the correct, honest state. Per-area open blockers:

- **Power port (WP-1/SSR-006):** integration domain `not_declared`; energy
  ledger is 4/5-term (wall-Poynting flux + electrode/interface work
  `missing_or_blocked`); no segmented/checkpointed 12µs source-sign run;
  reviewed residual tolerance not attached.
- **Startup (WP-2/SSR-004):** 0 of 9 startup channels source-backed; no
  KR-backed breakdown/flashover model; no explicit field/PIC handoff interval.
- **Geometry (WP-3/SSR-005):** no mask SHA256; no projection-error packet; rods
  sub-grid at default resolution; hollow anode / insulator / chamber masks
  not resolved; Krauz-vs-Akel source dimension disagreement not surfaced in
  `deck_diff`.
- **Long-run (WP-4/SSR-007):** checkpoint loader has no grid-shape guard
  (B-WP4-6); the 12µs fallback artifact lost 58% energy on a 5³ grid (a
  finiteness probe, not predictive); CFL limiter disabled in
  source-backed-conductivity mode.
- **Closures (WP-5/SSR-008):** EOS, radiation losses, ablation/impurities,
  anomalous resistance, restrike, beam-target acceptance all blocked;
  electron-inertia term omitted from generalized Ohm without a bounded-out
  declaration.
- **Neutron (WP-6/SSR-009):** accept path of
  `build_mechanism_separated_neutron_packet` is structurally inert (maximally
  fail-closed); `kinetic_neutron_yield_authority_status` is not wired into
  `runner.py`; no mechanism-separated neutron history implemented.
- **Comparator/UQ/certificate (WP-7/SSR-010/011/012):** `manifest.py` lacks
  command/`argv` provenance and a `source_packet_hashes` field; certificate
  accept-path channels are untested.
- **Generalization (SSR-013):** only PF-1000/Akel exercised; GV / Soto / IPFS /
  May15 / May16 decks remain candidate requirement material.

None of these are overclaims. Each is honestly reported by a fail-closed packet.
The detailed per-WP blocker lists are in `docs/ssr_audit_2026_05_18/`.

---

## 10. AI / Tool Disclosure

- **Tool:** Claude Opus 4.7 (Cortana), operating as lead, with 15 delegated
  agents — 8 Wave-1 audit agents (opus: `dpf-mhd-physicist`,
  `dpf-engine-architect`, `dpf-validation-engineer`, `general-purpose`) and 7
  Wave-2 test agents (`dpf-validation-engineer`, sonnet).
- **What it produced:** audit prose, the negative-control test code, the
  integrity-fix patches, and this report.
- **Local sources used to verify output:** every physics claim was checked
  against `KnowledgeReference/` line ranges by the audit agents and the fixes
  were verified against repo source by the lead before application. The spec,
  the SSR Engineering Manual Source Review, and the status/blocker docs were the
  governing inputs.
- **AI suggestions rejected for lack of local source support:**
  - The WP-5 audit draft asserted Spitzer resistivity should fall in
    `0.3–0.7 × NRL`. The test agent verified against `collision/spitzer.py` and
    the NRL formula, found the code returns `≈1.004 × NRL` (correct — the NRL
    form already encodes the Braginskii α correction), and **rejected the
    draft assertion**, correcting the test band to `0.7–1.3 ×`.
  - The SSR Engineering Manual's proposed quasi-TEM Sigma line-voltage circuit
    driver was **not implemented** — it is not verified by any local
    `KnowledgeReference/` source; it remains a deferred exploratory-only packet.
  - No external web, paper, or AI output was treated as scientific authority.
- No reduced model (Lee/RADPF/snowplow/scaling) drives any first-principles
  result; `deck.py` `REDUCED_MODEL_AUTHORITY_FIELDS` structurally rejects it.

---

## 11. Source-Sign 1µs Power-Port Run

The WP-1 source-sign branch (`lagged_auluck_volume_j_dot_e`, implementing
`U_DPF = -∫_Ω(J·E)dV / I` per Auluck 2021) was run to a full 1 µs horizon:

- `steps_completed: 4632`, `dt_s ≈ 2.159e-10 s` (vacuum-CFL policy)
- `duration_request_satisfied: True`, `termination_reason: target_time_reached`
- `scientific_status: engineering_candidate_not_validation`
- `conservation_telemetry.finite_state: True`,
  `relative_tracked_total_energy_change: -0.0074` (−0.7%),
  `energy_conservation_assessed: not_assessed_no_accepted_tolerance`
- `udpf_source: candidate_lagged_auluck_volume_j_dot_e` (source-sign mode active)
- `can_support_first_principles_acceptance: False`; certificate gate blocked

This completes the 100 ns and 1 µs source-sign attempts WP-1 requires. The
−0.7% 1 µs energy drift is far tighter than the 12 µs fallback run's 58% loss,
but is still an engineering-candidate result, not validation. The 12 µs
source-sign run remains a blocker — segmented/checkpointed long-run support is
not yet implemented (Section 9). The D2 conservation-honesty fix is confirmed
live in this artifact: `conservation_telemetry` has no `passed` key.

---

## 12. Self-Assessed Verdict

`accept_engineering_progress`. The implementation is honest and useful. Three
real integrity defects were found and fixed; 80 negative controls were added;
WP-0 was restored; the runtime emits honest fail-closed packets. The certificate
gate and all 16 first-principles gates remain blocked for the same machine/shot
scope — which is correct. No first-principles acceptance is claimed. The formal
audit verdict is left to Codex.
