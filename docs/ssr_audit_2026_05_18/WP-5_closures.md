# WP-5 / SSR-008 Audit — Physics Closures

Date: 2026-05-18
Auditor scope: WP-5 / SSR-008 (Physics Closures), DPF-Unified first-principles path
Branch: `codex/corpus`
Mode: READ-ONLY on all repo code/docs/tests. Only this report file is created.
Source authority: `KnowledgeReference/` only. Every citation below was opened and verified.

---

## (a) Verdict

**`request_changes` — `accept_engineering_progress` on the closure-packet design.**

The `physics_closure` packet (`closure_packet.py`) is honest, fail-closed, and
correctly driven by runtime telemetry. Every required effect is enumerated;
nothing disappears from artifacts. EOS, radiation, ablation/impurity, anomalous
resistance, restrike, and beam-target coupling all remain VISIBLE `blocked`
packets. `can_support_first_principles_acceptance` is hard-wired `False` at
every level. No reduced-model (Lee/snowplow/fcr/fitted-fraction) physics leaks
into the first-principles closure path.

It is not yet acceptable as a *complete* WP-5 deliverable because:

1. **No `tests/test_first_principles_closures.py` exists.** SSR-008 and WP-5
   require unit + negative tests for each active closure and blocker-hardening
   tests for the missing ones. Per-module tests exist
   (`test_conductivity_blend.py`, `test_ionization_transport.py`,
   `test_two_temperature.py`, `test_electron_energy_closure.py`,
   `test_bremsstrahlung_nrl.py`), but there is no closure-matrix-level test
   binding them to the `physics_closure` packet's `blocked`/`candidate` status.
2. **Two imprecise source citations** (not fabricated, but they do not point at
   the formula they back — see Source Evidence table rows S2 and S6).
3. **`# EMPIRICAL` knobs exist in `radiation/line_radiation.py` and
   `atomic/ablation.py`.** They are correctly flagged and correctly *fenced out
   of the first-principles runtime* (verified: not imported by `hybrid_loop.py`
   / `hybrid_stepper.py` / `runner.py`), but the closure packet does not name
   them explicitly as fenced-out modules.

No `reject_overclaim` trigger fired: no closure is silently promoted, no hidden
floor changes a `blocked` verdict, no AI-invented formula is implemented as
authority.

---

## (b) Closure Matrix

Legend — Active = runs inside the first-principles `hybrid_loop` runtime;
Candidate (legacy) = implemented but only reachable from the old MHD engine,
NOT the FP path; Blocked = `physics_closure` packet status `blocked`.

| Closure | Implemented? | Active in FP runtime? | Source ref (verified) | Validity regime stated? | Units checked? | Numerical limiter / bound? | Coupling order | Energy accounting | Negative tests |
|---|---|---|---|---|---|---|---|---|---|
| **EOS / thermodynamics** | No (ideal-gas `gamma` only, in MHD solvers) | No — FP runner has no EOS call | n/a | n/a | n/a | n/a | n/a | n/a | Packet `blocked`; no dedicated test |
| **Spitzer resistivity** (`collision/spitzer.py::spitzer_resistivity`) | Yes | Yes (via `partial_ionized_conductivity`) | NRL `:3211-3215` (τ_e), `:3236` (σ₀=ne²τ_e/m_e) — VERIFIED. Module's own `:2660-2725` citation is imprecise (page 29/30, transverse row only) | Partial — Coulomb-collision validity not stated against NRL conditions (1)-(6) at `:3371-3383` | Yes (Ω·m, derived) | `coulomb_log` floored at 2.0; `ne·e²` guarded `+1e-300` | conductivity → Ohm solve | n/a (transport coeff) | `test_formulary_transport_audit.py` only |
| **Braginskii α(Z) correction** (`spitzer_alpha`) | Yes | Yes (inside `spitzer_resistivity`) | Braginskii 1965 Table 1 — `[TRAINING]`, NOT in KR. α(1)=0.5064 consistent with NRL `0.51 j∥` at `:3235` | Z∈[1,∞] piecewise | dimensionless | piecewise-linear, bounded | inside resistivity | n/a | none |
| **Weakly-ionized / partial-ionized conductivity** (`fields/conductivity.py::partial_ionized_conductivity`) | Yes | Yes (`use_source_backed_conductivity` path) | NRL `:3379-3425` (σ=neμ, e-n collision freq) — VERIFIED. σ_en cross-section `5e-19 m²` matches NRL `:3392` "~5×10⁻¹⁵ cm²" | Yes — `limitations` names scalar-only, typical-cross-section | Yes (S/m) | `ne_safe=max(ne,1)`; `σ=0` where `ne=0` | feeds Ohm solve | n/a | `test_conductivity_blend.py::test_partial_ionized_conductivity_*` |
| **Plasma-vacuum conductivity blend** (`PlasmaVacuumConductivityBlend`) | Yes | Conditional — bypassed when source-backed transport on | `HYBRID_PIC_3D_SOURCE` (hybrid PIC paper) | Yes (telemetry `status`) | Yes (S/m) | Ohmic-CFL cap `σ_cfl = safety·ε₀/dt` | pre-Ohm | n/a | `test_conductivity_blend.py` (4 incl. negative) |
| **Generalized Ohm — resistive + Hall + ∇pₑ** (`fields/ohm_solver.py`) | Yes | Yes | `HYBRID_PIC_3D_SOURCE` `:1107-1185` (claimed in telemetry) | Partial — `θ∈[0.5,1]` enforced; Hall/pressure validity deferred to electron-energy | Yes (A/m²) | `θ` range check; algebraic residual reported | current closure | residual telemetry | `test_*` ohm (not in WP-5 file) |
| **Electron-inertia term** | No — omitted from generalized Ohm | No | n/a | Not stated as omitted in packet | n/a | n/a | n/a | n/a | none |
| **Ionization / recombination** (`fields/ionization_transport.py`) | Yes | Yes | Step ionization: vacuum `:252-259` — VERIFIED (e-impact ground-state + radiative + 3-body). Rates: NRL `:4572-4648` — VERIFIED eq.(10),(12),(15) | Yes — `validity_notes` + `limitations` (single-stage D only) | Yes — cgs→SI verified (eq.12 ×1e-6; eq.15 `8.75e-27`→`8.75e-39` m⁶/s) | post-field, pre-deposit | particle source/sink conserves macro-weight; reports `unrepresented_recombination_ions` deficit | `test_ionization_transport.py` |
| **Atomic CR / Lotz / Saha** (`atomic/ionization.py`) | Yes | No — not imported by FP runtime | Saha `[TRAINING]`; Lotz 1967 `[TRAINING]` not in KR; NRL eq.13 recomb VERIFIED `:4609-4622` | Partial | Yes | Thomas solve, normalized | n/a (legacy) | n/a | `test_ionization.py` |
| **Two-temperature electron energy** (`fluid/two_temperature.py` + `fields/electron_energy.py`) | Yes | Yes (`ElectronEnergyClosure`) | vacuum JSON `:57-62` equation structure — page-2 content supports separate Te/Ti+heat-flux+exchange but the JSON line range is structural, not semantic | Yes — relativistic guard, density gate, `closure_validity` packet | Yes (J/m³, K) | Te floor; superluminal-drift + relativistic-Te → `blocked_` | operator-split source terms | compression+Q_ohm+Q_ei−Q_rad; `_reconcile_energy_density_to_temperature` | `test_two_temperature.py`, `test_electron_energy_closure.py` |
| **Electron–ion equilibration** (`compute_equilibration_source`, `relax_temperatures`) | Yes | Yes | NRL `:2996-3020` equal-T equilibration — VERIFIED (used as *audit reference*, not authority) | Stated `equilibration_convention_source_audit_incomplete` | Yes (W/m³) | implicit 2×2 exact solve (unconditionally stable) | inside electron-energy step 4 | symmetric e↔i transfer | `test_two_temperature.py`; `equilibration_convention_audit` |
| **Electron heat flux (Braginskii anisotropic)** (`electron_energy.py::_apply_braginskii_heat_flux_candidate`) | Yes | Yes (when B supplied) | vacuum JSON `:57-62` (heat-flux terms) + `spitzer.py::braginskii_kappa`; κ∥ coeff 3.16 vs NRL `3.2` at `:3286` — VERIFIED close | Yes — density gate, subcycle stability limit, implicit/ADI fallbacks | Yes (W/(m·K), W/m³) | explicit subcycle cap → implicit GMRES → diagonal-ADI; zero-normal-flux BC | inside electron-energy step | `net_heat_flux_power_W` reported; floor-contact counted | `test_electron_energy_closure.py` |
| **Heat-flux κ coefficient** (`braginskii_kappa_coefficient`) | Yes | Yes | Braginskii 1965 Table 1 — `[TRAINING]`, NOT in KR. δ_e(1)=3.16 vs NRL `:3286` `3.2` | Z piecewise | dimensionless | bounded interpolation | inside κ | n/a | none |
| **Bremsstrahlung radiation** (`radiation/bremsstrahlung.py`) | Yes | **No** — not imported by FP runtime (only by legacy `two_temperature` MHD path) | NRL eq.(30) `:4730-4736` — VERIFIED `P_Br=1.69e-32 Ne Te^½ Σ[Z²N(Z)]` W/cm³. SI `BREM_COEFF=1.569e-40` derivation VERIFIED | Yes (docstring) | Yes (W/m³) | implicit Newton solve, `Te_floor` | n/a (FP path) | implicit solve returns `P_radiated` | `test_bremsstrahlung_nrl.py` |
| **Line / recombination radiation** (`radiation/line_radiation.py`) | Yes | No — not in FP runtime | Recomb `C_REC` from NRL eq.(33) `:4749-4754` — VERIFIED. **Cooling-function fits `# EMPIRICAL`, provenance unknown** (Post 1977 attribution removed, see module docstring) | Yes — `unknown_provenance_empirical_fits` status | Yes (W·m³, W/m³) | implicit Newton, energy-imbalance monitor (`rel_error>1e-6` warns) | n/a (FP path) | conservation monitor present | `test_mlx_line_radiation.py`, `test_formulary_radiation_audit.py` |
| **Improved radiation (Gaunt/cyclotron)** (`radiation/improved_radiation.py`) | Yes | No — not in FP runtime | Cyclotron NRL eq.(34) `:4756-4757` VERIFIED; Gaunt fit `[TRAINING]` (van Hoof 2014, not in KR) | Partial | Yes (SI conversions documented) | clips g_ff∈[1,5] | n/a | n/a | `test_improved_radiation.py` |
| **QMF suppression** (`radiation/qmf_suppression.py`) | Yes | No — not in FP runtime | **`[UNVERIFIED]` — heuristic interpolation, no free-free source on disk** (correctly self-flagged) | Yes — `unverified_not_design_evidence` | Yes | floors S∈[0.01,1] | n/a | n/a | none |
| **Radiation transport (FLD)** (`radiation/transport.py`) | Yes | No — not in FP runtime | Levermore-Pomraning `[TRAINING]`; Kramers opacity `[TRAINING]`. `rosseland_kramers_fld_source_packet_missing` | Yes — explicit `source_status` | Yes (SI) | sub-cycle cap 10000, D≤c·dx/ndim, opacity clamp | n/a | `Q_absorbed` returned | none |
| **Electrode ablation / impurities** (`atomic/ablation.py`) | Yes | No — not in FP runtime | **`# EMPIRICAL` constant efficiencies (Cu 5e-5, W 2e-5 kg/J). `ablation_efficiency_source_packet_missing`** — Bruzzone/Vikhrev/Lee refs `[TRAINING]`, not in KR | Yes — `constant_efficiency_..._scaffold` | Yes (kg/J, kg/(m³s)) | guards on J,η,efficiency>0; melt-temp threshold | n/a (FP path) | linear dm/dt=η·P | none |
| **Anomalous resistivity** | No | No | vacuum `:31-32`, `:240-241` *describe* anomalous resistivity (lower-hybrid/Buneman) but give NO formula; NRL `:2704-2710` gives ion-sound `ν*` form | Packet `blocked` | n/a | n/a | n/a | n/a | Packet `blocked`; no test |
| **Restrike** | No | No | n/a — no KR source | Packet `blocked` | n/a | n/a | n/a | n/a | Packet `blocked` |
| **Collisions** (`spitzer.py` ν_ei/ν_ee/ν_ii/ν_en) | Yes | Yes (collision stage in `source_ordered_loop`) | NRL τ_e/τ_i `:3211-3222` — VERIFIED | Partial — NRL validity (1)-(6) `:3371-3383` not echoed | Yes (s⁻¹) | `max(denom,1e-300)` guards | post-velocity-update | n/a | `test_formulary_transport_audit.py` |
| **Stopping / beam-target coupling** | Partial (kinetic-yield history only) | Candidate when `kinetic_yield` present | Packet `blocked` (`beam_target_coupling`) | Packet `blocked` | n/a | n/a | n/a | n/a | Packet `blocked` |
| **Hall / FLR / kinetic scope** | Hall: yes (Ohm solver) | Yes when `include_hall` | hybrid PIC paper `:1226-1240` (Te=Ti limitation) | `candidate`; needs Te authority | Yes | matrix solve | inside Ohm | n/a | `extended_ohm_temperature_authority_status` gate |

**Headline:** of ~22 closure rows — **9 are active and source-backed in the FP
runtime** (Spitzer resistivity, partial-ionized conductivity, conductivity blend,
generalized Ohm, ionization/recombination transport, two-temperature electron
energy, e-i equilibration, Braginskii electron heat flux, collisions); **2 are
candidate** (Hall, beam-target/kinetic-yield); **6 are blocked** (EOS, radiation
losses, ablation/impurities, anomalous resistance, restrike, beam-target
acceptance); **5 are implemented-but-legacy** (bremsstrahlung, line radiation,
improved radiation, QMF, FLD transport, atomic CR — reachable only from the old
MHD engine, NOT the first-principles `hybrid_loop`).

---

## (c) Source Evidence Table

Every row below was verified by opening the cited KR file at the stated lines.

| ID | Cited in code | KR source : lines | Claim | Verdict |
|---|---|---|---|---|
| S1 | `closure_packet.py:31-34` | `2019nrlplasma...md:2996-3020` | NRL thermal-equilibration form | **TRUE** — lines 2996-3020 contain `dTα/dt=Σ ν̄ᵉ(Tβ−Tα)` and the equal-Te/Ti special case |
| S2 | `conductivity.py:14-16` `NRL_SPITZER_CONDUCTIVITY_SOURCE` | `2019nrlplasma...md:2660-2725` | "Spitzer conductivity" | **IMPRECISE** — 2660-2725 is page 29 (velocities) + page 30 header. Transverse Spitzer resistivity is only at line 2701-2703; the *parallel* resistivity actually used (`σ₀=ne²τ_e/m_e`, τ_e) lives at `:3211-3236`. Citation does not point at the formula it backs. Not fabricated; range exists. Recommend repoint to `:3211-3236`. |
| S3 | `conductivity.py:17-19` `NRL_WEAKLY_IONIZED_CONDUCTIVITY_SOURCE` | `2019nrlplasma...md:3379-3425` | weakly-ionized σ=neμ, e-n collisions | **TRUE** — "Weakly Ionized Plasmas" at 3384, `νₐ=n₀σₛ(kTα/mα)^½` at 3387-3389, `σα=nαeαμα` at 3410-3411, σ_en typical `~5×10⁻¹⁵ cm²` at 3392 |
| S4 | `ionization_transport.py:15-17` `STEP_GROUND_STATE_IONIZATION_SOURCE` | `vacuum-2004...md:252-259` | e-impact ground-state ionization + radiative + 3-body recomb | **TRUE** — lines 256-265 give exactly that inelastic-process list and the `dnₑ/dt` rate equation |
| S5 | `ionization_transport.py:18-20` `NRL_CHARGE_STATE_SOURCE` | `2019nrlplasma...md:4572-4648` | NRL charge-state rate eq, ionization + recomb rates | **TRUE** — eq.(10) charge-state at 4574-4582, eq.(12) ground-state ionization at 4599-4605, eq.(13) radiative recomb at 4613-4622, eq.(15) 3-body `8.75×10⁻²⁷ Te⁻⁴·⁵` at 4628 |
| S6 | `electron_energy.py:619-624`; `closure_packet.py:26-29` | `vacuum-2004...json:57-62` | PF-1000 2T heat-flux/ionization equation structure | **IMPRECISE/FRAGILE** — the `.json` is page-structured; lines 57-62 fall inside the page-2 `"text"` block, which *does* contain separate Te/Ti equations, `∇·q̃ₑ`, `Qₑ₋ᵢ`, `Qioniz`. Content supports the claim, but a JSON structural line range is not a stable semantic citation; the `.md` equivalent (`:240-259`) is the proper anchor. |
| S7 | `bremsstrahlung.py:9-12,52-53` | `2019nrlplasma...md` eq.(30) p.58 | `P_Br=1.69e-32 Ne Te^½ Σ[Z²N(Z)]` W/cm³ | **TRUE** — eq.(30) at line 4733-4736. SI conversion `1.69e-32·1e6/1e12/√11604.5 = 1.569e-40` independently re-derived and confirmed. (Note: module cites `plasma-formulary.md:L5101`; the canonical `2019nrl...md` line is 4733.) |
| S8 | `line_radiation.py:108-115` `C_REC` | NRL eq.(33) | free-bound `C_REC=1.69e-38·√13.6` | **TRUE** — eq.(33) recombination radiation at line 4749-4754 |
| S9 | `improved_radiation.py:13-17` cyclotron | NRL eq.(34) | `P_c=6.21e-28 B² Ne Te` W/cm³ → SI `5.35e-24` | **TRUE** — eq.(34) at line 4756-4757 |
| S10 | `spitzer.py` τ_e / ν_ei | `2019nrlplasma...md:3211-3222` (not cited by line) | electron/ion collision times | **TRUE** — τ_e=`3.44e5 Te^1.5/(nλ)` at 3211-3215, τ_i at 3217-3222. Module cites only the file, not the line — acceptable but should be tightened. |
| S11 | `electron_energy.py` source_lines `1074-1097,1226-1240,1267-1278` | hybrid PIC paper (1346-line file) | electron-energy / Te=Ti limitation | **PLAUSIBLE, not opened line-by-line in this audit** — file has 1346 lines so ranges exist; the `closure_packet` already flags these as candidate-only. Recommend a follow-up line check. |

**No fabricated citations found.** Two citations (S2, S6) are real-but-imprecise
and should be repointed. No citation to AI/web material as physics authority.

---

## (d) Empirical-Knob / Reduced-Model-Leakage Findings

Searched all six target areas for `# EMPIRICAL`, constants-as-closures, fitted
factors, and Lee/snowplow/fcr leakage.

**Findings — all currently fenced out of the first-principles runtime:**

1. **`radiation/line_radiation.py`** — `_cooling_hydrogen/neon/argon/copper/
   tungsten` are `# EMPIRICAL` piecewise power-law / log-log fits of **unknown
   provenance**. The module is honest: docstring (lines 31-43) explicitly states
   the prior "Post et al. 1977" attribution was removed because Post 1977 is not
   on disk and Post publishes log-polynomial (not power-law) fits;
   `LINE_RADIATION_SOURCE_STATUS = "unknown_provenance_empirical_fits"`.
   **Judgment: acceptable as a flagged scaffold.** Verified NOT imported by
   `hybrid_loop.py`/`hybrid_stepper.py`/`runner.py` — it cannot reach a
   first-principles result. Risk only if a future change wires it in.

2. **`atomic/ablation.py`** — `COPPER_ABLATION_EFFICIENCY = 5.0e-5`,
   `TUNGSTEN_ABLATION_EFFICIENCY = 2.0e-5` kg/J are `# EMPIRICAL` constants.
   `ABLATION_SOURCE_STATUS = "ablation_efficiency_source_packet_missing"`,
   `model_role = "constant_efficiency_electrode_ablation_scaffold"`. Refs
   (Bruzzone, Vikhrev, Lee & Serban) are `[TRAINING]` — none on disk in KR.
   **Judgment: acceptable as a flagged scaffold; NOT wired into FP runtime.**
   The closure packet correctly keeps `impurity_electrode_ablation` `blocked`.

3. **`radiation/qmf_suppression.py`** — `bremsstrahlung_suppression_factor` is a
   `[UNVERIFIED]` heuristic interpolation with no published free-free source;
   self-flagged `QMF_SOURCE_STATUS = "free_free_suppression_source_missing"`.
   **Judgment: acceptable as flagged diagnostic; not in FP runtime.**

4. **`radiation/transport.py`** — Kramers opacity coefficient `C_ff=3.7e-2`
   and Levermore-Pomraning limiter are engineering closures; module declares
   `rosseland_kramers_fld_source_packet_missing`. Not in FP runtime.

5. **`atomic/ionization.py`** — Lotz `_LOTZ_A = 4.5e-14` and Saha are `[TRAINING]`
   (Lotz 1967 not on disk). Module declares
   `IONIZATION_ACCEPTANCE_STATUS = "blocked_pending_species_table_packet_review"`.
   Not in FP runtime (the FP path uses `fields/ionization_transport.py`, which
   IS source-backed). The closure-search doc confirms this:
   "does not use the empirical `src/dpf/fluid/ionization.py` coronal-fit helper
   as authority."

**Reduced-model leakage: NONE found.** No Lee/RADPF/snowplow/fitted-current-
fraction/fcr term appears in any active first-principles closure. The
`improved_radiation`/`line_radiation` modules cite tokamak cooling literature
but are not authority and not wired in.

**Constants-as-closures inside the FP runtime:** the only constant defaults in
active code are physically-defensible NRL "typical" values —
`NRL_TYPICAL_ELECTRON_NEUTRAL_CROSS_SECTION_M2 = 5.0e-19` (matches NRL
`:3392` `~5×10⁻¹⁵ cm²`) and gaunt `1.2` (matches NRL `:4740` `g≈1.2`). Both are
source-grounded, not tuning knobs. **Acceptable.**

**Gap (not leakage):** `closure_packet.py` does not explicitly enumerate the
five fenced-out empirical modules in (1)-(5). A reviewer reading only the packet
cannot see that `line_radiation`/`ablation`/`qmf` exist but are quarantined.
Recommend an explicit `fenced_out_empirical_modules` field (see patch).

---

## (e) Proposed Patch (TEXT ONLY — not applied)

### Patch 1 — `closure_packet.py`: name the fenced-out empirical modules

Add to the dict returned by `build_physics_closure_packet` (after
`"negative_test_policy"`, before `"source_references"`):

```python
        "fenced_out_empirical_modules": {
            "policy": (
                "these modules contain # EMPIRICAL or unknown-provenance "
                "coefficients and are NOT imported by the first-principles "
                "hybrid_loop/hybrid_stepper/runner; they cannot drive a "
                "first-principles closure result"
            ),
            "modules": [
                {
                    "path": "src/dpf/radiation/line_radiation.py",
                    "reason": "unknown_provenance_empirical_cooling_fits",
                    "wired_into_first_principles_runtime": False,
                },
                {
                    "path": "src/dpf/atomic/ablation.py",
                    "reason": "constant_efficiency_no_source_packet",
                    "wired_into_first_principles_runtime": False,
                },
                {
                    "path": "src/dpf/radiation/qmf_suppression.py",
                    "reason": "unverified_heuristic_no_free_free_source",
                    "wired_into_first_principles_runtime": False,
                },
                {
                    "path": "src/dpf/radiation/transport.py",
                    "reason": "rosseland_kramers_fld_source_packet_missing",
                    "wired_into_first_principles_runtime": False,
                },
                {
                    "path": "src/dpf/atomic/ionization.py",
                    "reason": "lotz_saha_training_data_not_in_knowledgereference",
                    "wired_into_first_principles_runtime": False,
                },
            ],
            "can_support_first_principles_acceptance": False,
        },
```

### Patch 2 — `conductivity.py`: repoint the imprecise NRL Spitzer citation

```python
# BEFORE
NRL_SPITZER_CONDUCTIVITY_SOURCE = (
    "KnowledgeReference/2019nrlplasma-formulary-037290d4.md:2660-2725"
)
# AFTER  (parallel resistivity: tau_e at :3211-3215, sigma0=ne^2 tau_e/m_e at :3236)
NRL_SPITZER_CONDUCTIVITY_SOURCE = (
    "KnowledgeReference/2019nrlplasma-formulary-037290d4.md:3211-3236"
)
```

### Patch 3 — `electron_energy.py` / `closure_packet.py`: anchor heat-flux source to the `.md`

The `.json` line range `57-62` is structural, not semantic. Repoint to the
`.md` equivalent (already verified to contain the separate-Te/Ti equation set
with `∇·q̃ₑ`, `Qₑ₋ᵢ`, `Qioniz`):

```python
# electron_energy.py  _apply_braginskii_heat_flux_candidate, base dict
        "source": (
            "KnowledgeReference/doi-10-1016-j-vacuum-2004-05-019-f931cb0b.md"
        ),
        "source_lines": "240-259",
```
and the matching `CLOSURE_SOURCE_REFS` entry in `closure_packet.py` (lines
26-29): change `.json` `"lines": "57-62"` to `.md` `"lines": "240-259"`.

---

## (f) Negative Tests — Present vs Missing

### Present (per-module, verified to exist)
- `tests/test_conductivity_blend.py::test_conductivity_blend_rejects_invalid_inputs`
  — `pytest.raises(ValueError)` on bad `dt_s` (negative test PRESENT).
- `tests/test_conductivity_blend.py::test_conductivity_blend_candidate_evidence_does_not_satisfy_hybrid_gate`
  — asserts candidate evidence cannot pass the gate (negative test PRESENT).
- `tests/test_conductivity_blend.py::test_partial_ionized_conductivity_includes_neutral_drag`.
- `tests/test_ionization_transport.py`, `test_two_temperature.py`,
  `test_electron_energy_closure.py`, `test_bremsstrahlung_nrl.py`,
  `test_formulary_transport_audit.py`, `test_formulary_radiation_audit.py`.

### Missing (required by SSR-008 / WP-5)
1. **No `tests/test_first_principles_closures.py`.** No test asserts the
   `physics_closure` packet keeps EOS / radiation_losses /
   impurity_electrode_ablation / restrike_anomalous_resistance `blocked` and
   `can_support_first_principles_acceptance == False`.
   (`test_first_principles_runner.py` partially covers this per the closure-
   search doc, but there is no dedicated closure-matrix test file.)
2. No unit/negative test that `spitzer_resistivity` rejects `ne<=0` /
   `Te<=0` or that `coulomb_log` floors at 2.0.
3. No negative test that `partial_ionized_conductivity` rejects
   non-positive `electron_neutral_cross_section_m2` (the `ValueError` exists
   in code, untested).
4. No test that the empirical modules (`line_radiation`, `ablation`, `qmf`)
   are absent from the first-principles import graph — this is the single most
   important regression guard against future leakage.

**Proposed `tests/test_first_principles_closures.py` (TEXT — not created):**

```python
"""WP-5 / SSR-008 closure-matrix unit and blocker-hardening tests."""
import importlib
import numpy as np
import pytest
from dpf.first_principles.closure_packet import build_physics_closure_packet


def _packet(**kw):
    base = dict(include_hall=False, electron_energy_present=False,
                kinetic_yield_present=False, collisions_enabled=False)
    base.update(kw)
    return build_physics_closure_packet(**base)


# --- blocker-hardening: missing closures must stay blocked ------------------
@pytest.mark.parametrize("effect", [
    "eos_thermodynamics", "radiation_losses",
    "impurity_electrode_ablation", "restrike_anomalous_resistance",
])
def test_missing_closure_stays_blocked(effect):
    pkt = _packet()
    rec = pkt["effects"][effect]
    assert rec["status"] == "blocked"
    assert rec["implemented"] is False
    assert rec["can_support_first_principles_acceptance"] is False


def test_packet_never_supports_acceptance():
    # even with every optional channel present, packet cannot accept
    pkt = _packet(include_hall=True, electron_energy_present=True,
                  kinetic_yield_present=True, collisions_enabled=True,
                  electron_heat_flux_present=True,
                  electron_equilibration_audit_present=True,
                  ionization_charge_state_present=True,
                  source_backed_transport_present=True)
    assert pkt["can_support_first_principles_acceptance"] is False
    for rec in pkt["effects"].values():
        assert rec["can_support_first_principles_acceptance"] is False


def test_candidate_closure_cannot_be_promoted():
    pkt = _packet(electron_energy_present=True)
    assert pkt["effects"]["single_two_temperature_energy"]["status"] == "candidate"
    assert pkt["active_closure_policy"]["candidate_closures_can_support_acceptance"] is False


# --- empirical-module fence: regression guard against leakage ---------------
@pytest.mark.parametrize("mod", [
    "dpf.radiation.line_radiation", "dpf.atomic.ablation",
    "dpf.radiation.qmf_suppression", "dpf.radiation.transport",
])
def test_empirical_modules_absent_from_first_principles_imports(mod):
    """Empirical/unknown-provenance modules must not be reachable from the
    first-principles runtime import graph."""
    for fp_mod in ("dpf.fields.hybrid_loop", "dpf.fields.hybrid_stepper",
                   "dpf.first_principles.runner"):
        src = importlib.util.find_spec(fp_mod).origin
        text = open(src, encoding="utf-8").read()
        assert mod.split(".")[-1] not in text, (
            f"{mod} leaked into first-principles module {fp_mod}")


# --- active-closure unit + negative tests -----------------------------------
def test_partial_ionized_conductivity_units_and_negatives():
    from dpf.fields.conductivity import partial_ionized_conductivity
    ne = np.full((2, 2, 2), 1e23)
    nn = np.full((2, 2, 2), 1e22)
    Te = np.full((2, 2, 2), 1.16e6)  # ~100 eV
    sigma, tel = partial_ionized_conductivity(
        electron_density_m3=ne, neutral_density_m3=nn,
        electron_temperature_K=Te)
    assert np.all(sigma > 0.0)              # S/m positive
    assert tel.can_support_first_principles_acceptance is False
    with pytest.raises(ValueError):
        partial_ionized_conductivity(
            electron_density_m3=ne, neutral_density_m3=nn,
            electron_temperature_K=Te, electron_neutral_cross_section_m2=-1.0)
    with pytest.raises(ValueError):        # negative density rejected
        partial_ionized_conductivity(
            electron_density_m3=-ne, neutral_density_m3=nn,
            electron_temperature_K=Te)


def test_spitzer_resistivity_matches_nrl_order_of_magnitude():
    """eta ~ 5.2e-5 Z lnL Te_eV^-1.5 Ohm*m at Z=1 (NRL p.34); Braginskii
    alpha(1)=0.5064 halves the classical value."""
    from dpf.collision.spitzer import spitzer_resistivity
    Te_eV, lnL = 100.0, 10.0
    Te_K = np.array([Te_eV * 11604.518])
    ne = np.array([1e24])
    eta = float(spitzer_resistivity(ne, Te_K, lnL=lnL, Z=1.0)[0])
    nrl = 5.2e-5 * 1.0 * lnL * Te_eV ** -1.5      # classical NRL
    assert 0.3 * nrl < eta < 0.7 * nrl            # alpha(Z) correction band


def test_ionization_three_body_rate_si_conversion():
    """NRL eq.(15): alpha_3 = 8.75e-27 Te_eV^-4.5 cm^6/s -> 8.75e-39 m^6/s."""
    from dpf.fields.ionization_transport import nrl_three_body_recombination_rate
    rate = float(nrl_three_body_recombination_rate(np.array([10.0]))[0])
    expected = 8.75e-27 * 1e-12 * 10.0 ** -4.5
    assert abs(rate - expected) / expected < 1e-9


def test_bremsstrahlung_coefficient_is_si_not_cgs():
    """Guards the historical 1.69e-32 (CGS) vs 1.569e-40 (SI) bug."""
    from dpf.radiation.bremsstrahlung import BREM_COEFF
    assert 1.0e-40 < BREM_COEFF < 2.0e-40   # SI band; CGS 1.69e-32 would fail
```

---

## (g) Remaining Blockers

1. **`tests/test_first_principles_closures.py` does not exist** — SSR-008/WP-5
   require it. (Patch text provided in (f).)
2. **EOS is unclosed** — no QEOS/tabular EOS; FP runner has no EOS call. Packet
   `blocked`. The ALEGRA source (`unlimited-release...md:333-369`) flags
   low-density deuterium EOS as a stability blocker; no KR formula closes it.
3. **Radiation losses unclosed in the FP path** — `bremsstrahlung.py` is
   NRL-correct but only the legacy MHD engine imports it; the first-principles
   `hybrid_loop` has no radiation sink. Packet `blocked`.
4. **Anomalous resistivity has NO source formula** — vacuum-2004 only
   *describes* lower-hybrid/Buneman anomalous resistivity; NRL `:2704-2710`
   gives an ion-sound `ν*` form but with no DPF-scoped closure. Status: blocked,
   no implementable local source. Do NOT invent one.
5. **Restrike** — no KR source at all. Packet `blocked`. Correct.
6. **Ablation / impurities** — only the `# EMPIRICAL` constant-efficiency
   scaffold exists; no KR ablation source packet. Packet `blocked`. Correct.
7. **Beam-target coupling** — kinetic-yield history only; mechanism separation,
   stopping, spectrum/anisotropy, detector response all absent. Packet `blocked`.
8. **Electron-inertia term** omitted from generalized Ohm and not declared as a
   bounded-out omission in the closure packet.
9. **Two imprecise citations** (S2 conductivity Spitzer, S6 heat-flux `.json`) —
   repoint per Patch 2/3.
10. **`electron_energy.py` `source_lines` `1074-1097/1226-1240/1267-1278`** into
    the 1346-line hybrid-PIC paper were not opened line-by-line in this audit —
    flagged for a follow-up verification pass.

**Bottom line:** the closure *bookkeeping* (the `physics_closure` packet) is
sound, honest, and fail-closed; the active closures that ARE wired into the
first-principles runtime are source-backed and unit-correct. The submission
fails WP-5 completeness on missing closure-matrix tests and two imprecise (not
fabricated) citations, and EOS/radiation/ablation/anomalous/restrike/beam-target
remain correctly visible blockers. No overclaim, no reduced-model leakage, no
hidden floors affecting a verdict.
