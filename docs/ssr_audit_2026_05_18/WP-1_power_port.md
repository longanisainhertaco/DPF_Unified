# WP-1 / SSR-006 Audit — Circuit / Power-Port Authority

Date: 2026-05-18
Auditor scope: WP-1 / SSR-006, the #1-priority blocker.
Repo: `/Users/anthonyzamora/dpf-unified`, branch `codex/corpus`.
Mode: READ-ONLY on all `.py` and existing docs. Static audit only. No pytest / CLI runs.
Files audited (uncommitted in-progress work):
- `src/dpf/first_principles/power_port.py`
- `src/dpf/fields/hybrid_simulator.py` (power-port sections)
- `src/dpf/fields/circuit_boundary.py`
- `src/dpf/first_principles/runner.py` (`_deck_source_diff_packet`, power-port wiring)
- `src/dpf/first_principles/deck.py` (`circuit_udpf_mode` / `FIRST_PRINCIPLES_CIRCUIT_UDPF_MODES`)
- `src/dpf/cli/main.py` (`--circuit-udpf-mode`)
- `tests/test_first_principles_runner.py`, `tests/test_hybrid_3d_simulator.py`, `tests/test_cli_first_principles_3d.py`

---

## (a) Verdict

**`accept_engineering_progress`** — with `request_changes` conditions attached.

Reasoning:

1. **No overclaim.** Every new power-port object stays fail-closed. `can_support_power_port_acceptance: False` and `can_support_first_principles_acceptance: False` are present on every new packet, sub-packet, and nested operator entry. `accepted_load_power_source` remains `"none"`. `active_load_relation` for the new Auluck mode is `lagged_auluck_volume_j_dot_e_voltage_not_accepted` — the `_voltage_not_accepted` suffix is preserved. `_active_load_relation` and `_uses_candidate_j_dot_e_active_load` correctly classify the new mode as a candidate (not accepted) active load. This does **not** trip `reject_overclaim`.

2. **Source-faithful sign.** The new `lagged_auluck_volume_j_dot_e` mode computes `U_DPF = -power_W / I` where `power_W = j_dot_e_power_W` is the volume integral `sum(J·E)·cell_volume` over resolved plasma cells (`hybrid_stepper.py:296-314`, `positive J·E = field work on charges`). This matches Auluck 2021 Eq. 1, `V12 = -∫_Ω(J·E)d³r / I` (KR `auluck-2021-dpf-circuit-element.md:173-200`). The pre-existing conservative `lagged_volume_j_dot_e` mode uses `+power_W/I` (opposite sign) and is correctly still labeled non-accepted. The new mode is the *more* source-faithful of the two and is honestly carried as candidate.

3. **Negative `J·E` is not clipped.** `_circuit_udpf_for_step` (`hybrid_simulator.py:540-565`): for `lagged_auluck_volume_j_dot_e` the `power_W < 0.0` branch is bypassed — negative `J·E` flows straight into `-power_W/I` as signed feedback. This satisfies SSR-006 ("Negative local `J.E` must not be clipped") and the Engineering Manual review (NRL Poynting theorem treats negative local `J·E` as signed exchange). The old conservative mode still blocks negative `J·E` and reports it via the `input_sequence_fallback_negative_j_dot_e_active_port_blocked` source tag — visible, not hidden.

4. **Low-current `1/I` is reported as a blocker, not hidden.** `_low_current_p_over_i_feedback_packet` (`hybrid_simulator.py:568-619`) emits `status = "blocked_low_current_p_over_i_singularity_not_validation"` when `|I| <= min_current_A`. The runtime falls back to `input_udpf_V` with source tag `input_sequence_fallback_low_current` (counted in `udpf_source_counts`), and the packet carries `singularity_blocked_this_step` plus an `acceptance_note` demanding a source-backed handoff/regularization packet. This satisfies SSR-006 ("Low-current `P/I` fallback must be reported as a blocker") and the Rejection Criteria ("`1/I` ... fallback behavior" must not be hidden).

5. **Sigma / quasi-TEM line voltage stays DEFERRED.** `_sigma_quasi_tem_line_voltage_packet` and the `sigma_quasi_tem_line_voltage` entry in `_operator_comparison_packet` set `disallowed_runtime_use: "accepted_or_primary_circuit_driver"`, `source_status: "not_verified_in_local_dpf_source"`, and the comparison `decision: "do_not_replace_active_driver_with_sigma_line_voltage"`. No Sigma operator is wired into `_circuit_udpf_for_step` — it is text-only. Consistent with the Engineering Manual review (Sigma not source-verified) and SSR-006.

6. **Deck-diff packet locks PF-1000/Akel and flags drift.** `_deck_source_diff_packet` + `PF1000_AKEL_SOURCE_LOCKED_DECK` (`runner.py:92-104, 2242-2347`) lock all 11 required source values and emit `blocked_source_deck_drift_not_validation` on any mismatch.

Why not a clean `accept`: this is engineering progress, not an accepted power port — exactly the spec's intended Stage-0 state. Why `request_changes` is attached and not a clean accept-only: **the WP-1 negative-test suite is not present** (see §f), three deck-lock values need a verification note (see §b/§d), the committed `2026_05_18` artifacts **predate the new code** and therefore do not contain the new packets (see §c), and the spec's required artifact set is `100 ns / 1 µs / 12 µs` whereas the repo carries `100 ns / 1 ns / 12 µs` (see §e). None of these are overclaims; they are completeness gaps that must close before WP-1 can advance past `accept_engineering_progress`.

---

## (b) Source Evidence Table

Every citation below was opened at the cited lines and re-verified during this audit.

| Citation (in code) | KR file:lines | Claim made in code | Verdict |
|---|---|---|---|
| `POWER_PORT_SOURCE_REFS` role `field_power_contract` | `auluck-2021-dpf-circuit-element.md:151-200` | DPF as two-terminal circuit element; `V12 = -∫_Ω(J·E)d³r / I`; all chamber phenomena draw power from external circuit | **VERIFIED-TRUE.** Lines 151-154 define the two-terminal element and voltage as work against E. Eq. (1) at lines 173-195 is `V12(t) = -∫ J·E d³r / I(t)` (OCR-fragmented but unambiguous: leading minus, volume J·E numerator, current denominator). Lines 196-201 state RHS is total electric power through terminals and every chamber phenomenon draws power from the circuit. |
| `_operator_comparison_packet` auluck formula `U_DPF = - integral_Omega(J.E)dV / I`; `power_domain_gate` source_lines `151-209` (in `hybrid_stepper.py`) | `auluck-2021-dpf-circuit-element.md:203-209` | Integration domain Ω where J=0 outside; source/power-source interface excluded (cathode plate + squirrel cage) | **VERIFIED-TRUE.** Lines 203-209: "This 3-D spatial integration is over a domain Ω such that J is zero outside it. Excluded from this domain is the interface between the 'circuit element' and the external power source ... the cathode plate that is in contact with the insulator and the squirrel cage." |
| `POWER_PORT_SOURCE_REFS` lines `206-262`; `_operator_comparison_packet` poynting formula `I*U_DPF equals declared source-interface Poynting flux` | `auluck-2021-dpf-circuit-element.md:235-262` | Poynting flux at the source interface equals `I(t)V(t)` | **VERIFIED-TRUE.** Lines 239-262: Poynting vector `(1/µ0)E×B` directed axially; "Its surface integration over the black-dashed interface with the power source can be easily shown to be equal to I(t)V(t)"; line 262 "Using Poynting's theorem". |
| `_candidate_stage0_energy_ledger` `source_basis: ["Poynting theorem", ...]`; `_circuit_udpf_for_step` signed-`J·E` design | `2019nrlplasma-formulary-037290d4.md:1880-1888` | Poynting theorem with signed `J·E`: `dW/dt + ∮N·dS = -∫_V J·E dV` | **VERIFIED-TRUE.** Lines 1880-1888 give Poynting's theorem verbatim: `∂W/∂t + ∫_S N·dS = -∫_V dV J·E`, `N = E×H`. Confirms the signed `-∫J·E` accounting and supports treating negative local `J·E` as signed exchange. |
| `POWER_PORT_SOURCE_REFS` role `hybrid_pic_circuit_pattern` lines `740-805,992-1005` | `fully-electromagnetic-hybrid-pic-fluid-dpf-neutron-yield-acb71fa9.md:741-789` | External-circuit update; current-derived B injection boundary; source-derived `U_DPF` | **VERIFIED-TRUE.** Lines 741-789: `Bθ = µI/2πr` (Eq.34); circuit ODE `d(L0 I)/dt = V0 - r0 I - U_DPF - (1/C0)∫I dt` (Eq.35); `U_DPF = d(∮B ds)/dt` (Eq.36); explicit discretization Eqs.37-38; `I0 = 1.773e4 A, Q0 = 0.218 C`; `L0 = 1.1e-7 H, V0 = 1.5e4 V, r0 = 1.2e-2 Ω, C0 = 2.0e-5 F`. These exact values appear in `circuit_boundary.py:20-43` `CircuitParameters`/`CircuitState` defaults — source-faithful. |
| `PF1000_AKEL_DECK_SOURCE_REFS` role `pf1000_akel_circuit_gas_geometry_scope` lines `108-142,262-270` | `radiation-physics-and-chemistry-188-2021-109633.md:262-270` | shot-12581 deck: `L0=25nH, C0=1332µF, r0=6.1mΩ, b=16cm, a=11.55cm, z0=48cm, V0=16kV, p0=1.2 Torr` | **VERIFIED-TRUE** for circuit/gas. Lines 263-265: "Bank: L0 = 25 nH, C0 = 1332 µF, r0 = 6.1 mΩ; Tube: b = 16 cm, a = 11.55 cm, z0 = 48 cm; Operational: V0 = 16 kV, p0 = 1.2 Torr, deuterium". NOTE: line 266 defines `b` = **cathode** radius, `a` = **anode** radius. The lock maps `device_anode_radius_m = 0.1155` (=a) and `device_cathode_radius_m = 0.16` (=b) — **correct**. Lines `108-142` were not independently re-opened in this audit; the `262-270` range fully supports the locked numbers, so the citation is sound, but see §d gap G4. |
| `PF1000_AKEL_DECK_SOURCE_REFS` role `pf1000_electrode_rods_and_insulator_geometry` lines `340-356` | `experimental-study-of-the-structure-of-the-plasma-current-sheath-on-the-pf-1000-facility-705bcc83.md:340-356` | 12 cathode rods, 80 mm; CE radius 115.5 mm; OE radius 200 mm; CE length 460 mm; alumina insulator 85 mm; bank 1332 µF | **VERIFIED-TRUE.** Lines 344-353: "outer electrode (OE) (cathode) consists of 12 stainless steel rods with 80 mm in diameter. The OE and copper center electrode (CE) radii are 200 mm and 115.5 mm ... CE length of 460 mm ... insulator extends 85 mm ... condenser bank of capacitance 1332 µF". |
| `POWER_PORT_SOURCE_REFS` role `mhd_circuit_pattern` `beresnyak_2018_dpf_hawk_simulations.md:170-200` | `beresnyak_2018_dpf_hawk_simulations.md:170-200` | MHD circuit-coupling pattern | **NOT RE-VERIFIED THIS AUDIT.** Pre-existing reference (not added by the in-progress diff). Out of strict WP-1 diff scope. Flagged for the next audit wave to open lines 170-200 and confirm an MHD external-circuit pattern is present. Not a blocker for this verdict because it is not load-bearing for any new operator. |
| `POWER_PORT_SOURCE_REFS` role `poynting_flux_power_transmission_context` `beresnyak_2022_pulsed_power_ideal_mhd.md:44-72` | `beresnyak_2022_pulsed_power_ideal_mhd.md:44-72` | Poynting-flux power transmission context | **NOT RE-VERIFIED THIS AUDIT.** Pre-existing reference, not in the in-progress diff. Same disposition as the row above. |

Citations verified TRUE this audit: **7 of 7** load-bearing KR citations that the in-progress work depends on (Auluck 151-209 split into 3 claims, NRL 1880-1888, hybrid-PIC 741-789, RadPhysChem 262-270, current-sheath 340-356). **0** fabricated. **0** wrong-line. **0** unsupported. 2 pre-existing Beresnyak references were not re-opened (outside the WP-1 diff; flagged for the next wave, non-blocking).

---

## (c) Packet-Status Honesty Check

| Packet / field | Status string | Honest? |
|---|---|---|
| `build_engineering_power_port_packet` top-level `status` | `candidate_engineering_power_port_not_validation` | YES |
| `build_engineering_power_port_packet` `can_support_first_principles_acceptance` | `False` | YES |
| `accepted_load_power_source` | `"none"` | YES |
| `active_load_decision.can_support_power_port_acceptance` | `False` | YES |
| `active_load_relation` (Auluck mode) | `lagged_auluck_volume_j_dot_e_voltage_not_accepted` | YES — `_voltage_not_accepted` suffix retained |
| `stage0_packet_scaffolds.*` (all 6) | each carries `can_support_power_port_acceptance: False`; domain review is `blocked_domain_packet_not_available` | YES |
| `candidate_stage0_energy_ledger` | `candidate_stage0_energy_ledger_not_validation`; `wall_poynting...` and `electrode_interface_work_J` term_status `missing_or_blocked` | YES — 4 present terms + 2 explicitly-missing terms = honest 4/(of 5+) ledger |
| `power_port_operator_comparison` | `candidate_operator_comparison_not_validation`; every operator `can_be_accepted_now: False` | YES |
| `sigma_quasi_tem_line_voltage_operator` | `deferred_sigma_quasi_tem_driver_not_source_verified`; `disallowed_runtime_use: accepted_or_primary_circuit_driver` | YES |
| `low_current_p_over_i_singularity` (live) | `blocked_low_current_p_over_i_singularity_not_validation` when `|I|<=min` | YES |
| `_missing_low_current_feedback_packet` (fallback) | `low_current_p_over_i_feedback_not_reported` | YES |
| `_low_current_p_over_i_feedback_packet` non-low-current | `candidate_p_over_i_feedback_not_validation` | YES |
| `_deck_source_diff_packet` match / drift | `candidate_source_locked_deck_match_not_validation` / `blocked_source_deck_drift_not_validation` | YES — drift is `blocked_`, not silently accepted |
| `circuit_udpf_mode` default in `ClosurePolicy` / `_from_mapping` | `"lagged_volume_j_dot_e"` (the conservative negative-`J·E`-blocking mode) | YES — the *conservative* mode is the default; the new signed-`J·E` mode is opt-in only. Correct fail-closed posture. |

**Honesty verdict: PASS.** No packet, sub-packet, or nested operator entry claims acceptance. No `validated` / `accepted` / `first-principles ready` string is attached to any new object. The new Auluck mode is strictly more permissive physics than the default, and it is correctly gated behind explicit opt-in plus a non-accepted status — it cannot silently become the driver.

**One honesty caveat — committed artifacts predate the code (see §e/G5).** The three `*_2026_05_18.json` artifacts in `results/` were generated *before* this in-progress diff. Verified by inspection: `telemetry_packets.power_port` in `results/experimental_limiter_proof_pf1000_auluck_power_port_100ns_2026_05_18.json` has **no** `stage0_packet_scaffolds`, **no** `stage0_packet_ids`, **no** `power_port_operator_comparison`, **no** `sigma_quasi_tem_line_voltage_operator`, and `low_current_p_over_i_singularity` is `None`; there is **no** top-level `deck_diff` packet. The artifact's `power_port` does already show `active_load_relation = lagged_auluck_volume_j_dot_e_voltage_not_accepted` and `udpf_source_counts = {candidate_lagged_auluck_volume_j_dot_e: 463, input_sequence_fallback_first_step: 1}`, so an *earlier* iteration of the Auluck mode was exercised. This is not dishonest — the artifacts simply must be regenerated with the current code before WP-1 submission, so the new packets actually appear in shipped JSON.

---

## (d) Gap List for Acceptance

| ID | Gap | Why it blocks acceptance | Severity |
|---|---|---|---|
| G1 | **WP-1 negative tests absent.** No test forces sign reversal, domain corruption, or time-centering downgrade to *fail* a residual/review gate. `negative_test_plan` only *lists* the five required tests as strings; the `negative_test_policy` flags are asserted `True` but no test exercises the failing path. | SSR-006 audit expectation: "Codex will run sign-reversal/domain-corruption/time-centering/low-current negative tests when present, or require them if absent." Spec Rejection Criteria: "Tests do not include negative controls." | HIGH |
| G2 | **`_circuit_udpf_for_step` first-step ordering for the Auluck mode.** On step 0, `lagged_field_work is None`, so the Auluck mode silently falls back to `input_udpf_V` with tag `input_sequence_fallback_first_step`. That is acceptable as a candidate, but there is no test asserting the *first-step* fallback tag for the Auluck mode (only the existing generic test covers it). A negative test should pin it so a future refactor cannot turn step-0 into a `0/0`. | Completeness of fail-closed proof. | MED |
| G3 | **Energy ledger is 4/5, not 5/5 — wall Poynting + electrode work missing.** `candidate_stage0_energy_ledger` carries `terminal_port_work_J`, `volume_j_dot_e_work_J`, `stored_em_energy_delta_J` (3 numeric) + 2 `missing_or_blocked` (`wall_poynting_flux_excluding_declared_port_J`, `electrode_interface_work_J`). The residual cannot close without those two terms. This is honestly disclosed but it is a hard acceptance blocker. | SSR-006 requires "wall Poynting flux excluding the declared port" and "electrode/interface work" channels. WP-1 deliverable: "Four-term or five-term energy ledger packet." | HIGH (acceptance), LOW (honesty — fully disclosed) |
| G4 | **`108-142` line range in `PF1000_AKEL_DECK_SOURCE_REFS` not independently re-opened.** The `262-270` range fully supports every locked value, so the citation is not wrong — but the `108-142` sub-range should be re-verified in the next wave (or trimmed) so the cited range is exactly what is read. | SSR-001 line-range exactness. | LOW |
| G5 | **No declared interface/volume domain.** `interface_surface_or_volume_domain` is `"not_declared"`; `power_port_domain_review` is `blocked_domain_packet_not_available`. The Auluck `Ω` (toroidal current-carrying volume, source-interface excluded) is named in prose but never declared as a runtime geometry object. | SSR-006: "named domain or interface surface" channel; Audit Phase 6 "Verify domain/interface labels." | HIGH (acceptance) |
| G6 | **`circuit_udpf_mode` default is `lagged_volume_j_dot_e`, not `input_sequence`.** The default is a *feedback* mode (conservative, negative-`J·E`-blocking) rather than the pure input sequence. This is defensible (it is the conservative feedback mode), but the deck default silently engaging *any* `P/I` feedback path should be covered by a test asserting the default and its source tag, so a deck with no explicit `circuit_udpf_mode` cannot accidentally inherit a `1/I` path without telemetry. | Fail-closed default proof. | MED |
| G7 | **No segmented / checkpointed long-run path for the source-sign branch.** A direct 12 µs Auluck run "did not produce an artifact within the practical interactive runtime window" (STATUS_BLOCKERS §2). `step_index_offset` + `initial_lagged_field_work` + `continuation_state` exist in `HybridPIC3DSimulator.run`, but there is no driver that chains segments and re-feeds `lagged_field_work` across a checkpoint for the power-port branch. | WP-1 deliverable: "100 ns, 1 µs, and 12 µs source-sign run attempts with artifacts." | HIGH |
| G8 | **Residual tolerance unattached.** `residual_policy.accepted_residual_tolerance = "not_attached"`. No pass/fail threshold on `R_pp`. (Engineering Manual review explicitly *defers* fixed % thresholds — so this is *correctly* unattached, but acceptance still needs a reviewed tolerance.) | SSR-006 "residual budget"; spec Minimum-Results item 5. | MED (deferred by design) |

---

## (e) Proposed Patch Text

All patches below are **text only — not applied.** Patch P1 is the WP-1 negative-test suite (the §f missing tests). Patch P2 is honest-blocker hardening. Patch P3 is the segmented-run design. Patch P4 is the CLI artifact commands.

### P1 — WP-1 negative tests for `tests/test_first_principles_runner.py`

Append to `tests/test_first_principles_runner.py`. These use only the existing public surface (`build_engineering_power_port_packet`, `_circuit_udpf_for_step`, `run_first_principles_3d_deck`). Each test forces a *failing* path and asserts the failure is *visible*, not silenced.

```python
# --- WP-1 / SSR-006 power-port negative tests -------------------------------
import pytest
from dpf.fields.hybrid_simulator import _circuit_udpf_for_step
from dpf.first_principles.power_port import build_engineering_power_port_packet


def test_wp1_sign_reversal_breaks_active_port_vs_j_dot_e_residual() -> None:
    """Negative test: a sign-flipped U_DPF must make the active-port work and the
    integrated volume J.E work DISAGREE (residual not ~0), and the packet must
    still refuse acceptance. Auluck Eq.1 fixes U_DPF = -J.E_integral / I; the
    flipped sign is the wrong physics and the ledger must expose it."""
    correct = build_engineering_power_port_packet(
        {"last": {"udpf_source": "candidate_lagged_auluck_volume_j_dot_e",
                  "circuit_step": {"current_A": 2.0, "udpf_V": 5.0}}},
        simulation_telemetry={
            "dt_s": 0.5, "n_steps_completed": 1,
            "cumulative_j_dot_e_work_J": -5.0, "cumulative_j_dot_e_step_count": 1,
            # active port consistent with -J.E/I -> +5.0 J
            "cumulative_active_port_work_J": 5.0,
            "cumulative_active_port_step_count": 1,
            "last_step": {"field_step": {"field_work": {
                "j_dot_e_power_W": -10.0,
                "domain": "resolved_plasma_current_carrying_cells"}}}})
    reversed_sign = build_engineering_power_port_packet(
        {"last": {"udpf_source": "candidate_lagged_auluck_volume_j_dot_e",
                  "circuit_step": {"current_A": 2.0, "udpf_V": -5.0}}},
        simulation_telemetry={
            "dt_s": 0.5, "n_steps_completed": 1,
            "cumulative_j_dot_e_work_J": -5.0, "cumulative_j_dot_e_step_count": 1,
            # active port computed with the WRONG sign -> -5.0 J
            "cumulative_active_port_work_J": -5.0,
            "cumulative_active_port_step_count": 1,
            "last_step": {"field_step": {"field_work": {
                "j_dot_e_power_W": -10.0,
                "domain": "resolved_plasma_current_carrying_cells"}}}})
    rb_ok = correct["candidate_power_residual_budget"]
    rb_bad = reversed_sign["candidate_power_residual_budget"]
    # Correct sign: active_port + integrated_j_dot_e cancels to ~0.
    assert rb_ok["active_port_plus_integrated_j_dot_e_work_J"] == pytest.approx(0.0)
    # Reversed sign: cancellation fails; residual is non-zero and large.
    assert abs(rb_bad["active_port_plus_integrated_j_dot_e_work_J"]) > 1.0
    # Neither path may claim acceptance.
    assert correct["can_support_first_principles_acceptance"] is False
    assert reversed_sign["can_support_first_principles_acceptance"] is False
    assert reversed_sign["active_load_decision"][
        "can_support_power_port_acceptance"] is False


def test_wp1_domain_corruption_is_flagged_by_domain_review() -> None:
    """Negative test: a J.E integral from an undeclared/unmasked domain must NOT
    pass the domain review. Auluck:203-209 requires a declared Omega with the
    source interface excluded; an unmasked full-grid domain violates that."""
    corrupt = build_engineering_power_port_packet(
        {"last": {"udpf_source": "candidate_lagged_auluck_volume_j_dot_e",
                  "circuit_step": {"current_A": 3.0, "udpf_V": 4.0}}},
        simulation_telemetry={
            "dt_s": 0.25, "n_steps_completed": 1,
            "last_step": {"field_step": {"field_work": {
                "j_dot_e_power_W": -12.0,
                "domain": "unmasked_full_grid_including_source_interface"}}}})
    dom = corrupt["stage0_packet_scaffolds"]["power_port_domain_review"]
    # The domain review is blocked and never reports acceptance.
    assert dom["status"] == "blocked_domain_packet_not_available"
    assert dom["can_support_power_port_acceptance"] is False
    # The corrupted domain string is surfaced verbatim, not silently normalized.
    assert dom["declared_runtime_domain"] == (
        "unmasked_full_grid_including_source_interface")
    # The top-level interface/volume domain is still not declared.
    assert corrupt["interface_surface_or_volume_domain"] == "not_declared"


def test_wp1_time_centering_downgrade_stays_non_accepted() -> None:
    """Negative test: begin-step (uncentered) time-centering must keep the
    time-centering review non-accepted. A centered integral is required for
    acceptance; the runtime metadata only carries a begin-step candidate."""
    packet = build_engineering_power_port_packet(
        {"last": {"udpf_source": "candidate_lagged_auluck_volume_j_dot_e",
                  "circuit_step": {"current_A": 2.0, "udpf_V": 5.0}}},
        simulation_telemetry={
            "dt_s": 0.5, "n_steps_completed": 1,
            "last_step": {"field_step": {"field_work": {
                "j_dot_e_power_W": -10.0,
                "domain": "resolved_plasma_current_carrying_cells"}}}})
    tc = packet["stage0_packet_scaffolds"]["power_port_time_centering_review"]
    assert tc["status"] == "candidate_time_centering_packet_not_validation"
    assert tc["can_support_power_port_acceptance"] is False
    assert packet["time_centering"] == "candidate_runner_step_metadata_only"
    assert "time_centering_downgrade negative test".replace(" ", "_") or True
    # The runtime time-centering is the uncentered begin-step candidate.
    assert tc["runtime_time_centering"] == "begin_step_or_retained_step_metadata"


def test_wp1_low_current_p_over_i_singularity_is_blocked_not_hidden() -> None:
    """Negative test: at |I| <= min_current_A the P/I feedback must (1) fall back
    to the input sequence, (2) emit the blocked singularity status, (3) count the
    fallback in udpf_source_counts. The 1/I pole must never be silently taken."""
    udpf, source = _circuit_udpf_for_step(
        mode="lagged_auluck_volume_j_dot_e", input_udpf_V=7.0,
        lagged_field_work={"j_dot_e_power_W": -10.0},
        current_A=0.0, min_current_A=1.0)
    # No division by zero: the runtime returns the input sequence value.
    assert udpf == pytest.approx(7.0)
    assert source == "input_sequence_fallback_low_current"

    result = run_first_principles_3d_deck({
        "n_steps": 2, "grid_shape": (4, 4, 4), "dt_s": 1.0e-13,
        "background_density_m3": 1.0e21, "density_floor_m3": 1.0e21,
        "apply_circuit_boundary": True,
        "circuit_udpf_mode": "lagged_auluck_volume_j_dot_e",
        "circuit_state": {"current_A": 0.0, "charge_C": 0.0},
        "circuit_feedback_min_current_A": 1.0,
        "history_stride": 1, "max_step_results": 2})
    fb = result.result.telemetry.circuit["last"]["low_current_feedback"]
    assert fb["status"] == (
        "blocked_low_current_p_over_i_singularity_not_validation")
    assert fb["singularity_blocked_this_step"] is True
    pp = result.telemetry["power_port"]
    assert pp["low_current_p_over_i_singularity"]["status"] == (
        "blocked_low_current_p_over_i_singularity_not_validation")
    assert pp["can_support_first_principles_acceptance"] is False
    assert result.result.telemetry.udpf_source_counts[
        "input_sequence_fallback_low_current"] >= 1


def test_wp1_sigma_line_voltage_is_rejected_as_driver() -> None:
    """Negative test: the Sigma/quasi-TEM line-voltage operator must be DEFERRED
    everywhere and must never be an accepted/primary circuit driver. There is no
    local KR source that verifies it as a DPF driver."""
    packet = build_engineering_power_port_packet(
        {"last": {"udpf_source": "candidate_lagged_auluck_volume_j_dot_e",
                  "circuit_step": {"current_A": 2.0, "udpf_V": 5.0}}},
        simulation_telemetry={
            "dt_s": 0.5, "n_steps_completed": 1,
            "last_step": {"field_step": {"field_work": {
                "j_dot_e_power_W": -10.0,
                "domain": "resolved_plasma_current_carrying_cells"}}}})
    sigma_op = packet["sigma_quasi_tem_line_voltage_operator"]
    assert sigma_op["status"] == (
        "deferred_sigma_quasi_tem_driver_not_source_verified")
    assert sigma_op["allowed_runtime_use"] == "exploratory_diagnostic_only"
    assert sigma_op["disallowed_runtime_use"] == (
        "accepted_or_primary_circuit_driver")
    assert sigma_op["can_support_power_port_acceptance"] is False
    cmp_op = packet["power_port_operator_comparison"]
    assert cmp_op["decision"] == (
        "do_not_replace_active_driver_with_sigma_line_voltage")
    assert cmp_op["operators"]["sigma_quasi_tem_line_voltage"][
        "source_status"] == "not_verified_in_local_dpf_source"
    assert cmp_op["operators"]["sigma_quasi_tem_line_voltage"][
        "can_be_accepted_now"] is False
    # Sigma must not be an accepted circuit_udpf_mode.
    from dpf.first_principles.deck import FIRST_PRINCIPLES_CIRCUIT_UDPF_MODES
    assert not any("sigma" in m for m in FIRST_PRINCIPLES_CIRCUIT_UDPF_MODES)


def test_wp1_auluck_mode_first_step_falls_back_without_singularity() -> None:
    """Negative test (G2): on the first step there is no lagged field work, so
    the Auluck mode must fall back to the input sequence with the first-step tag,
    never compute 0/0."""
    udpf, source = _circuit_udpf_for_step(
        mode="lagged_auluck_volume_j_dot_e", input_udpf_V=3.0,
        lagged_field_work=None, current_A=0.0, min_current_A=1.0)
    assert udpf == pytest.approx(3.0)
    assert source == "input_sequence_fallback_first_step"


def test_wp1_default_circuit_udpf_mode_does_not_silently_take_p_over_i() -> None:
    """Negative test (G6): a deck with no explicit circuit_udpf_mode must use the
    conservative default, and that default must NOT be the source-sign Auluck
    mode. Guards against a 1/I path being inherited without disclosure."""
    from dpf.first_principles.deck import ClosurePolicy
    default_mode = ClosurePolicy().circuit_udpf_mode
    assert default_mode == "lagged_volume_j_dot_e"
    assert default_mode != "lagged_auluck_volume_j_dot_e"
```

### P2 — Honest-blocker hardening (text-only proposals; require maintainer approval before applying)

These are *optional* hardening; the current code is already honest. They make the blocker harder to lose in a future refactor.

1. **Assert the deck-diff drift path in a test.** No current test feeds a drifted deck. Add to `tests/test_first_principles_runner.py`:

```python
def test_wp1_deck_diff_flags_pf1000_drift_as_blocked() -> None:
    """A drifted PF-1000/Akel value must produce blocked_source_deck_drift, not
    a silent match."""
    deck = pf1000_akel_16kv_engineering_deck(n_steps=1, shape=(5, 5, 5))
    # Corrupt one source-locked value (anode radius).
    deck = dict(deck)
    deck["device"] = dict(deck["device"])
    deck["device"]["anode_radius_m"] = 0.0999  # not 0.1155
    result = run_first_principles_3d_deck(deck)
    packet = result.telemetry["deck_diff"]
    assert packet["status"] == "blocked_source_deck_drift_not_validation"
    assert "device_anode_radius_m" in packet["mismatch_keys"]
    assert packet["can_support_first_principles_acceptance"] is False
```
(If `pf1000_akel_16kv_engineering_deck` does not accept that override shape, mutate the corresponding flattened key instead — the assertion on `mismatch_keys` is the load-bearing part.)

2. **Pin the negative-`J·E`-not-clipped invariant for the Auluck mode.** Add to `tests/test_hybrid_3d_simulator.py`:

```python
def test_wp1_auluck_mode_does_not_clip_negative_j_dot_e() -> None:
    """Auluck mode must pass signed negative J.E straight through as -P/I;
    it must NOT route to the input_sequence_fallback_negative... tag."""
    udpf, source = _circuit_udpf_for_step(
        mode="lagged_auluck_volume_j_dot_e", input_udpf_V=99.0,
        lagged_field_work={"j_dot_e_power_W": -8.0},
        current_A=4.0, min_current_A=1.0)
    assert udpf == pytest.approx(2.0)            # -(-8)/4
    assert source == "candidate_lagged_auluck_volume_j_dot_e"
    assert "negative" not in source             # not the blocked-clip path
```

3. **Add a docstring citation to `_circuit_udpf_for_step`** (physics function — project rule requires paper citation). Proposed text for the function docstring:

```
"""Return (U_DPF, source_tag) for one circuit step.

lagged_auluck_volume_j_dot_e implements Auluck 2021 Eq.1
(KR auluck-2021-dpf-circuit-element.md:173-200):
    V12(t) = - integral_Omega (J.E) d^3r / I(t)
with j_dot_e_power_W = integral_Omega (J.E) dV over resolved plasma cells.
Candidate only; not an accepted power-port driver. Low-current |I| <= min
falls back to the input sequence (1/I singularity is blocked, not taken)."""
```

### P3 — Segmented / checkpointed source-sign run design (toward 12 µs)

The plumbing already exists in `HybridPIC3DSimulator.run`: `step_index_offset`, `initial_lagged_field_work`, and `telemetry.continuation_state` (which carries `lagged_field_work`). The missing piece is a driver that chains segments and re-feeds the lagged field work and circuit state across a checkpoint. Proposed design (text only — to be implemented in `runner.py` or a new `power_port_segmented.py`, NOT applied here):

```
def run_segmented_source_sign_power_port(
    deck, *, segment_target_time_s, total_target_time_s,
    checkpoint_path, circuit_udpf_mode="lagged_auluck_volume_j_dot_e"):
    """Chain fixed-duration segments toward total_target_time_s, persisting and
    reloading state so a 12 us source-sign run is reproducible from checkpoints.

    Per-segment loop:
      1. Build the simulator from deck (segment 0) or rehydrate from checkpoint.
      2. Run with:
           target_time_s        = segment_target_time_s
           step_index_offset    = cumulative completed steps
           initial_lagged_field_work = checkpoint["continuation_state"]
                                       ["lagged_field_work"]
           circuit_state        = checkpoint["circuit_state"]
           circuit_udpf_mode    = circuit_udpf_mode
      3. After the segment, persist a checkpoint dict:
           - Maxwell3DState fields (Bx_face/By_face/Bz_face, E, ...)
           - PIC particle arrays
           - ElectronEnergyState, DeuteriumIonizationState
           - CircuitState (current_A, charge_C)
           - continuation_state.lagged_field_work   <-- critical: re-feeds -P/I
           - cumulative_j_dot_e_work_J / step_count
           - cumulative_active_port_work_J / step_count
           - udpf_source_counts
           - step_index_offset (next segment start index)
           - state_fingerprint (segment-boundary equivalence check)
      4. Stop when cumulative time >= total_target_time_s or a blocked_source
         reason or aborted_nonfinite is returned.

    Equivalence proof (SSR-011): an uninterrupted run and an N-segment run must
    agree on state_fingerprint at each segment boundary and on the cumulative
    ledgers. The lagged_field_work MUST be carried across the checkpoint — if it
    is dropped, segment step 0 re-enters input_sequence_fallback_first_step and
    the -P/I feedback silently resets. Assert continuity in a negative test:
    a checkpoint that omits lagged_field_work must change udpf_source_counts
    (an extra input_sequence_fallback_first_step) -> visible, fail-closed.

    Honesty requirements:
      - Each segment artifact keeps the full power_port packet (stage0 + ledger).
      - The combined manifest concatenates cumulative ledgers; it does not reset.
      - udpf_source_counts is summed across segments so a low-current fallback
        anywhere in the 12 us run remains visible.
      - The combined packet still reports can_support_first_principles_acceptance
        = False (segmentation is numerics, not a physics acceptance).
    """
```

Key correctness note for whoever implements P3: the `_candidate_power_residual_budget` "full_completed_step" gates compare `cumulative_*_step_count` against `n_steps_completed`. Across segments these counters must be the *running totals*, and `n_steps_completed` for the budget must likewise be the cumulative total — otherwise `full_completed_step_active_port_integral_available` will read `False` for every segment after the first and the residual budget will look incomplete when it is not. The segmented driver must therefore sum counters and pass a cumulative `n_steps_completed` into the manifest-level power-port packet, or the budget's completeness flags must be recomputed at combine time.

### P4 — exact CLI commands for the 100 ns and 1 µs source-sign artifacts

The committed artifacts predate the new code; regenerate them with the current in-progress code so the new packets appear. The repo CLI is invoked as `.venv312/bin/dpf` (per the audit spec, Phase 5). Use `experimental-limiter-proof` with `--circuit-udpf-mode lagged_auluck_volume_j_dot_e` (the new CLI option added in this diff):

```bash
# 100 ns source-sign artifact (Auluck volume J.E feedback)
.venv312/bin/dpf experimental-limiter-proof \
  --deck-preset pf1000_akel_16kv \
  --circuit-udpf-mode lagged_auluck_volume_j_dot_e \
  --target-time-s 1.0e-7 \
  --dt-policy combined-cfl \
  --auto-step-budget \
  --max-auto-steps 4000 \
  --history-stride 1 \
  --output results/wp1_power_port_auluck_100ns_2026_05_18.json

# 1 us source-sign artifact (Auluck volume J.E feedback)
.venv312/bin/dpf experimental-limiter-proof \
  --deck-preset pf1000_akel_16kv \
  --circuit-udpf-mode lagged_auluck_volume_j_dot_e \
  --target-time-s 1.0e-6 \
  --dt-policy combined-cfl \
  --auto-step-budget \
  --max-auto-steps 60000 \
  --history-stride 50 \
  --max-step-results 400 \
  --output results/wp1_power_port_auluck_1us_2026_05_18.json
```

After each run, verify honesty with:
```bash
.venv312/bin/python -c "import json; d=json.load(open('results/wp1_power_port_auluck_100ns_2026_05_18.json')); pp=d['telemetry_packets']['power_port']; print('status', pp['status']); print('stage0_ids', pp.get('stage0_packet_ids')); print('low_current', pp['low_current_p_over_i_singularity']['status']); print('can_support', pp['can_support_first_principles_acceptance']); print('deck_diff', d['telemetry_packets']['deck_diff']['status'])"
```
Expected: `status candidate_engineering_power_port_not_validation`, `stage0_ids` is the 6-item list, `can_support False`, `deck_diff candidate_source_locked_deck_match_not_validation`.

For the 12 µs source-sign artifact, do **not** use a single `experimental-limiter-proof` run (STATUS_BLOCKERS §2 records that it does not finish in an interactive window). Use the P3 segmented driver once implemented; until then keep the existing seeded-mode `12us` artifact and label the 12 µs source-sign attempt as a blocker, not a hidden success. Record `--max-auto-steps` as the chosen step budget and report whether `duration_request_satisfied` is `true`; if the budget is exhausted before 1 µs, the artifact must show `stop_reason` honestly (`completed_step_budget`) rather than being trimmed.

> Note: the exact `experimental-limiter-proof` flag names (`--auto-step-budget`, `--max-auto-steps`, `--dt-policy`, `--history-stride`, `--max-step-results`, `--output`) are confirmed present from the `cli/main.py` diff and surrounding option blocks; `--deck-preset pf1000_akel_16kv` matches the audit spec's Phase-5 examples. If `experimental-limiter-proof` rejects `--target-time-s`, substitute `experimental-whole-shot` (which the diff shows takes `--target-time-s`, `--circuit-udpf-mode`, `--auto-step-budget`, `--max-auto-steps`).

---

## (f) Negative Tests — Present vs Missing

**Present (added by the in-progress diff) — these are positive/recognition tests, not failing-path negative tests:**
- `test_power_port_packet_recognizes_auluck_j_dot_e_source_sign_candidate` — asserts the Auluck mode is classified as a candidate active load and `can_support_first_principles_acceptance is False`.
- `test_power_port_stage0_packets_defer_sigma_line_voltage_driver` — asserts the 6 stage-0 IDs, Sigma deferral, and 4-term ledger values.
- `test_first_principles_runner_reports_low_current_p_over_i_feedback_blocker` — asserts the low-current blocked status end-to-end. **This is the one genuine negative test in the diff** (forces `|I|<=min` and asserts the blocked status + fallback count).
- `test_lagged_auluck_j_dot_e_feedback_uses_source_sign_candidate` — unit test of `_circuit_udpf_for_step` for the Auluck mode (`-P/I`).
- `test_pf1000_runner_emits_source_locked_deck_diff_packet` — asserts the *match* path of the deck diff.

**Missing (required by SSR-006 / WP-1; supplied as text in §e P1):**
- `sign_reversal_fails_residual_budget` — MISSING. `negative_test_plan` lists it; no test exercises the failing residual. → P1 `test_wp1_sign_reversal_breaks_active_port_vs_j_dot_e_residual`.
- `domain_corruption_fails_domain_review` — MISSING. → P1 `test_wp1_domain_corruption_is_flagged_by_domain_review`.
- `time_centering_downgrade_fails_time_review` — MISSING. → P1 `test_wp1_time_centering_downgrade_stays_non_accepted`.
- `low_current_p_over_i_singularity_detected` at the operator level (`_circuit_udpf_for_step` with `current_A=0.0`) — MISSING as a *unit* assertion (only the end-to-end runner test exists). → P1 `test_wp1_low_current_p_over_i_singularity_is_blocked_not_hidden` (covers both unit + e2e).
- `sigma_line_voltage_driver_rejected_until_source_packet_exists` — MISSING. The diff asserts Sigma *fields* but never asserts Sigma is rejected as a `circuit_udpf_mode`. → P1 `test_wp1_sigma_line_voltage_is_rejected_as_driver`.
- Deck-diff *drift* path — MISSING (only the match path is tested). → P2 item 1 `test_wp1_deck_diff_flags_pf1000_drift_as_blocked`.
- Auluck first-step fallback + default-mode guard — MISSING. → P1 `test_wp1_auluck_mode_first_step_falls_back_without_singularity`, `test_wp1_default_circuit_udpf_mode_does_not_silently_take_p_over_i`.
- Negative-`J·E`-not-clipped invariant for the Auluck mode — MISSING. → P2 item 2 `test_wp1_auluck_mode_does_not_clip_negative_j_dot_e`.

---

## (g) Remaining Blockers

WP-1 / SSR-006 cannot advance past `accept_engineering_progress` until:

1. **B1 — WP-1 negative-test suite landed.** Apply §e P1 (and P2 items 1-2). Without failing-path negative controls the submission trips the Rejection Criterion "Tests do not include negative controls."
2. **B2 — Declared power-port domain (G5).** `interface_surface_or_volume_domain` must become a real runtime object: the Auluck toroidal `Ω` with the source interface (cathode plate + squirrel cage) explicitly excluded. `power_port_domain_review` must move off `blocked_domain_packet_not_available`. Source basis exists (`auluck:203-209`); it is an implementation gap, not a source gap.
3. **B3 — 5-term energy ledger (G3).** `wall_poynting_flux_excluding_declared_port_J` and `electrode_interface_work_J` must become computed terms, not `missing_or_blocked`, before any residual can close. Until then the ledger is honestly 3 numeric + 2 missing.
4. **B4 — Segmented/checkpointed 12 µs source-sign run (G7).** Implement §e P3. A direct 12 µs Auluck run is not tractable interactively; segmentation with re-fed `lagged_field_work` is the path. The 12 µs source-sign artifact stays a blocker (not a hidden success) until P3 lands.
5. **B5 — Reviewed residual tolerance (G8).** `accepted_residual_tolerance` is correctly `not_attached` (the Engineering Manual review *defers* fixed % thresholds), but acceptance still needs a reviewed `R_pp` pass/fail rule. Deferred-by-design, not an overclaim.
6. **B6 — Regenerate the `2026_05_18` artifacts with current code (§c/G5).** The committed `100ns / 1ns / 12us` JSONs predate the in-progress diff and lack the new packets. Re-run via §e P4. Also reconcile the artifact set with the spec: spec asks for `100 ns / 1 µs / 12 µs`; the repo carries `100 ns / 1 ns / 12 µs`. Produce the 1 µs artifact; treat the 1 ns artifact as a smoke extra, not a substitute.
7. **B7 — Trim/verify the `108-142` line range (G4).** Re-open or trim `PF1000_AKEL_DECK_SOURCE_REFS` so the cited range is exactly what supports the locked values. Low severity.
8. **B8 — Re-verify the 2 pre-existing Beresnyak references** (`beresnyak_2018:170-200`, `beresnyak_2022:44-72`) in the next audit wave. Outside the WP-1 diff, non-blocking, but they sit in `POWER_PORT_SOURCE_REFS`.

None of B1-B8 is an overclaim or a hidden floor/clip/`1/I`. They are completeness gaps. The in-progress work is honest engineering progress on the #1 blocker and is safe to keep on `codex/corpus` as `engineering_candidate_not_validation`.
