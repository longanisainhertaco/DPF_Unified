# CodexFindings

Date: 2026-05-05

Scope reviewed: `/Users/anthonyzamora/dpf-unified`, including the README, engine/circuit/fluid/MLX paths, diagnostics, radiation modules, validation tests, and the local `KnowledgeReference/*.md` corpus. Scientific judgments below use only the local KnowledgeReference files as source of truth.

Plan note 2026-05-05:

- The detailed forward plan requested as `CortexFindings.md` has been created in the repository. `CodexFindings.md` remains the running findings and ratchet log, while `CortexFindings.md` records the reviewed plan sequence from target authority through end-to-end high-fidelity demonstration.

## Verdict

I am against treating the current project as a scientifically validated, end-to-end Dense Plasma Focus MHD simulation tool today.

I am for treating it as a useful engineering scaffold with several credible, source-backed components: circuit/RLC integration, Lee/snowplow waveform fitting, Bosch-Hale thermonuclear rates, Lee/Saw beam-target yield estimates, NRL bremsstrahlung/transport pieces, and some Goyon-style reduced-order pinch scalings. The current implementation is not yet a self-consistent predictive MHD/kinetic DPF simulator. The most important gap is that late pinch and neutron production in the KnowledgeReference are repeatedly described as kinetic/beam/instability dominated, while large parts of this code still use snowplow, empirical, or internally flagged approximations.

## Finding 1: Pure-MLX MHD mode does not actually feed MHD inductance back into the circuit

Severity: Critical

Code evidence:

- `README.md:3` describes the project as a resistive MHD simulator with circuit coupling.
- `README.md:75` lists "MHD-circuit coupling (density-weighted L_p)" as tested in `metal/mlx_coupling.py` and `metal/mlx_engine.py`.
- `src/dpf/metal/mlx_engine.py:1-6` and `src/dpf/metal/mlx_engine.py:82-85` state that MHD density-weighted plasma inductance feeds back into the circuit in MHD mode.
- In the actual pure-MLX loop, `src/dpf/metal/mlx_engine.py:289-297` says the Poynting coupling is not wired and sets `Lp_mhd_val = Lp_sp`.
- `src/dpf/metal/mlx_engine.py:300` steps the circuit with `Lp=Lp_sp`, `dLp_dt=dLp_dt_sp`, and `back_emf=0.0`.

KnowledgeReference basis:

- `KnowledgeReference/unlimited-release-printed-september-2009-alegra-hedp-simulations-of-the-dense-plasma-focus.md:287-326` says MHD is useful through early pinch but charge separation, instabilities, and beam-target neutron production are outside ordinary MHD.
- `KnowledgeReference/neutron-generation-dynamics-inside-a-ma-class-dense-plasma-focus-z-pinch-5.md:174-215` describes MHD as useful for run-down and parts of run-in/liftoff, then transitions to kinetic modeling for finite Larmor and mean-free-path effects.
- `KnowledgeReference/the-dense-plasma-focus-a-versatile-dense-pinch-for-diverse-applications.md:184-190` says final pinch behavior involves instabilities, beam formation, finite Larmor radius effects, anomalous resistivity, and strong electric fields, so MHD must be complemented by kinetic/PIC modeling.

Explanation:

The README and pure-MLX docstring overstate what the advertised MLX API does. The MHD solver can run, but the pure-MLX circuit is still driven by snowplow inductance. That means pure-MLX MHD runs are not evidence that spatially resolved MHD is predicting the circuit waveform. They are, at best, snowplow-coupled circuit runs with an MHD side calculation.

Important nuance:

The main Python engine has a more honest hybrid path in `src/dpf/engine/circuit_coupling.py:87-138`, where MHD feedback can be blended after trust gating. But the pure-MLX engine explicitly bypasses that coupling, while the README names `metal/mlx_engine.py` as tested MHD-circuit coupling.

Required resolution:

Either wire MHD feedback into `run_mlx_discharge(mode="mhd")` or rename the mode/status so users understand that the circuit remains snowplow-driven. Validation claims should distinguish "MHD fields are advanced" from "MHD fields drive the circuit."

Ratchet update 2026-05-05:

- Module touched: `src/dpf/metal/mlx_engine.py`.
- The pure-MLX engine no longer aliases `Lp_mhd_nH` to snowplow `Lp_sp`.
- The engine now records `Lp_snowplow_nH` and `Lp_mhd_nH` separately, and `Lp_nH` is the actual circuit-loading inductance.
- MHD circuit influence is now trust-gated: the circuit remains snowplow-loaded until the MHD density-weighted Lp is finite, positive, in an accepted phase, and comparable to the analytic snowplow load; then it blends toward MHD Lp.
- Added regression coverage in `tests/test_mlx_circuit_coupling.py` so the MHD Lp series cannot silently become a snowplow alias again.
- Remaining scientific limit: this is still a density-weighted Lee-style coupling, not a first-principles Poynting/kinetic closure. It improves module honesty and data separation but does not close the late-pinch/beam-target gap identified by the KnowledgeReference.
- Verification status: `python3 -m py_compile src/dpf/metal/mlx_engine.py tests/test_mlx_circuit_coupling.py` passed; `python3 -m pytest tests/test_scaling_laws.py -q` passed. The focused MLX test group could not run in this environment because Python aborted while importing `mlx.core`.

## Finding 2: There is no single coherent Lee/RADPF implementation across the project

Severity: High

Code evidence:

- `src/dpf/metal/mlx_snowplow.py:4-12` claims Lee 5-phase equations, but `src/dpf/metal/mlx_snowplow.py:8` lists only rundown, radial inward shock, and pinch.
- `src/dpf/metal/mlx_snowplow.py:48-75` accepts `pinch_column_fraction` and stores `_pcf`, but the radial transition sets `_z_f = self._a * 1e-5` in `src/dpf/metal/mlx_snowplow.py:168-172`; the stored `_pcf` is not used there.
- `src/dpf/metal/mlx_snowplow.py:280-282` terminates when the shock reaches `0.01*a`.
- `src/dpf/metal/mlx_snowplow.py:321-339` returns `R_plasma: 0.0` and `get_R_plasma() == 0.0`.
- `src/dpf/fluid/snowplow.py:183-186` hard-codes `r_pinch_min = 0.17*a`, citing PF-1000-specific support.
- `src/dpf/fluid/snowplow.py:219-240` implements a two-step radial current fraction with a smooth 50 ns sigmoid.

KnowledgeReference basis:

- `KnowledgeReference/a-course-on-plasma-focus-numerical-experiments-s-lee-and-s-h-saw-part-1-basic-course.md:15069-15093` lays out the gross Lee/RADPF phases and gives deuterium model limits including `rmin = 0.13a`, maximum length `0.7a`, shock transit, and pinch lifetime.
- `KnowledgeReference/a-course-on-plasma-focus-numerical-experiments-s-lee-and-s-h-saw-part-1-basic-course.md:15569-15577` includes reflected-shock equations and says the radiative phase is not significant for deuterium gross parameters.
- `KnowledgeReference/a-course-on-plasma-focus-numerical-experiments-s-lee-and-s-h-saw-part-1-basic-course.md:16250-16256` warns that the radial model assumes instantaneous communication/infinite signal speed and gives speeds that are too high.
- `KnowledgeReference/faeton-i-investigation-of-plasma-dynamics-and-radiation-output-of-a-100-kv-plasma-focus-device.md:80-85` supports a two-step radial fitting idea for restrike/current-factor behavior, but not this exact smoothing law.

Explanation:

The project has multiple snowplow/Lee implementations that are not numerically or structurally equivalent. The MLX version advertises "5-phase" but stops at pinch and has no plasma resistance. The fluid version includes reflected/post-pinch behavior and a different minimum radius. The validation version is another independent interpretation. This makes it difficult to say which Lee/RADPF model is the project source of truth.

Required resolution:

Consolidate the Lee model into one documented reference implementation and require backend parity tests against it. Backend-specific approximations should be explicitly labeled. Device-specific constants such as `0.17*a` should be parameterized or tied to the KnowledgeReference conditions that justify them.

Ratchet update 2026-05-05:

- Module touched: `src/dpf/metal/mlx_snowplow.py`.
- The module no longer claims to implement the full 5-phase Lee/RADPF model. Its docstring now states the implemented scope: axial snowplow, radial inward slug, then a pinch stop.
- The reduced MLX model now enforces the KR deuterium gross compression boundary `r_min = 0.13a` instead of collapsing to `0.01a` near the axis.
- `pinch_column_fraction` now affects the radial focus length through `z_pinch_limit = min(pcf * L_anode, 0.7a)`, using the KR deuterium gross maximum `z_p = 0.7a`.
- The model still returns zero plasma resistance because it does not implement reflected shock, radiative pinch, anomalous resistance, or expanded-column post-focus circuit physics. This is now a declared scope limit rather than an implicit missing phase.
- Added `tests/test_mlx_snowplow.py` to cover the pcf cap, the KR `r_min` stop, and geometry validation. The test loads this scalar file directly because importing the `dpf.metal` package aborts on `mlx.core` in this environment.
- Verification status: `python3 -m pytest tests/test_mlx_snowplow.py -q` passed; `python3 -m py_compile src/dpf/metal/mlx_snowplow.py tests/test_mlx_snowplow.py` passed.

## Finding 3: Post-pinch/current-dip behavior contains internally flagged, non-KR constants

Severity: High

Code evidence:

- `src/dpf/fluid/snowplow.py:570-575` labels the 2x post-pinch dynamic-resistance amplifier as `UNVERIFIED`.
- `src/dpf/fluid/snowplow.py:811-824` labels the `8.0*rho0` post-shock density compromise as `UNVERIFIED`.
- `src/dpf/fluid/snowplow.py:874-879` applies a factor of 3 to post-pinch expansion velocity.

KnowledgeReference basis:

- `KnowledgeReference/lee_radpf_theory.md:5323-5331` warns that anomalous resistance can create an unphysical voltage spike and describes freezing the piston before the expanded-column phase.
- `KnowledgeReference/lee_radpf_theory.md:5337-5344` describes the expanded column as a uniform current-carrying column, not the same as a calibrated disruption-resistance model.
- `KnowledgeReference/neutron-generation-dynamics-inside-a-ma-class-dense-plasma-focus-z-pinch-5.md:323-335` supports an approximate expansion velocity `v_exp ~ v_imp/3` and expansion duration scaling, but not the exact implementation choices above.
- `KnowledgeReference/a-course-on-plasma-focus-numerical-experiments-s-lee-and-s-h-saw-part-1-basic-course.md:15569-15577` says the deuterium gross-parameter model can stop after reflected-shock/piston interaction because the radiative phase is not significant.

Explanation:

The post-pinch current dip and crowbar behavior are important for full-discharge waveforms, but this code currently relies on fitted or internally acknowledged unverified constants. That can be acceptable for a waveform-fitting scaffold, but it is not a KnowledgeReference-backed predictive model of the disruption phase.

Required resolution:

Keep these terms behind an explicit empirical-model flag and add a short derivation file listing every post-pinch constant, its KnowledgeReference support, and its validated device range. If no KR support exists, keep the constant but label outputs as fitted/empirical.

Ratchet update 2026-05-05:

- Module touched: `src/dpf/fluid/snowplow.py`.
- The reflected-shock post-shock density ratio is no longer the unverified `8.0*rho0` compromise. It now uses the KR Rankine-Hugoniot compression ratio `(gamma+1)/(gamma-1) = 4` for the deuterium `gamma=5/3` model.
- The hidden post-pinch expansion factor of 3 was removed from the active default. The expansion velocity now uses an explicit multiplier of `1.0`, matching the local model statement `v_expand = r_pinch/tau_m0`.
- The remaining post-pinch anomalous resistance multiplier is still empirical. It is now named as `_post_pinch_resistance_multiplier` and reported in result dictionaries along with `R_spitzer`, `R_anom`, and `post_pinch_empirical_resistance`.
- Added `tests/test_snowplow_post_pinch_audit.py` to lock the KR reflected-shock density ratio, prevent the hidden factor-3 expansion from returning, and require post-pinch outputs to report empirical resistance components.
- Verification status: `python3 -m pytest tests/test_snowplow_post_pinch_audit.py -q` passed; `python3 -m pytest tests/test_snowplow_consolidated.py::TestReflectedShockPhase::test_pinch_transitions_to_reflected tests/test_snowplow_consolidated.py::TestReflectedInductance::test_dL_dt_negative_during_expansion -q` passed; `python3 -m py_compile src/dpf/fluid/snowplow.py tests/test_snowplow_post_pinch_audit.py` passed.
- Remaining scientific limit: the post-pinch anomalous resistance remains a fitted closure. It is no longer hidden, but it is not yet a KR-derived predictive term.

Ratchet update 2026-05-05, D2 cold-fill pressure profile:

- Module touched: `src/dpf/fluid/snowplow.py` and `tests/test_snowplow_post_pinch_audit.py`.
- The radial-profile export now computes cold unshocked fill pressure with molecular D2 mass (`m_D2`) instead of atomic deuteron mass (`m_d`).
- KnowledgeReference basis: the Lee/RADPF course distinguishes molecular/atomic weight and dissociation number for gases, with dissociation number 2 for deuterium/hydrogen, and separately uses `gamma=5/3` for fully ionized deuterium in the radial phase (`KnowledgeReference/a-course-on-plasma-focus-numerical-experiments-s-lee-and-s-h-saw-part-1-basic-course.md:4488-4490`, `15438`, `16208`).
- Added a regression test so cold fill pressure in exported radial profiles stays tied to molecular D2 before shock dissociation/ionization.
- Verification status: `python3 -m pytest tests/test_snowplow_post_pinch_audit.py -q` passed; `python3 -m py_compile src/dpf/fluid/snowplow.py tests/test_snowplow_post_pinch_audit.py` passed.
- Remaining scientific limit: this fixes a local pressure initialization. It does not resolve the broader ideal-gas/EOS limitation during breakdown and ionization.

Ratchet update 2026-05-05, Lee comparison reflected-shock density:

- Modules touched: `src/dpf/validation/lee_model_comparison.py` and `tests/test_lee_model_comparison_audit.py`.
- The independent Lee-model comparison path no longer uses the unverified `8.0*rho0` reflected-shock density compromise.
- Added `_D2_STRONG_SHOCK_COMPRESSION = 4.0` from `gamma=5/3` Rankine-Hugoniot compression and use it for reflected-shock post-shock density, matching the production `SnowplowModel` ratchet.
- Added an audit test so this comparison implementation cannot silently diverge back to the old factor-8 closure.
- Verification status: `python3 -m pytest tests/test_lee_model_comparison_audit.py tests/test_snowplow_post_pinch_audit.py -q` passed; `python3 -m py_compile src/dpf/validation/lee_model_comparison.py tests/test_lee_model_comparison_audit.py` passed.
- Remaining scientific limit: this removes an unsupported density multiplier from a comparison model. Reflected-shock/post-pinch dynamics still require experimental phase evidence before supporting predictive claims.

## Finding 4: Neutron-yield modules contain good source-backed pieces, but total-yield prediction should be limited

Severity: High

Code evidence:

- `src/dpf/diagnostics/neutron_yield.py:1-19` correctly scopes itself as thermonuclear Bosch-Hale yield.
- `src/dpf/diagnostics/neutron_yield.py:135-173` computes DD thermonuclear yield rate from density, ion temperature, and volume.
- `src/dpf/diagnostics/beam_target.py:1-16` implements the Lee/Saw beam-target formula with calibrated `Cn`.
- `src/dpf/diagnostics/beam_target.py:163-239` implements the Lee/Saw beam-target yield.
- The integration tests mostly check existence/non-negativity: `tests/test_mhd_physics_integration.py:80-102`.

KnowledgeReference basis:

- `KnowledgeReference/bosch-hale-1992-fusion-reactivity.md:24-36` and `KnowledgeReference/bosch-hale-1992-fusion-reactivity.md:81-110` support the Bosch-Hale DD cross-section/reactivity fits and their validity range.
- `KnowledgeReference/lee_radpf_theory.md:4060-4104` supports the Lee/Saw beam-target yield form, `E_beam = 3*Vmax`, and calibration around `Yn = 7e9` at 0.5 MA.
- `KnowledgeReference/original-deuteron-beam-fluence-emitted-from-dense-plasma-focus.md:219-272` supports the Lee beam fluence/voltage relationships, including `U = 3 Vmax`.
- `KnowledgeReference/unlimited-release-printed-september-2009-alegra-hedp-simulations-of-the-dense-plasma-focus.md:318-326` says neutron production is largely non-thermonuclear and ordinary MHD is expected to underpredict experiment.
- `KnowledgeReference/seyler-2021-kr-doped-dpf-mhd.md:59-69` says MHD cannot capture kinetic effects or beam-target neutrons and that thermonuclear yield can be about 1% of total in the discussed regime.
- `KnowledgeReference/the-dense-plasma-focus-a-versatile-dense-pinch-for-diverse-applications.md:2650-2698` discusses non-equilibrium and beam-target dominance at lower current.
- `KnowledgeReference/neutron-generation-dynamics-inside-a-ma-class-dense-plasma-focus-z-pinch-5.md:405-448` says both thermonuclear and beam-target mechanisms can contribute and that later peaks align with disruption/beam-target events.

Explanation:

The thermonuclear and Lee/Saw beam-target formulas are among the strongest parts of the code under the KnowledgeReference-only rule. The problem is not those formulas. The problem is using them as if the rest of the simulation self-consistently supplies the pinch state, beam energy, beam density, disruption timing, and anisotropy. The KnowledgeReference repeatedly says those quantities are tied to kinetic physics and instabilities.

Required resolution:

Report neutron predictions as decomposed model estimates: thermonuclear Bosch-Hale, Lee/Saw beam-target, and any empirical anisotropy/event model separately. Do not present a single total neutron yield as MHD-predicted until the pinch state and beam generation are validated against source-backed diagnostics.

Ratchet update 2026-05-05:

- Modules touched: `src/dpf/diagnostics/yield_tracker.py`, `src/dpf/engine/state_management.py`, and `src/dpf/engine/core.py`.
- `YieldResult` now carries explicit `model_components` and `validity_notes` metadata, separating Bosch-Hale thermonuclear reactivity, Lee/Saw beam-target estimation, and the summed estimate.
- Added `YieldResult.to_summary_dict()` so diagnostics/export code can report `Y_thermonuclear`, `Y_beam_target`, `Y_neutron`, `bt_fraction`, peak yield time, model sources, and validity caveats together.
- Engine diagnostic records now include the source-explicit neutron-yield summary under the `neutrons` diagnostic block.
- `SimulationEngine.run()` now returns `neutron_yield_details` for the decomposed metadata while preserving the existing scalar `total_neutron_yield` API. I intentionally did not overload the existing `neutron_yield` key because validation code treats that key as a scalar total.
- Added regression coverage in `tests/test_yield_tracker.py` for metadata presence, component preservation, and post-accumulation summaries.
- Verification status: `python3 -m pytest tests/test_yield_tracker.py -q` passed; `python3 -m pytest tests/test_neutron_yield.py tests/test_infrastructure_consolidated.py::TestNeutronYieldIntegration::test_neutron_yield_in_summary -q` passed; `python3 -m py_compile src/dpf/diagnostics/yield_tracker.py src/dpf/engine/state_management.py src/dpf/engine/core.py tests/test_yield_tracker.py` passed.
- Remaining scientific limit: this ratchet does not make MHD generate beam-target physics. It makes the model boundary explicit and machine-readable so later validation cannot accidentally treat the summed yield as a first-principles MHD prediction.

## Finding 5: Empirical scaling laws should not be used as predictive validation

Severity: Medium

Code evidence:

- `tests/test_scaling_laws.py:8-48` verifies broad order, monotonicity, a saturation flag, and narrative text, not physical predictive accuracy.
- `README.md:136-148` already notes that grid convergence is misleading because current is circuit/snowplow-driven, and that full MHD-driven current prediction is still in progress.

KnowledgeReference basis:

- `KnowledgeReference/neutron-scaling-laws-from-numerical-experiments.md:16-43` supports Lee-style scaling in terms of `I_pinch` and `I_peak`, but notes that `I_pinch` is often guessed unless the current waveform is fitted.
- `KnowledgeReference/the-dense-plasma-focus-a-versatile-dense-pinch-for-diverse-applications.md:2773-2875` lists the assumptions behind `I^4` scaling and describes PF-1000 deviations that can make the law overpredict heavily.
- `KnowledgeReference/on-the-failure-of-neutron-yield-scaling-in-the-dense-plasma-focus-s-k-h-auluck-international.md:13-42` explains scaling failure in large installations and the role of drive parameter/electrode geometry.
- `KnowledgeReference/scholz-2007-pf1000-part2-jphysd.md:110-119` says published yield scaling can follow `Ec^2`, `I^4`, or even `I^5`, and that `Ip` is the current through the dense pinch, not necessarily total bank current.

Explanation:

Scaling-law diagnostics are useful for sanity checks and narrative context. They are not validation of the MHD solver or neutron predictions. Under the KnowledgeReference rule, a scaling-law output should be treated as a regime warning and rough estimate, not a substitute for validating pinch current, voltage, beam energy, density, and neutron timing.

Required resolution:

Keep scaling outputs in a "diagnostic/estimate" namespace and explicitly separate them from solver validation metrics.

Ratchet update 2026-05-05:

- Modules touched: `src/dpf/diagnostics/scaling_laws.py` and `app_mhd.py`.
- `ScalingResult` now carries `model_role = diagnostic_estimate` and `validation_role = not_solver_validation`.
- Added source/validity metadata for Lee/Saw `I^4`, cross-device scaling, energy scaling, and Bennett temperature so downstream consumers can distinguish empirical estimates from solver validation.
- Added `ScalingResult.to_summary_dict()` and changed `app_mhd.py` to export that structured summary under `result["scaling_laws"]`.
- `scaling_narrative()` now labels the output as a diagnostic estimate and explicitly says it is not solver validation.
- Added regression coverage in `tests/test_scaling_laws.py` for the metadata and summary dictionary.
- Verification status: `python3 -m pytest tests/test_scaling_laws.py tests/test_mhd_physics_integration.py::test_mhd_bennett_diagnostic_keys_present -q` passed; `python3 -m py_compile src/dpf/diagnostics/scaling_laws.py app_mhd.py tests/test_scaling_laws.py` passed.
- Remaining scientific limit: the numeric scaling formulas themselves remain empirical estimates. This ratchet prevents their accidental promotion into validation evidence; it does not make them predictive.

## Finding 6: Radiation and high-Z dopant physics are not predictive under the KR-only standard

Severity: High

Code evidence:

- `src/dpf/radiation/bremsstrahlung.py:1-45` is well sourced to the NRL formulary.
- `src/dpf/radiation/line_radiation.py:27-39` states that the implemented line-radiation piecewise power-law fits have unknown provenance.
- `src/dpf/radiation/line_radiation.py:68-127` marks hydrogen and neon cooling functions as empirical/unverified in form or coefficient provenance.
- `src/dpf/radiation/line_radiation.py:403-429` exposes those fits as the public coronal cooling function.
- `src/dpf/diagnostics/pb11_yield.py:1-26` cites Nevins/Swain, Rider, Becker, and Nevins, but those source tables are not in the KnowledgeReference corpus reviewed here.
- `src/dpf/diagnostics/pb11_yield.py:36-80` uses tabulated p-B11 values and a Gamow coefficient marked empirical.

KnowledgeReference basis:

- `KnowledgeReference/plasma-formulary.md:4887-4934` and `KnowledgeReference/plasma-formulary.md:5064-5109` support basic optically thin radiation and bremsstrahlung forms.
- `KnowledgeReference/seyler-2021-kr-doped-dpf-mhd.md:184-190` says Kr-doped DPF modeling used LEOS D/Kr, mixed EOS, opacities, and multigroup radiation diffusion.
- `KnowledgeReference/seyler-2021-kr-doped-dpf-mhd.md:488-517` says Kr changes radiative cooling and sheath structure, while experiments show effects not captured by 2D MHD.
- `KnowledgeReference/unlimited-release-printed-september-2009-alegra-hedp-simulations-of-the-dense-plasma-focus.md:332-362` emphasizes EOS, advanced conductivities, radiation, and two-temperature modeling for HEDP DPF simulation.
- `README.md:86` says tabulated EOS and radiation transport are not implemented.
- `KnowledgeReference/focus-fusion-overview-of-progress-towards-p-b11-fusion-with-the-dense-plasma-focus.md:85-94` says p-B11 DPF faces high ion-energy and high-Z X-ray cooling challenges.
- `KnowledgeReference/plasma-formulary.md:4266-4271` supports the p+B11 reaction and energy release but not the Nevins/Swain table values used in code.

Explanation:

Bremsstrahlung is on solid footing. The line-radiation and p-B11 numerical yield pieces are not source-backed enough for predictive claims under the user's rules. The KnowledgeReference standard for high-Z dopants is materially more complex than the current code: EOS, charge-state modeling, opacities, and radiation transport matter.

Required resolution:

Mark line radiation, dopant performance, and p-B11 yield outputs as empirical/qualitative until the exact source tables and derivations are added to `KnowledgeReference` and regression tests compare against them.

Ratchet update 2026-05-05:

- Modules touched: `src/dpf/radiation/line_radiation.py` and `src/dpf/radiation/__init__.py`.
- The line-radiation module docstring no longer presents the piecewise fits as verified CHIANTI/ADAS-derived fits. It now calls them empirical reduced coronal-equilibrium fits.
- Added `line_radiation_model_metadata()` and exported it through `dpf.radiation`.
- The metadata separates the NRL-backed bremsstrahlung component from empirical line-radiation fits and hydrogenic recombination approximation.
- The metadata explicitly marks high-Z/dopant cooling as `empirical_cooling_estimate`, `not_high_z_predictive`, and not a substitute for multigroup radiation diffusion with tabulated EOS/opacities/charge-state kinetics.
- Added `tests/test_radiation_model_metadata.py` to lock this provenance boundary.
- Verification status: `python3 -m pytest tests/test_radiation_model_metadata.py tests/test_physics.py::TestCoolingFunction tests/test_physics.py::TestLinePower tests/test_physics.py::TestImplicitLineCooling tests/test_physics.py::TestCuLineRadiationDominance -q` passed; `python3 -m py_compile src/dpf/radiation/line_radiation.py src/dpf/radiation/__init__.py tests/test_radiation_model_metadata.py` passed.
- Remaining scientific limit: this ratchet does not add tabulated EOS, opacities, charge-state kinetics, or multigroup transport. It only prevents the existing reduced cooling fits from being represented as predictive high-Z radiation physics.

Ratchet update 2026-05-05, p-B11 diagnostics:

- Modules touched: `src/dpf/diagnostics/pb11_yield.py` and `tests/test_pb11_yield.py`.
- Added a KnowledgeReference note to the module docstring: the local corpus supports the p+B11 reaction/Q-value, but the Nevins/Swain/Rider/Becker reactivity sources used by the table are not currently verified in `KnowledgeReference`.
- Added `pb11_model_metadata()` with `model_role = reactivity_table_estimate`, `validation_role = not_dpf_feasibility_validation`, and `predictive_dpf_pb11 = False`.
- The metadata separates reaction/Q support from reactivity-table provenance, thermonuclear volume integration, non-Maxwellian beam absence, alpha transport absence, fuel-mixing absence, radiation-loss absence, and the source-table gap.
- Added regression coverage in `tests/test_pb11_yield.py` so p-B11 outputs cannot be silently promoted to source-backed DPF feasibility claims.
- Verification status: `python3 -m pytest tests/test_pb11_yield.py -q` passed; `python3 -m py_compile src/dpf/diagnostics/pb11_yield.py tests/test_pb11_yield.py` passed.
- Remaining scientific limit: this does not validate p-B11 DPF operation. It preserves the calculator as a marked estimate until the cited reactivity sources and DPF feasibility constraints are added to the local corpus and tested.

## Finding 7: Validation is strongest at circuit waveform level, weak at spatial MHD/pinch physics level

Severity: High

Code evidence:

- `README.md:90-98` explicitly says circuit-level Lee snowplow validation exists, while MHD-level validation against spatially resolved experimental data does not.
- `README.md:124-127` reports a validation regression and red CI status.
- `README.md:136-148` says grid convergence is misleading because the circuit/snowplow ODE drives current, and full MHD-driven current prediction remains in progress.
- `tests/test_validation_ci.py:109-145` has useful PF-1000 circuit/waveform tests.
- `tests/test_validation_ci.py:147-181` notes that the fuller Gribkov waveform exposes a post-peak gap hidden by shorter Scholz data.
- `tests/test_validation_ci.py:192-232` includes multi-device sanity tests, with expected xfail for difficult/reconstructed waveform cases.
- `tests/test_mhd_physics_integration.py:222-239` only verifies that inductance/current arrays exist and are non-negative/evolving, not that MHD feedback is physically validated.
- `tests/test_mhd_physics_integration.py:245-257` skips explicit m=0 perturbation seeding for Python/Metal backends.

KnowledgeReference basis:

- `KnowledgeReference/unlimited-release-printed-september-2009-alegra-hedp-simulations-of-the-dense-plasma-focus.md:370-376` notes 2D cathode-bar limitations and the need for full 3D for some effects.
- `KnowledgeReference/unlimited-release-printed-september-2009-alegra-hedp-simulations-of-the-dense-plasma-focus.md:550-558` says MHD is useful but limited and neutron yields are underestimated.
- `KnowledgeReference/seyler-2021-kr-doped-dpf-mhd.md:406-448` says 2D MHD likely overestimates pinch tightness/radiated power because it lacks 3D/kinetic effects.
- `KnowledgeReference/neutron-generation-dynamics-inside-a-ma-class-dense-plasma-focus-z-pinch-5.md:713-784` shows reduced MHD-kinetic modeling can reproduce trends but not absolute values without discrepancies.

Explanation:

The validation suite is valuable, but it validates the circuit/snowplow workflow much more than the MHD physics. The existing tests are not sufficient evidence that the code predicts sheath morphology, pinch density/temperature, instability timing, beam formation, or neutron timing from first principles.

Required resolution:

Add validation tiers:

1. Circuit/Lee waveform validation.
2. Snowplow phase/timing validation.
3. Spatial MHD verification against analytic tests.
4. Spatial DPF validation against density/B-field/temperature diagnostics.
5. Neutron/yield validation with mechanism decomposition and timing.

Only tier 1 is currently strong.

Ratchet update 2026-05-05:

- Modules touched: `src/dpf/validation/quality_assessment.py` and `src/dpf/validation/__init__.py`.
- Added a `ValidationTier` dataclass and `validation_tier_report(result)` helper.
- `QualityAssessment` now carries a fixed five-tier validation report: circuit/Lee waveform, snowplow phase/timing, spatial MHD verification, spatial DPF experimental validation, and neutron mechanism/timing.
- `assess_quality()` still returns its existing grade and checks, but its summary now includes tier statuses so a good circuit-grade result cannot be mistaken for spatial DPF validation.
- The validation package now exports `QualityAssessment`, `QualityCheck`, `ValidationTier`, `assess_quality`, and `validation_tier_report`.
- Added regression coverage in `tests/test_quality_assessment.py` so MHD results without spatial experimental evidence are marked `verification_only`/`not_validated`, and neutron-yield details are marked as estimates until mechanism timing is validated.
- Verification status: `python3 -m pytest tests/test_quality_assessment.py -q` passed; `python3 -m py_compile src/dpf/validation/quality_assessment.py src/dpf/validation/__init__.py tests/test_quality_assessment.py` passed.
- Remaining scientific limit: this ratchet does not add spatially resolved DPF experimental validation. It prevents validation reports from overstating the current evidence tier.

Ratchet update 2026-05-05, predictive-readiness gate:

- Modules touched: `src/dpf/validation/quality_assessment.py`, `src/dpf/validation/__init__.py`, and `tests/test_quality_assessment.py`.
- Added `PredictiveReadiness` and `predictive_readiness_report(result)`.
- The gate requires five supported evidence tiers before a result can be marked `predictive_ready`: circuit/Lee waveform validation, snowplow phase/timing validation, spatial MHD code verification, spatial DPF experimental validation, and neutron mechanism/timing/spectrum/anisotropy validation.
- `assess_quality()` now carries `predictive_readiness` in `QualityAssessment` and includes a readiness line in its summary. This means a result can still get a good quality grade while being explicitly blocked from predictive end-to-end claims.
- `validation_tier_report()` now distinguishes mere diagnostics from attached validation evidence: snowplow diagnostics are `partial` until `snowplow_validation` exists, MHD backend presence is `verification_only` until `mhd_verification` exists, and neutron-yield estimates are `decomposed_estimate` until mechanism timing, spectrum, and anisotropy evidence all exist.
- Verification status: `python3 -m pytest tests/test_quality_assessment.py -q` passed; `python3 -m py_compile src/dpf/validation/quality_assessment.py src/dpf/validation/__init__.py tests/test_quality_assessment.py` passed.
- Remaining scientific limit: the gate can certify evidence when it is attached, but it does not create the missing spatial and neutron timing validation data. It turns that missing evidence into an enforceable blocker instead of a narrative caveat.

Ratchet update 2026-05-05, app-level readiness export:

- Modules touched: `app_mhd.py` and `tests/test_mhd_physics_integration.py`.
- `run_mhd_simulation` post-processing now exports `validation_tiers` and `predictive_readiness` as JSON-friendly dictionaries on every result where the validation module is importable.
- Added an integration test asserting that ordinary D2 simulation output exposes the readiness gate and remains `not_predictive_ready` when spatial DPF experimental validation is absent.
- Verification status: `python3 -m pytest tests/test_mhd_physics_integration.py::test_predictive_readiness_exported_and_blocks_unvalidated_claims tests/test_quality_assessment.py -q` passed; `python3 -m py_compile app_mhd.py tests/test_mhd_physics_integration.py src/dpf/validation/quality_assessment.py` passed.
- Remaining scientific limit: the app can now report predictive-readiness blockers automatically, but the spatial validation and neutron timing evidence still must be generated or ingested before readiness can pass.

Ratchet update 2026-05-05, strict evidence schema:

- Modules touched: `src/dpf/validation/quality_assessment.py` and `tests/test_quality_assessment.py`.
- Tightened the readiness gate so truthy placeholder dictionaries are no longer sufficient to mark a tier `supported`.
- Circuit/Lee waveform validation now requires explicit `passed: True` plus peak-current, peak-time, and waveform-shape coverage. Runtime current scalars alone are only `diagnostic_present`.
- Snowplow validation now requires explicit `passed: True` plus axial, radial, and pinch phase coverage.
- Spatial DPF validation now requires explicit `passed: True` plus density, magnetic-field, and temperature coverage.
- Neutron mechanism/timing validation now requires explicit `passed: True` plus thermonuclear and beam-target mechanism coverage.
- Added tests proving a complete evidence package passes and placeholder evidence remains blocked.
- Verification status: `python3 -m pytest tests/test_quality_assessment.py tests/test_mhd_physics_integration.py::test_predictive_readiness_exported_and_blocks_unvalidated_claims -q` passed; `python3 -m py_compile src/dpf/validation/quality_assessment.py tests/test_quality_assessment.py` passed.
- Remaining scientific limit: these are still schema gates. They define what evidence must look like; they do not yet generate the missing experimental validation evidence.

Ratchet update 2026-05-05, circuit evidence producer:

- Modules touched: `src/dpf/validation/quality_assessment.py`, `src/dpf/validation/__init__.py`, and `tests/test_quality_assessment.py`.
- Added `circuit_validation_evidence_from_result()` to convert existing `ValidationSuite` circuit metrics plus waveform NRMSE evidence into the strict `circuit_validation` shape used by the predictive-readiness gate.
- The helper canonicalizes `peak_current`, `peak_current_time`/`peak_time`, and waveform NRMSE/shape evidence into the required peak-current, peak-time, and waveform-shape metrics.
- Added tests proving peak-current and timing evidence alone are insufficient, and that adding passing waveform-shape evidence can support tier 1.
- Verification status: `python3 -m pytest tests/test_quality_assessment.py -q` passed; `python3 -m py_compile src/dpf/validation/quality_assessment.py src/dpf/validation/__init__.py tests/test_quality_assessment.py` passed.
- Remaining scientific limit: this produces tier-1 evidence from existing circuit validation outputs only. It does not address snowplow phase validation, spatial MHD validation, or neutron timing validation.

Ratchet update 2026-05-05, MHD verification evidence producer:

- Modules touched: `src/dpf/validation/quality_assessment.py`, `src/dpf/validation/__init__.py`, and `tests/test_quality_assessment.py`.
- Tightened tier-3 support so `mhd_verification: {"passed": true}` is not enough. The evidence must include named analytic tests.
- Added `mhd_verification_evidence_from_tests()` to build strict MHD verification evidence from named test results.
- Current required analytic coverage is Sod and Brio-Wu, matching the project’s existing MHD verification claims. Test-name normalization handles labels such as `Brio-Wu` and `brio_wu`.
- Added tests proving Sod alone is insufficient, Sod plus Brio-Wu supports tier 3, and placeholder MHD verification remains `verification_only`.
- Verification status: `python3 -m pytest tests/test_quality_assessment.py tests/test_mhd_physics_integration.py::test_predictive_readiness_exported_and_blocks_unvalidated_claims -q` passed; `python3 -m py_compile src/dpf/validation/quality_assessment.py src/dpf/validation/__init__.py tests/test_quality_assessment.py` passed.
- Remaining scientific limit: tier 3 is still code verification, not DPF validation. It supports the numerical MHD solver only; tier 4 spatial DPF experimental validation remains required for predictive claims.

Ratchet update 2026-05-05, snowplow phase evidence producer:

- Modules touched: `src/dpf/validation/quality_assessment.py`, `src/dpf/validation/__init__.py`, and `tests/test_quality_assessment.py`.
- Added `snowplow_validation_evidence_from_phase_errors()` to convert phase timing relative errors into strict tier-2 evidence.
- The helper requires axial, radial, and pinch phase timing errors to be present and within tolerance before the evidence passes.
- Added tests proving missing pinch-phase evidence blocks tier 2, and complete phase coverage supports tier 2.
- Verification status: `python3 -m pytest tests/test_quality_assessment.py -q` passed; `python3 -m py_compile src/dpf/validation/quality_assessment.py src/dpf/validation/__init__.py tests/test_quality_assessment.py` passed.
- Remaining scientific limit: this helper converts phase validation metrics into evidence. It does not create the underlying experimental phase/timing comparisons by itself.

Ratchet update 2026-05-05, spatial and neutron evidence producers:

- Modules touched: `src/dpf/validation/quality_assessment.py`, `src/dpf/validation/__init__.py`, and `tests/test_quality_assessment.py`.
- Added `spatial_validation_evidence_from_quantity_errors()` to convert spatial diagnostic relative errors into strict tier-4 evidence. It requires density, magnetic-field, and temperature diagnostics to pass.
- Added `neutron_timing_validation_evidence_from_errors()` to convert mechanism timing errors into strict tier-5 evidence. It requires thermonuclear and beam-target timing coverage to pass.
- Added evidence-label normalization for common aliases such as `rho`, `ne`, `B_field`, `Te`, and `beam-target`.
- Added tests proving incomplete spatial/neutron evidence remains blocked and complete evidence supports the relevant tiers.
- Verification status: `python3 -m pytest tests/test_quality_assessment.py -q` passed; `python3 -m py_compile src/dpf/validation/quality_assessment.py src/dpf/validation/__init__.py tests/test_quality_assessment.py` passed.
- Remaining scientific limit: these helpers define and validate evidence shape. The actual spatial diagnostic datasets and neutron timing comparisons still need to be generated or ingested from KnowledgeReference-backed data.

Ratchet update 2026-05-05, KR neutron timing target extraction:

- Modules touched: `src/dpf/validation/kr_targets.py`, `src/dpf/validation/__init__.py`, and `tests/test_kr_targets.py`.
- Added `mjolnir_neutron_timing_targets()` as a KnowledgeReference-backed tier-5 validation target for the MJOLNIR neutron generation dynamics paper.
- The target records the KR source path and line ranges, the 60 kV / 735 kJ / 2.8 MA / 2.1 MA shot context, thermonuclear emission at stagnation, beam-target timing targets about 5 ns and 10 ns after stagnation, 96 ns detector time-of-flight for 2.45 MeV neutrons, spectral broadening up to about 5 MeV, and anisotropy trends.
- Tests assert these extracted values and also assert that the KR target is not predictive-readiness evidence by itself because no simulation comparison has passed yet.
- Verification status: `python3 -m pytest tests/test_kr_targets.py -q` passed; `python3 -m py_compile src/dpf/validation/kr_targets.py src/dpf/validation/__init__.py tests/test_kr_targets.py` passed.
- Remaining scientific limit: the project now has a local source-backed neutron timing target, but still lacks a simulation-to-target comparison producing tier-5 evidence.

Ratchet update 2026-05-05, MJOLNIR neutron timing comparison:

- Modules touched: `src/dpf/validation/kr_targets.py`, `src/dpf/validation/__init__.py`, and `tests/test_kr_targets.py`.
- Added `mjolnir_neutron_timing_evidence_from_history()` to compare mechanism-separated simulated neutron histories against the KR-backed MJOLNIR timing target.
- The helper accepts a `YieldResult` or exported `yield_time_resolved` dictionary, checks thermonuclear timing relative to stagnation, checks beam-target timing against the about-5 ns target, and can optionally require the about-10 ns measurement-correlation target.
- The output uses the strict tier-5 evidence shape (`passed`, `mechanisms`, source, target, details), so predictive readiness can be supported only by a simulation-to-target comparison rather than by the target data alone.
- Tests cover a KR-like passing synthetic history, a missing beam-target failure, and the weaker path where stagnation is inferred from the thermonuclear peak and explicitly labeled.
- Verification status: `python3 -m pytest tests/test_kr_targets.py tests/test_quality_assessment.py::TestQualityAssessment::test_neutron_timing_evidence_requires_both_mechanisms -q` passed; `python3 -m py_compile src/dpf/validation/kr_targets.py src/dpf/validation/__init__.py tests/test_kr_targets.py` passed.
- Remaining scientific limit: this is now a source-backed comparison path, but production simulation runs still must expose mechanism-separated neutron histories with independent stagnation timing before tier-5 validation should be claimed.

Ratchet update 2026-05-05, app-level MJOLNIR timing evidence export:

- Modules touched: `app_mhd.py` and `tests/test_mhd_physics_integration.py`.
- Added `_phase_stagnation_time_s()` to extract the first phase-labeled pinch/reflected/post-pinch time from app results.
- `run_mhd_simulation` post-processing now runs the MJOLNIR neutron timing comparison for D2 MJOLNIR results with time-resolved neutron histories.
- If stagnation comes from phase timing, the comparison is exported as `neutron_mechanism_timing_validation` and can support tier 5 when it passes. If stagnation is inferred from the thermonuclear neutron peak, the comparison is exported only as `neutron_timing_validation_candidate` and does not feed the predictive-readiness gate.
- Tests prove both behaviors using synthetic MJOLNIR histories.
- Verification status: `python3 -m pytest tests/test_mhd_physics_integration.py::test_mjolnir_neutron_timing_evidence_is_exported_when_phase_timed tests/test_mhd_physics_integration.py::test_mjolnir_inferred_neutron_timing_remains_candidate_only tests/test_kr_targets.py -q` passed; `python3 -m py_compile app_mhd.py tests/test_mhd_physics_integration.py src/dpf/validation/kr_targets.py` passed.
- Remaining scientific limit: tier-5 export is now wired for MJOLNIR-like histories, but end-to-end predictive readiness is still blocked by tier-4 spatial DPF validation and by the need for real production MJOLNIR runs to generate the required phase-timed neutron history.

Ratchet update 2026-05-05, PF-1000 density-proxy spatial target:

- Modules touched: `src/dpf/diagnostics/xray_imaging.py`, `src/dpf/validation/kr_targets.py`, `src/dpf/validation/__init__.py`, `tests/test_synthetic_diagnostics.py`, and `tests/test_kr_targets.py`.
- Added `radiating_pinch_geometry_from_image()` to extract diameter, axial length, and axial position from a synthetic gated X-ray/optical emission image.
- Added `pf1000_spatial_pinch_targets()` from the PF-1000 KnowledgeReference report: 4 hPa, 734 kJ, 1.66 MA shot context; 589 nm filtered camera density-proxy basis; about 5 mm minimum radiating diameter; about 5 cm radiating length; about 1 cm dense spherical structure; 30-50 ns dense-structure lifetime.
- Added `pf1000_spatial_pinch_evidence_from_geometry()` to compare synthetic radiating-pinch geometry with the KR target.
- Tests prove a matching synthetic image yields density-proxy evidence, but that evidence alone still leaves tier 4 `not_validated` because magnetic-field and temperature diagnostics remain missing.
- Verification status: `python3 -m pytest tests/test_synthetic_diagnostics.py::TestXrayImaging tests/test_kr_targets.py -q` passed; `python3 -m py_compile src/dpf/diagnostics/xray_imaging.py src/dpf/validation/kr_targets.py src/dpf/validation/__init__.py tests/test_synthetic_diagnostics.py tests/test_kr_targets.py` passed.
- Remaining scientific limit: this closes only the density-proxy geometry part of spatial validation. It does not provide magnetic-field or temperature validation, and it does not make the code predictive-ready.

Ratchet update 2026-05-05, LLNL 1.2 kJ EM fluctuation target:

- Modules touched: `src/dpf/validation/kr_targets.py`, `src/dpf/validation/__init__.py`, and `tests/test_kr_targets.py`.
- Added `llnl_12kj_em_fluctuation_targets()` from the LLNL kinetic-simulation comparison paper: EM pick-up probe bandwidth to 5 GHz, high-quality pinch activity in the 3-4 GHz band, simulated Ez probe strongest in the same band, and simulated 10-40 T pinch fields corresponding to lower-hybrid frequencies of about 4.6-18 GHz.
- Added `llnl_12kj_em_fluctuation_evidence_from_signal()` to compare a simulated EM probe waveform against the 3-4 GHz target band using FFT power.
- The evidence marks only the `magnetic_field`/EM-fluctuation portion of tier 4 and explicitly lists density and temperature as missing for full tier-4 readiness.
- Tests prove a 3.5 GHz synthetic signal passes the EM target, a 1 GHz signal fails, and EM evidence alone still leaves tier 4 `not_validated`.
- Verification status: `python3 -m pytest tests/test_kr_targets.py -q` passed; `python3 -m py_compile src/dpf/validation/kr_targets.py src/dpf/validation/__init__.py tests/test_kr_targets.py` passed.
- Remaining scientific limit: this adds a source-backed magnetic/EM validation path, but it is not a calibrated magnetic-field map and does not supply temperature validation.

Ratchet update 2026-05-05, DPF temperature-regime target:

- Modules touched: `src/dpf/validation/kr_targets.py`, `src/dpf/validation/__init__.py`, and `tests/test_kr_targets.py`.
- Added `dpf_pinch_temperature_targets()` from the DPF review source: 1-2 mm pinch diameter, density and temperature above `1e19 cm^-3` and 1 keV, reflected-shock/magnetic-compression temperature around 2 keV, thermal X-ray temperature range from 0.4 keV to greater than 4 keV, ion temperature around 1 keV, and compressed magnetic field greater than 100 T.
- Added `dpf_pinch_temperature_evidence()` to compare simulated ion/electron/X-ray temperature summaries against the KR temperature range.
- The evidence marks only the temperature-regime portion of tier 4 and explicitly lists density and magnetic-field diagnostics as missing for full tier-4 readiness.
- Tests prove in-range temperature evidence passes, out-of-range temperature evidence fails, and temperature evidence alone still leaves tier 4 `not_validated`.
- Verification status: `python3 -m pytest tests/test_kr_targets.py -q` passed; `python3 -m py_compile src/dpf/validation/kr_targets.py src/dpf/validation/__init__.py tests/test_kr_targets.py` passed.
- Remaining scientific limit: this is a broad DPF regime check, not device-specific calibrated temperature validation. It should not be combined with unrelated device evidence to claim end-to-end predictive validation.

Ratchet update 2026-05-05, scope-aware spatial evidence combiner:

- Modules touched: `src/dpf/validation/quality_assessment.py`, `src/dpf/validation/kr_targets.py`, `src/dpf/validation/__init__.py`, and `tests/test_quality_assessment.py`.
- Added `combine_spatial_validation_evidence()` to combine density, magnetic-field/EM, and temperature evidence only when all components share one validation scope.
- Partial KR evidence now carries `validation_scope` so PF-1000 density geometry, LLNL EM fluctuations, and generic DPF temperature-regime checks cannot be accidentally merged into a single full tier-4 pass.
- Tests prove cross-scope evidence remains `not_validated` even when all three quantities are individually present, and same-scope evidence can support tier 4.
- Verification status: `python3 -m pytest tests/test_quality_assessment.py tests/test_kr_targets.py -q` passed; `python3 -m py_compile src/dpf/validation/quality_assessment.py src/dpf/validation/__init__.py src/dpf/validation/kr_targets.py tests/test_quality_assessment.py tests/test_kr_targets.py` passed.
- Remaining scientific limit: the combiner creates the safe path for full tier-4 evidence, but real same-shot/device density, magnetic-field, and temperature validation data still need to be produced or ingested.

Ratchet update 2026-05-05, app-level spatial validation components:

- Modules touched: `app_mhd.py` and `tests/test_mhd_physics_integration.py`.
- `run_mhd_simulation` post-processing now emits `spatial_validation_components` when a result has enough temperature history to compare against the KR DPF temperature-regime target.
- App post-processing also combines any existing `spatial_validation_components` with the scope-aware combiner. It promotes the result to `spatial_validation` only when the combined evidence passes; otherwise it stores `spatial_validation_candidate`.
- Tests prove a temperature-only component remains a candidate and leaves tier 4 blocked, while complete same-scope density, magnetic-field, and temperature components are promoted and support tier 4.
- Verification status: `python3 -m pytest tests/test_mhd_physics_integration.py::test_app_exports_temperature_spatial_component_without_tier4_promotion tests/test_mhd_physics_integration.py::test_app_promotes_complete_same_scope_spatial_components tests/test_quality_assessment.py::TestQualityAssessment::test_combined_spatial_evidence_requires_consistent_scope tests/test_quality_assessment.py::TestQualityAssessment::test_combined_spatial_evidence_supports_tier_four_when_scope_matches -q` passed; `python3 -m py_compile app_mhd.py tests/test_mhd_physics_integration.py src/dpf/validation/quality_assessment.py` passed.
- Remaining scientific limit: the app can now carry spatial evidence safely, but ordinary runs still lack same-scope experimental density, magnetic-field, and temperature comparisons.

Ratchet update 2026-05-05, circuit waveform evidence producer:

- Modules touched: `src/dpf/validation/quality_assessment.py`, `src/dpf/validation/__init__.py`, and `tests/test_quality_assessment.py`.
- Added `circuit_validation_evidence_from_waveform()` to compare a simulated current trace directly against a registered experimental waveform and emit strict tier-1 evidence.
- The helper requires peak current, peak time, and waveform-shape NRMSE to pass before tier 1 can be supported.
- Tests prove the PF-1000 registered waveform supports tier 1 and a 50% amplitude-distorted trace is rejected.
- Verification status: `python3 -m pytest tests/test_quality_assessment.py::TestQualityAssessment::test_circuit_waveform_evidence_can_support_tier_one tests/test_quality_assessment.py::TestQualityAssessment::test_circuit_waveform_evidence_rejects_distorted_trace -q` passed; `python3 -m py_compile src/dpf/validation/quality_assessment.py src/dpf/validation/__init__.py tests/test_quality_assessment.py` passed.
- Remaining scientific limit: this validates only the circuit waveform tier. It does not validate snowplow phase dynamics, spatial MHD, or neutron mechanisms.

Ratchet update 2026-05-05, app-level circuit waveform evidence export:

- Modules touched: `app_mhd.py` and `tests/test_mhd_physics_integration.py`.
- App post-processing now auto-attaches `circuit_validation` when a result uses a registered experimental device and provides `t_us` plus `I_MA` waveform histories.
- The exported evidence flows through the existing validation-tier report, so a matching PF-1000 waveform can support tier 1 without requiring a caller to invoke validation helpers manually.
- Tests prove a PF-1000 registered waveform exported through `_apply_post_processing()` supports tier 1.
- Verification status: `python3 -m pytest tests/test_mhd_physics_integration.py::test_app_exports_circuit_waveform_validation_for_registered_device tests/test_quality_assessment.py::TestQualityAssessment::test_circuit_waveform_evidence_can_support_tier_one tests/test_quality_assessment.py::TestQualityAssessment::test_circuit_waveform_evidence_rejects_distorted_trace -q` passed; `python3 -m py_compile app_mhd.py tests/test_mhd_physics_integration.py src/dpf/validation/quality_assessment.py` passed.
- Remaining scientific limit: this makes circuit validation easier to carry through the app result, but it is still only a waveform-level electrical validation. It does not validate snowplow phase timing, spatial plasma state, kinetic beam-target physics, or neutron production.

Ratchet update 2026-05-05, snowplow phase-history validation path:

- Modules touched: `src/dpf/validation/quality_assessment.py`, `src/dpf/validation/__init__.py`, `app_mhd.py`, `tests/test_quality_assessment.py`, and `tests/test_mhd_physics_integration.py`.
- Added `snowplow_phase_observation_from_history()` to summarize observed rundown/radial/pinch phase coverage without promoting phase labels to validation.
- Added `snowplow_validation_evidence_from_phase_history()` to compare phase-label histories against explicit reference timing targets: axial rundown end / radial start, radial transit duration, and absolute pinch/stagnation time.
- App post-processing now emits `snowplow_validation_candidate` for targetless phase histories and promotes to `snowplow_validation` only when caller-supplied reference phase targets are present and all three timing checks pass.
- KnowledgeReference basis: the Lee model course defines axial and radial phases and states that radial-phase fitting is done against current rollover/dip through the end of the radial phase (`KnowledgeReference/a-course-on-plasma-focus-numerical-experiments-s-lee-and-s-h-saw-part-1-basic-course.md:886-891`, `14922-14936`). The same source gives example phase timing semantics where axial end, radial transit, and pinch/gross compression times are separate quantities (`16239-16244`, `16298-16304`). PF-1000 SXR/neutron timing work also uses derivative dip timing as a reference event (`KnowledgeReference/radiation-physics-and-chemistry-188-2021-109633.md:132-137`).
- Tests prove target-backed phase histories can support tier 2, missing pinch observation fails tier 2, and app-level targetless phase labels remain candidate evidence only.
- Verification status: `python3 -m pytest tests/test_quality_assessment.py::TestQualityAssessment::test_snowplow_phase_history_can_support_tier_two_with_targets tests/test_quality_assessment.py::TestQualityAssessment::test_snowplow_phase_history_requires_pinch_observation tests/test_quality_assessment.py::TestQualityAssessment::test_snowplow_phase_observation_is_not_validation tests/test_mhd_physics_integration.py::test_app_keeps_targetless_snowplow_phase_history_as_candidate tests/test_mhd_physics_integration.py::test_app_exports_target_backed_snowplow_validation -q` passed; `python3 -m py_compile app_mhd.py src/dpf/validation/quality_assessment.py src/dpf/validation/__init__.py tests/test_quality_assessment.py tests/test_mhd_physics_integration.py` passed.
- Remaining scientific limit: the code now has a strict tier-2 validation path, but the project still needs real device/shot-specific phase timing targets attached to production runs before ordinary simulations can claim snowplow validation.

Ratchet update 2026-05-05, MHD shock-tube verification evidence adapter:

- Modules touched: `src/dpf/validation/quality_assessment.py`, `src/dpf/validation/__init__.py`, and `tests/test_quality_assessment.py`.
- Added `mhd_verification_evidence_from_shock_tube_results()` to convert existing Sod and Brio-Wu shock-tube verification outputs into strict tier-3 evidence.
- Sod evidence requires density, velocity, and pressure L1 errors under tolerances plus finite/positive sanity checks. Brio-Wu evidence requires finite state, positive density/pressure, normal-field preservation, wave structure, and tangential-field sign change.
- KnowledgeReference basis: local numerical-method references describe one-dimensional Riemann problems as standard verification tests and list Brio-Wu shock-tube tests for ideal MHD solvers (`KnowledgeReference/a-structure-preserving-semi-implicit-imex-finite-volume-scheme-for-ideal-magnetohydrodynamics-at.md:2065-2081`, `KnowledgeReference/asymptotic-preserving-semi-implicit-finite-volume-scheme-for-extended-magnetohydrodynamics-yi-han.md:1395-1457`). The FLASH validation paper also identifies Brio-Wu, Orszag-Tang, and rotor problems as MHD benchmarks (`KnowledgeReference/validation-of-flash-for-magnetically-driven-inertial-confinement-fusion-target-design.md:93-95`).
- Tests prove passing shock-tube outputs support tier 3 and a failed Brio-Wu sign-change check blocks tier 3.
- Verification status: `python3 -m pytest tests/test_quality_assessment.py::TestQualityAssessment::test_mhd_shock_tube_results_can_support_tier_three tests/test_quality_assessment.py::TestQualityAssessment::test_mhd_shock_tube_results_reject_failed_brio_wu_check tests/test_quality_assessment.py::TestQualityAssessment::test_mhd_verification_evidence_requires_named_analytic_tests -q` passed; `python3 -m py_compile src/dpf/validation/quality_assessment.py src/dpf/validation/__init__.py tests/test_quality_assessment.py` passed.
- Remaining scientific limit: tier 3 is code verification. Even fully passing shock-tube verification cannot substitute for tier-4 DPF spatial experimental validation.

Ratchet update 2026-05-05, app-level PF-1000 X-ray geometry spatial component:

- Modules touched: `app_mhd.py` and `tests/test_mhd_physics_integration.py`.
- App post-processing now detects `xray_image` or `synthetic_xray_image` plus `xray_y_cell_m` and `xray_z_cell_m` for PF-1000 results.
- It extracts `pf1000_radiating_pinch_geometry` with `radiating_pinch_geometry_from_image()` and appends PF-1000 density-proxy evidence using `pf1000_spatial_pinch_evidence_from_geometry()`.
- The evidence remains a `spatial_validation_candidate` unless same-scope magnetic-field and temperature components are also present.
- KnowledgeReference basis: PF-1000 gated imaging is treated as a bremsstrahlung density proxy and reports about 5 mm radiating diameter, about 5 cm radiating length, and 30-50 ns dense-structure lifetime (`KnowledgeReference/scholz-2006-pf1000-mega-joule.md:333-346`, `375-383`, `420`).
- Tests prove matching synthetic image geometry is exported as the density component while tier 4 remains `not_validated`.
- Verification status: `python3 -m pytest tests/test_mhd_physics_integration.py::test_app_exports_pf1000_xray_geometry_as_density_component tests/test_synthetic_diagnostics.py::TestXrayImaging::test_radiating_pinch_geometry_from_image tests/test_kr_targets.py::test_pf1000_spatial_pinch_geometry_evidence_covers_density_only -q` passed; `python3 -m py_compile app_mhd.py tests/test_mhd_physics_integration.py src/dpf/diagnostics/xray_imaging.py src/dpf/validation/kr_targets.py` passed.
- Remaining scientific limit: this is still only a density-proxy geometry comparison for PF-1000. It does not validate magnetic-field maps, temperature fields, or same-shot spatial state.

Ratchet update 2026-05-05, app-level LLNL EM fluctuation spatial component:

- Modules touched: `app_mhd.py` and `tests/test_mhd_physics_integration.py`.
- App post-processing now detects `em_probe_times_s` plus `em_probe_signal` on LLNL 1.2 kJ DPF results and appends `llnl_12kj_em_fluctuation_evidence_from_signal()` as a magnetic-field/EM spatial component.
- A 3-4 GHz dominant probe signal can satisfy the EM component, but the combined spatial evidence remains a candidate unless density and temperature are validated in the same scope.
- KnowledgeReference basis: the LLNL comparison paper describes an EM pick-up probe with 5 GHz bandwidth and reports high-quality pinch activity and simulated Ez probe power in the 3-4 GHz band, with simulated 10-40 T pinch fields and lower-hybrid context (`KnowledgeReference/comparisons-of-dense-plasma-focus-kinetic-simulations-with-experimental-measurements.md:120-122`, `156-164`, `168-170`, `234-238`).
- Tests prove a 3.5 GHz synthetic signal is exported as a magnetic-field/EM component and a low-frequency signal is rejected by the KR target helper.
- Verification status: `python3 -m pytest tests/test_mhd_physics_integration.py::test_app_exports_llnl_em_probe_as_magnetic_component tests/test_kr_targets.py::test_llnl_em_fluctuation_evidence_detects_3_to_4_ghz_band tests/test_kr_targets.py::test_llnl_em_fluctuation_evidence_rejects_low_frequency_signal -q` passed; `python3 -m py_compile app_mhd.py tests/test_mhd_physics_integration.py src/dpf/validation/kr_targets.py` passed.
- Remaining scientific limit: this is an EM fluctuation comparison, not a calibrated magnetic-field map, and it cannot be merged with PF-1000 density or generic temperature evidence into tier-4 validation because the scopes differ.

Ratchet update 2026-05-05, MJOLNIR neutron spectrum gate:

- Modules touched: `src/dpf/validation/kr_targets.py`, `src/dpf/validation/quality_assessment.py`, `src/dpf/validation/__init__.py`, `app_mhd.py`, `tests/test_kr_targets.py`, `tests/test_quality_assessment.py`, and `tests/test_mhd_physics_integration.py`.
- Added `mjolnir_neutron_spectrum_evidence()` to compare mechanism-separated neutron energies against the KR expectation of a narrow thermonuclear spectrum around 2.45 MeV and a broader beam-target spectrum extending to higher energies.
- Tightened tier 5 from "mechanism/timing" to "mechanism/timing/spectrum": timing evidence alone now remains `decomposed_estimate`; tier 5 is `supported` only when timing and spectrum evidence both pass.
- App post-processing now exports `neutron_spectrum_validation` for MJOLNIR results that provide `neutron_spectrum_samples_MeV` with `thermonuclear` and `beam_target` samples.
- KnowledgeReference basis: the MJOLNIR neutron-generation paper separates thermonuclear stagnation emission from beam-target disruption emission and reports spectrum/anisotropy expectations, including narrow 2.45 MeV thermonuclear neutrons and broader high-energy beam-target neutrons up to about 5 MeV (`KnowledgeReference/neutron-generation-dynamics-inside-a-ma-class-dense-plasma-focus-z-pinch-5.md:405-448`, `548-616`).
- Tests prove KR-like timing plus spectrum supports tier 5, timing alone does not, and a thermal-like beam spectrum is rejected.
- Verification status: `python3 -m pytest tests/test_kr_targets.py::test_mjolnir_timing_evidence_passes_with_kr_like_history tests/test_kr_targets.py::test_mjolnir_spectrum_evidence_requires_narrow_thermo_and_broad_beam tests/test_kr_targets.py::test_mjolnir_spectrum_evidence_rejects_thermal_like_beam tests/test_quality_assessment.py::TestQualityAssessment::test_neutron_timing_evidence_requires_both_mechanisms tests/test_quality_assessment.py::TestQualityAssessment::test_predictive_readiness_passes_only_with_all_required_tiers tests/test_mhd_physics_integration.py::test_mjolnir_neutron_timing_evidence_is_exported_when_phase_timed -q` passed; `python3 -m py_compile app_mhd.py src/dpf/validation/quality_assessment.py src/dpf/validation/kr_targets.py src/dpf/validation/__init__.py tests/test_kr_targets.py tests/test_quality_assessment.py tests/test_mhd_physics_integration.py` passed.
- Remaining scientific limit: this still does not validate neutron angular anisotropy, detector response, or real shot-to-shot yield statistics. It prevents timing-only evidence from being overstated.

Ratchet update 2026-05-05, MJOLNIR neutron anisotropy gate:

- Modules touched: `src/dpf/validation/kr_targets.py`, `src/dpf/validation/quality_assessment.py`, `src/dpf/validation/__init__.py`, `app_mhd.py`, `tests/test_kr_targets.py`, `tests/test_quality_assessment.py`, and `tests/test_mhd_physics_integration.py`.
- Added `mjolnir_neutron_anisotropy_evidence()` to compare on-axis/off-axis neutron yields against the KR anisotropy targets.
- Tightened tier 5 again: supported neutron validation now requires mechanism timing, spectrum, and anisotropy evidence. Timing plus spectrum without anisotropy remains `decomposed_estimate`.
- App post-processing now exports `neutron_anisotropy_validation` for MJOLNIR results that provide `neutron_anisotropy` with `on_axis_yield`, `off_axis_yield`, and optional `yield_regime`.
- KnowledgeReference basis: the MJOLNIR neutron paper reports low-yield on-axis/off-axis activation within about 10 percent error and high-yield on-axis activation about 60-100 percent higher than off-axis depending on reaction channel (`KnowledgeReference/neutron-generation-dynamics-inside-a-ma-class-dense-plasma-focus-z-pinch-5.md:596-616`).
- Tests prove high-yield on-axis excess passes, the wrong high-yield trend fails, and app-level MJOLNIR results need timing, spectrum, and anisotropy together for tier 5 support.
- Verification status: `python3 -m pytest tests/test_kr_targets.py::test_mjolnir_timing_evidence_passes_with_kr_like_history tests/test_kr_targets.py::test_mjolnir_anisotropy_evidence_accepts_high_yield_on_axis_excess tests/test_kr_targets.py::test_mjolnir_anisotropy_evidence_rejects_wrong_high_yield_trend tests/test_quality_assessment.py::TestQualityAssessment::test_neutron_timing_evidence_requires_both_mechanisms tests/test_quality_assessment.py::TestQualityAssessment::test_predictive_readiness_passes_only_with_all_required_tiers tests/test_mhd_physics_integration.py::test_mjolnir_neutron_timing_evidence_is_exported_when_phase_timed -q` passed; `python3 -m py_compile app_mhd.py src/dpf/validation/quality_assessment.py src/dpf/validation/kr_targets.py src/dpf/validation/__init__.py tests/test_kr_targets.py tests/test_quality_assessment.py tests/test_mhd_physics_integration.py` passed.
- Remaining scientific limit: anisotropy validation still depends on simulated detector/activation outputs supplied to the result. The app does not yet produce calibrated detector response from first principles.

Ratchet update 2026-05-05, circuit waveform source-authority gate:

- Modules touched: `src/dpf/validation/quality_assessment.py`, `src/dpf/validation/experimental_devices.py`, and `tests/test_quality_assessment.py`.
- `circuit_validation_evidence_from_waveform()` now records device `kr_status`, `reliability`, and `waveform_provenance`.
- With `require_kr_verified=True` by default, tier-1 circuit evidence can pass only for measured, KR-verified waveform records. Reconstructed, reference-only, or unverified waveforms are blocked even if their numerical waveform-shape metrics match.
- Marked the standard PF-1000 measured Scholz waveform record as `kr_status="verified"` because it is sourced to the local PF-1000 KnowledgeReference files already used in this review.
- Tests prove the PF-1000 measured waveform still supports tier 1, while the reconstructed unverified PF-1000-16kV waveform is rejected by source authority.
- Verification status: `python3 -m pytest tests/test_quality_assessment.py::TestQualityAssessment::test_circuit_waveform_evidence_can_support_tier_one tests/test_quality_assessment.py::TestQualityAssessment::test_circuit_waveform_evidence_rejects_unverified_reconstructed_trace tests/test_mhd_physics_integration.py::test_app_exports_circuit_waveform_validation_for_registered_device -q` passed; `python3 -m py_compile src/dpf/validation/quality_assessment.py src/dpf/validation/experimental_devices.py tests/test_quality_assessment.py app_mhd.py tests/test_mhd_physics_integration.py` passed.
- Remaining scientific limit: only PF-1000 standard circuit evidence was upgraded here. Other device records need line-by-line KR status work before their waveforms can support validation claims.

Ratchet update 2026-05-05, validation claim wording cleanup:

- Modules touched: `app_mhd.py`, `src/dpf/validation/quality_assessment.py`, and `CodexFindings.md`.
- Removed the app docstring phrase that called the 0D Lee/snowplow phase "well-validated" without attached evidence.
- Updated the validation-tier wording in this findings file and the tier-5 `validation_role` to reflect the stricter mechanism/timing/spectrum/anisotropy gate.
- Verification status: `python3 -m py_compile app_mhd.py src/dpf/validation/quality_assessment.py` passed; `python3 -m pytest tests/test_quality_assessment.py::TestQualityAssessment::test_predictive_readiness_passes_only_with_all_required_tiers tests/test_mhd_physics_integration.py::test_mjolnir_neutron_timing_evidence_is_exported_when_phase_timed -q` passed.
- Remaining scientific limit: wording cleanup reduces overclaiming; it does not add new validation data.

Ratchet update 2026-05-05, snowplow phase-target source-authority gate:

- Modules touched: `src/dpf/validation/quality_assessment.py`, `app_mhd.py`, `tests/test_quality_assessment.py`, and `tests/test_mhd_physics_integration.py`.
- `snowplow_validation_evidence_from_phase_history()` now requires reference phase targets to come from `KnowledgeReference/` and carry `reference_kr_status="verified"` before tier-2 evidence can pass.
- App post-processing reads `snowplow_phase_target_metadata` and passes the target source authority into the snowplow validation helper.
- Tests prove verified KR phase targets can support tier 2, unverified naked targets are rejected, and missing pinch observations still fail even with verified metadata.
- Verification status: `python3 -m pytest tests/test_quality_assessment.py::TestQualityAssessment::test_snowplow_phase_history_can_support_tier_two_with_targets tests/test_quality_assessment.py::TestQualityAssessment::test_snowplow_phase_history_rejects_unverified_targets tests/test_quality_assessment.py::TestQualityAssessment::test_snowplow_phase_history_requires_pinch_observation tests/test_mhd_physics_integration.py::test_app_exports_target_backed_snowplow_validation -q` passed; `python3 -m py_compile app_mhd.py src/dpf/validation/quality_assessment.py tests/test_quality_assessment.py tests/test_mhd_physics_integration.py` passed.
- Remaining scientific limit: this blocks unsourced phase-target spoofing, but production device/shot phase targets still need to be extracted and marked verified before ordinary runs can support tier 2.

Ratchet update 2026-05-05, spatial component source-authority gate:

- Modules touched: `src/dpf/validation/quality_assessment.py`, `tests/test_quality_assessment.py`, and `tests/test_mhd_physics_integration.py`.
- `combine_spatial_validation_evidence()` now counts only KR-sourced tier-4 component evidence with `model_role` beginning `simulation_to_kr_` when building a full spatial validation pass.
- Same-scope density, magnetic-field, and temperature components can still support tier 4 when each component has KR source authority.
- Unsourced components with matching scope no longer contribute to the combined spatial diagnostics.
- Tests prove cross-scope KR components remain blocked, same-scope KR components support tier 4, unsourced same-scope components are rejected, malformed tier metadata fails closed, and app-level spatial promotion still works when components include source authority.
- Verification status: `python3 -m pytest tests/test_quality_assessment.py::TestQualityAssessment::test_combined_spatial_evidence_requires_consistent_scope tests/test_quality_assessment.py::TestQualityAssessment::test_combined_spatial_evidence_supports_tier_four_when_scope_matches tests/test_quality_assessment.py::TestQualityAssessment::test_combined_spatial_evidence_rejects_unsourced_components tests/test_quality_assessment.py::TestQualityAssessment::test_combined_spatial_evidence_handles_malformed_tier_metadata tests/test_mhd_physics_integration.py::test_app_promotes_complete_same_scope_spatial_components tests/test_mhd_physics_integration.py::test_app_exports_pf1000_xray_geometry_as_density_component tests/test_mhd_physics_integration.py::test_app_exports_llnl_em_probe_as_magnetic_component -q` passed; `python3 -m py_compile src/dpf/validation/quality_assessment.py tests/test_quality_assessment.py tests/test_mhd_physics_integration.py app_mhd.py` passed.
- Remaining scientific limit: this blocks forged spatial components, but it still does not supply a same-shot/device density, magnetic-field, and temperature dataset.

Ratchet update 2026-05-05, neutron evidence source-authority gate:

- Modules touched: `src/dpf/validation/quality_assessment.py`, `tests/test_quality_assessment.py`, `tests/test_kr_targets.py`, and `tests/test_mhd_physics_integration.py`.
- Tier-5 validation now counts timing, spectrum, and anisotropy evidence only when each evidence object is KR-sourced, tier-5, and has a `model_role` beginning `simulation_to_kr_`.
- Generic or forged neutron evidence dictionaries can still be carried as estimates, but they no longer support predictive-readiness tier 5.
- Tests prove KR-sourced timing/spectrum/anisotropy supports tier 5, generic timing evidence alone remains `decomposed_estimate`, and app-level MJOLNIR helper output still supports tier 5 when all three KR-backed evidence objects are present.
- Verification status: `python3 -m pytest tests/test_quality_assessment.py::TestQualityAssessment::test_neutron_timing_evidence_requires_both_mechanisms tests/test_quality_assessment.py::TestQualityAssessment::test_predictive_readiness_passes_only_with_all_required_tiers tests/test_quality_assessment.py::TestQualityAssessment::test_validation_tier_report_distinguishes_supported_tiers tests/test_kr_targets.py::test_mjolnir_timing_evidence_passes_with_kr_like_history tests/test_mhd_physics_integration.py::test_mjolnir_neutron_timing_evidence_is_exported_when_phase_timed -q` passed; `python3 -m py_compile src/dpf/validation/quality_assessment.py tests/test_quality_assessment.py tests/test_kr_targets.py tests/test_mhd_physics_integration.py app_mhd.py` passed.
- Remaining scientific limit: source authority protects the gate; calibrated detector/activation outputs and real shot comparisons still need to exist for production runs.

Ratchet update 2026-05-05, MHD verification metadata gate:

- Modules touched: `src/dpf/validation/quality_assessment.py` and `tests/test_quality_assessment.py`.
- `mhd_verification_evidence_from_tests()` now emits `validation_tier=3` and `model_role="code_verification_analytic_tests"`.
- Tier 3 now requires explicit code-verification metadata in addition to passing Sod and Brio-Wu analytic-test labels.
- Bare dictionaries with `passed: true` and named analytic tests no longer support the MHD verification tier.
- Tests prove helper-generated evidence supports tier 3, shock-tube adapter evidence still supports tier 3, and bare named-test dictionaries remain `verification_only`.
- Verification status: `python3 -m pytest tests/test_quality_assessment.py::TestQualityAssessment::test_mhd_verification_evidence_requires_named_analytic_tests tests/test_quality_assessment.py::TestQualityAssessment::test_mhd_verification_rejects_bare_named_tests_without_metadata tests/test_quality_assessment.py::TestQualityAssessment::test_mhd_shock_tube_results_can_support_tier_three tests/test_quality_assessment.py::TestQualityAssessment::test_predictive_readiness_passes_only_with_all_required_tiers -q` passed; `python3 -m py_compile src/dpf/validation/quality_assessment.py tests/test_quality_assessment.py` passed.
- Remaining scientific limit: this is code verification only. It does not validate DPF spatial fields.

Ratchet update 2026-05-05, V&V report source-authority cleanup:

- Modules touched: `src/dpf/validation/vv_report.py` and `tests/test_vv_report.py`.
- The V&V report no longer presents the device registry as hardcoded device-level PASS/accuracy validation. It now reports device source status: `VALIDATION_READY`, `RECONSTRUCTED_ONLY`, `REFERENCE_ONLY`, `KR_UNVERIFIED`, `WAVEFORM_KR_UNVERIFIED`, or `INCOMPLETE_WAVEFORM`.
- A device is `VALIDATION_READY` only when its registry record is KR-verified, measured-reliability, and backed by a measured waveform. Reconstructed and reference-only records are explicitly excluded from validation claims.
- Removed the hardcoded PF-1000 statistical-validation `PASS` row from the report. Statistical validation is now marked source-gated until an explicit KR-sourced comparison bundle is attached.
- Tests prove PF-1000 is validation-ready by source authority, PF-1000-16kV is reconstructed-only, NX2 is reference-only, and the report no longer emits the old pass/accuracy phrases.
- Verification status: `python3 -m pytest tests/test_vv_report.py -q` passed; `python3 -m py_compile src/dpf/validation/vv_report.py tests/test_vv_report.py` passed.
- Remaining scientific limit: this is a reporting cleanup. It prevents overclaiming, but it does not run a new validation comparison or create missing spatial/neutron evidence.

Ratchet update 2026-05-05, V&V report radiation wording:

- Modules touched: `src/dpf/validation/vv_report.py` and `tests/test_vv_report.py`.
- The module coverage table no longer calls line radiation `CHIANTI-style`. It now says `Line radiation (empirical coronal fits)`, matching the provenance metadata added to `src/dpf/radiation/line_radiation.py`.
- Tests prove the V&V report includes the empirical wording and excludes the old CHIANTI-style label.
- Verification status: `python3 -m pytest tests/test_vv_report.py tests/test_radiation_model_metadata.py -q` passed; `python3 -m py_compile src/dpf/validation/vv_report.py tests/test_vv_report.py` passed.
- Remaining scientific limit: wording is now honest, but high-Z/dopant radiation still lacks tabulated EOS/opacities/transport validation.

Ratchet update 2026-05-05, waveform-source authority split:

- Modules touched: `src/dpf/validation/experimental_device.py`, `src/dpf/validation/experimental_devices.py`, `src/dpf/validation/quality_assessment.py`, `src/dpf/validation/vv_report.py`, `tests/test_quality_assessment.py`, and `tests/test_vv_report.py`.
- Added `waveform_kr_status` to `ExperimentalDevice` so device-parameter source authority and waveform-trace source authority are no longer conflated.
- `circuit_validation_evidence_from_waveform()` now requires `kr_status="verified"`, `reliability="measured"`, `waveform_provenance="measured"`, and `waveform_kr_status="verified"` before tier-1 circuit evidence can pass under the default KR-only policy.
- PF-1000 remains validation-ready because the standard Scholz waveform is hand-digitized from the on-file PF-1000 KnowledgeReference. POSEIDON-60kV and UNU-ICTP now report `WAVEFORM_KR_UNVERIFIED`: their device tables are KR-supported, but their current waveform arrays are from IPFS/external archive traces rather than the local corpus.
- The V&V report now shows a separate `Waveform KR` column and reports `1/9 devices validation-ready` under the stricter source rule.
- Tests prove PF-1000 still supports tier 1, reconstructed PF-1000-16kV is rejected, external-archive POSEIDON-60kV is rejected even when the waveform numerically matches, and the V&V report exposes the new statuses.
- Verification status: `python3 -m pytest tests/test_quality_assessment.py::TestQualityAssessment::test_circuit_waveform_evidence_can_support_tier_one tests/test_quality_assessment.py::TestQualityAssessment::test_circuit_waveform_evidence_rejects_unverified_reconstructed_trace tests/test_quality_assessment.py::TestQualityAssessment::test_circuit_waveform_evidence_rejects_external_archive_trace tests/test_vv_report.py -q` passed; `python3 -m py_compile src/dpf/validation/experimental_device.py src/dpf/validation/experimental_devices.py src/dpf/validation/quality_assessment.py src/dpf/validation/vv_report.py tests/test_quality_assessment.py tests/test_vv_report.py` passed.
- Remaining scientific limit: this is source filtering, not new validation. POSEIDON-60kV and UNU-ICTP can be promoted only after their waveform traces are either tied directly to on-file KnowledgeReference figures/tables or the external waveform sources are ingested into `KnowledgeReference`.

Ratchet update 2026-05-05, app-level external waveform block:

- Modules touched: `tests/test_mhd_physics_integration.py`.
- Added an app-level regression proving a POSEIDON-60kV result that exactly matches its registered waveform still fails `circuit_validation` and remains `diagnostic_present` at tier 1 because `waveform_kr_status` is `unverified`.
- Verification status: `python3 -m pytest tests/test_mhd_physics_integration.py::test_app_exports_circuit_waveform_validation_for_registered_device tests/test_mhd_physics_integration.py::test_app_blocks_external_archive_waveform_from_tier_one tests/test_quality_assessment.py::TestQualityAssessment::test_circuit_validation_tier_requires_source_authority -q` passed.
- Remaining scientific limit: this locks app behavior to the source gate; it does not verify the POSEIDON waveform source.

Ratchet update 2026-05-05, README validation-claim alignment:

- Modules touched: `README.md` and `tests/test_readme_claims.py`.
- The README no longer opens by saying the simulator is validated against four device waveforms, and no longer claims six-device zero-calibration circuit validation or a validation campaign as scientific evidence.
- The validation section now states the KR-only source gate, the five-tier predictive-readiness blocker, and the current device-source status: only the standard PF-1000 Scholz waveform record is validation-ready under the local corpus rule.
- The PF-1000 section is now a circuit source record rather than a pass/fail validation result, and the engineering campaign is explicitly labeled as regression/engineering testing rather than scientific validation evidence.
- Added a README claim regression test so the broad validation phrases cannot be reintroduced silently.
- Verification status: `python3 -m pytest tests/test_readme_claims.py -q` passed; `python3 -m py_compile tests/test_readme_claims.py` passed.
- Remaining scientific limit: this ratchet fixes public claim hygiene. It does not add missing spatial, snowplow phase, or neutron validation evidence.

Ratchet update 2026-05-05, SCOPE validation-claim alignment:

- Modules touched: `docs/SCOPE.md` and `tests/test_scope_claims.py`.
- `docs/SCOPE.md` no longer describes the Lee/snowplow model as validated against six devices or lists a `6/7 PASS` cross-device validation claim.
- The scope document now frames DPF-Unified as a Lee-MHD simulation workbench with source-gated validation evidence, and states that only the standard PF-1000 Scholz waveform is currently tier-1 validation-ready under the KR-only rule.
- Historical PF-1000 error notes are now explicitly engineering context, not predictive-readiness evidence.
- The citation guidance now says circuit waveform comparison infrastructure with source-gated PF-1000 tier-1 evidence, not end-to-end DPF validation.
- Added a scope-claim regression test to prevent the old six-device/pass language from returning.
- Verification status: `python3 -m pytest tests/test_scope_claims.py -q` passed; `python3 -m py_compile tests/test_scope_claims.py` passed.
- Remaining scientific limit: this is documentation claim control only. The missing tier-2 phase targets, tier-4 spatial comparisons, and tier-5 calibrated neutron diagnostics are still open.

Ratchet update 2026-05-05, V&V summary source-gating:

- Modules touched: `docs/V_AND_V_SUMMARY.md` and `tests/test_v_and_v_summary_claims.py`.
- The V&V summary no longer lists a circuit-level `VALIDATED` section with six devices passing, zero calibration, 24-shot PF-1000 statistics, or IPFS/archive traces as validation evidence.
- The validation section now mirrors the source-authority model: PF-1000 is the only currently validation-ready registered device; POSEIDON-60kV and UNU-ICTP are blocked by waveform source; reconstructed/reference-only records remain blocked.
- The MHD section still reports code verification, but explicitly says same-scope density/B-field/temperature validation has not been produced.
- Added a V&V summary claim regression test to prevent the old pass table from returning.
- Verification status: `python3 -m pytest tests/test_v_and_v_summary_claims.py -q` passed; `python3 -m py_compile tests/test_v_and_v_summary_claims.py` passed.
- Remaining scientific limit: this fixes stale V&V documentation; it does not produce the experimental evidence bundles needed for predictive readiness.

Ratchet update 2026-05-05, JOSS draft validation-claim withdrawal:

- Modules touched: `docs/joss-paper-draft.md` and `tests/test_joss_draft_claims.py`.
- The JOSS draft now carries a 2026-05-05 status note marking it stale under the current KnowledgeReference-only validation gate.
- Removed the seven-device validation claim, the six-device waveform pass table, and the PF-1000 24-shot statistical validation claim from the active draft text.
- Replaced the validation section with source-gated status: PF-1000 circuit waveform is tier-1 source-ready only; external/reconstructed waveforms are blocked; spatial MHD and neutron evidence remain open.
- Added a draft-claim regression test so the withdrawn paper claims cannot reappear unnoticed.
- Verification status: `python3 -m pytest tests/test_joss_draft_claims.py -q` passed; `python3 -m py_compile tests/test_joss_draft_claims.py` passed.
- Remaining scientific limit: this prevents outward-facing publication overclaiming; it does not create validation evidence.

Ratchet update 2026-05-05, AI disclosure validation-claim alignment:

- Modules touched: `docs/AI_DISCLOSURE.md` and `tests/test_ai_disclosure_claims.py`.
- Removed the old six-device waveform validation and PF-1000 24-shot statistical validation language from the AI usage disclosure.
- The disclosure now says validation claims are source-gated against local `KnowledgeReference/` records, and that only the standard PF-1000 Scholz waveform is currently validation-ready for tier-1 circuit evidence.
- Added a disclosure claim regression test to keep the withdrawn validation numbers out of the current-facing disclosure.
- Verification status: `python3 -m pytest tests/test_ai_disclosure_claims.py -q` passed; `python3 -m py_compile tests/test_ai_disclosure_claims.py` passed.
- Remaining scientific limit: this is documentation claim hygiene, not new validation.

Ratchet update 2026-05-05, validation post-processing error visibility:

- Modules touched: `app_mhd.py` and `tests/test_mhd_physics_integration.py`.
- App post-processing no longer silently drops failures in the validation evidence stages. Circuit waveform, snowplow phase, spatial validation, MJOLNIR neutron validation, and predictive-readiness exceptions are now recorded under `result["validation_errors"]` with stage, exception type, and message.
- The simulation remains non-fatal, but a broken validation helper now becomes explicit evidence of a validation pipeline failure instead of looking like missing evidence.
- Added an integration test that forces circuit evidence generation to fail and proves the app records the validation error while still exporting predictive readiness.
- Verification status: `python3 -m pytest tests/test_mhd_physics_integration.py::test_app_exports_circuit_waveform_validation_for_registered_device tests/test_mhd_physics_integration.py::test_app_records_validation_errors_when_evidence_generation_fails tests/test_mhd_physics_integration.py::test_predictive_readiness_exported_and_blocks_unvalidated_claims -q` passed; `python3 -m py_compile app_mhd.py tests/test_mhd_physics_integration.py` passed.
- Remaining scientific limit: this improves validation pipeline integrity only. It does not create the missing validation evidence.

Ratchet update 2026-05-05, validation pipeline errors block readiness:

- Modules touched: `src/dpf/validation/quality_assessment.py` and `tests/test_quality_assessment.py`.
- `predictive_readiness_report()` now treats `result["validation_errors"]` as first-class blockers. Even if all five evidence tiers are otherwise present, readiness is blocked with status `validation_pipeline_error` until the pipeline errors are cleared.
- The blocker list now includes the failing validation stage, exception type, and message, so downstream tools can distinguish "missing evidence" from "evidence generation failed."
- Added a focused test proving a fully supported evidence package is downgraded when a validation pipeline error exists.
- Verification status: `python3 -m pytest tests/test_quality_assessment.py::TestQualityAssessment::test_predictive_readiness_passes_only_with_all_required_tiers tests/test_quality_assessment.py::TestQualityAssessment::test_predictive_readiness_blocks_validation_pipeline_errors tests/test_mhd_physics_integration.py::test_app_records_validation_errors_when_evidence_generation_fails -q` passed; `python3 -m py_compile src/dpf/validation/quality_assessment.py tests/test_quality_assessment.py app_mhd.py tests/test_mhd_physics_integration.py` passed.
- Remaining scientific limit: this protects the readiness gate from silent failures; it does not supply new experimental comparisons.

Ratchet update 2026-05-05, raw waveform comparison source metadata:

- Modules touched: `src/dpf/validation/experimental_comparison.py` and `tests/test_experimental_comparison_source_authority.py`.
- `validate_current_waveform()` now returns `source_authority` metadata with device `kr_status`, `reliability`, `waveform_provenance`, `waveform_kr_status`, `validation_ready`, and `validation_role`.
- Numeric peak/timing/NRMSE metrics remain available, but the return value now says whether they are a `tier1_circuit_evidence_candidate` or only a `numeric_comparison_only`.
- Tests prove PF-1000 is marked tier-1 candidate while POSEIDON-60kV, despite perfect self-comparison NRMSE against its own array, is marked numeric-only because the waveform source is not KR-verified.
- Verification status: `python3 -m pytest tests/test_experimental_comparison_source_authority.py tests/test_quality_assessment.py::TestQualityAssessment::test_circuit_waveform_evidence_can_support_tier_one tests/test_quality_assessment.py::TestQualityAssessment::test_circuit_waveform_evidence_rejects_external_archive_trace -q` passed; `python3 -m py_compile src/dpf/validation/experimental_comparison.py tests/test_experimental_comparison_source_authority.py` passed.
- Remaining scientific limit: this improves provenance visibility for raw metrics; callers still need to use strict evidence helpers and source-backed targets for validation claims.

Ratchet update 2026-05-05, raw neutron-yield comparison source metadata:

- Modules touched: `src/dpf/validation/experimental_comparison.py` and `tests/test_experimental_comparison_source_authority.py`.
- `validate_neutron_yield()` now returns `source_authority` and `validity_notes` stating that total-yield order checks are `numeric_yield_comparison_only`, not tier-5 neutron physics validation.
- The returned metadata explicitly says `validation_ready=False` because total yield alone does not validate neutron mechanism, timing, spectrum, anisotropy, detector response, or beam-target physics.
- Tests prove a PF-1000 exact-yield match remains within order of magnitude but is still marked numeric-only.
- Verification status: `python3 -m pytest tests/test_experimental_comparison_source_authority.py tests/test_neutron_yield.py::TestValidateNeutronYield -q` passed; `python3 -m py_compile src/dpf/validation/experimental_comparison.py tests/test_experimental_comparison_source_authority.py` passed.
- Remaining scientific limit: this keeps useful yield sanity checks while preventing them from substituting for KR-backed tier-5 evidence.

Ratchet update 2026-05-05, validation-ready device registry helper:

- Modules touched: `src/dpf/validation/experimental_devices.py`, `src/dpf/validation/experimental.py`, `src/dpf/validation/__init__.py`, and `tests/test_experimental_comparison_source_authority.py`.
- Added `get_validation_ready_devices()` as the canonical registry query for tier-1 circuit source authority.
- The helper requires KR-verified device parameters, measured reliability, measured waveform provenance, KR-verified waveform source, and waveform arrays.
- It currently returns only `PF-1000`, matching the V&V report and README/SCOPE claim surfaces.
- Verification status: `python3 -m pytest tests/test_experimental_comparison_source_authority.py tests/test_vv_report.py -q` passed; `python3 -m py_compile src/dpf/validation/experimental_devices.py src/dpf/validation/experimental.py src/dpf/validation/__init__.py tests/test_experimental_comparison_source_authority.py` passed.
- Remaining scientific limit: this helper centralizes the current source gate. It does not promote additional devices until their waveform records are verified from `KnowledgeReference`.

Ratchet update 2026-05-05, tier-1 circuit source-authority hardening:

- Modules touched: `src/dpf/validation/quality_assessment.py` and `tests/test_quality_assessment.py`.
- `validation_tier_report()` no longer accepts bare `circuit_validation` dictionaries as tier-1 support just because peak-current, peak-time, and waveform-shape metrics are true.
- Tier 1 now requires circuit evidence source authority: either `source_authority.validation_ready=True`, `source_authority.passed=True`, or the explicit KR-verified/measured device and waveform fields.
- Tests prove complete-but-unsourced circuit evidence remains only `diagnostic_present`, while PF-1000 evidence from the source-authority helper still supports tier 1.
- Verification status: `python3 -m pytest tests/test_quality_assessment.py::TestQualityAssessment::test_predictive_readiness_passes_only_with_all_required_tiers tests/test_quality_assessment.py::TestQualityAssessment::test_validation_tier_report_distinguishes_supported_tiers tests/test_quality_assessment.py::TestQualityAssessment::test_circuit_validation_evidence_can_support_tier_one tests/test_quality_assessment.py::TestQualityAssessment::test_circuit_validation_tier_requires_source_authority tests/test_quality_assessment.py::TestQualityAssessment::test_circuit_waveform_evidence_can_support_tier_one -q` passed; `python3 -m py_compile src/dpf/validation/quality_assessment.py tests/test_quality_assessment.py` passed.
- Remaining scientific limit: this blocks unsourced tier-1 spoofing. It still does not create source-backed phase, spatial, or neutron evidence.

Ratchet update 2026-05-05, tier-2 and tier-4 source-authority hardening:

- Modules touched: `src/dpf/validation/quality_assessment.py` and `tests/test_quality_assessment.py`.
- `validation_tier_report()` now requires source authority for snowplow phase/timing evidence and spatial DPF validation evidence.
- Tier 2 support requires verified KR phase-target source authority, so complete phase-error dictionaries without source metadata remain `partial`.
- Tier 4 support requires either direct KR-sourced tier-4 `simulation_to_kr_` metadata or a combined spatial evidence object whose components all passed source authority. Bare density/B-field/temperature dictionaries no longer support spatial validation.
- Tests prove unsourced complete snowplow and spatial evidence are blocked, while target-backed snowplow validation and same-scope KR spatial components still support their tiers.
- Verification status: `python3 -m pytest tests/test_quality_assessment.py -q` passed; `python3 -m pytest tests/test_mhd_physics_integration.py::test_app_exports_target_backed_snowplow_validation tests/test_mhd_physics_integration.py::test_app_promotes_complete_same_scope_spatial_components tests/test_mhd_physics_integration.py::test_predictive_readiness_exported_and_blocks_unvalidated_claims -q` passed; `python3 -m py_compile src/dpf/validation/quality_assessment.py tests/test_quality_assessment.py` passed.
- Remaining scientific limit: this closes another spoofing path in the readiness gate. It still leaves the real work of extracting and comparing same-shot/source-backed phase and spatial diagnostics.

## Finding 8: Several foundational modules are worth keeping and building around

Severity: Positive

Code evidence:

- `src/dpf/diagnostics/neutron_yield.py:1-19` clearly limits itself to thermonuclear Bosch-Hale yield.
- `src/dpf/diagnostics/beam_target.py:1-16` documents the Lee/Saw beam-target formula and calibration.
- `src/dpf/radiation/bremsstrahlung.py:1-45` derives its SI coefficient from the NRL formulary.
- `src/dpf/engine/circuit_coupling.py:87-138` has a conservative MHD handoff/blending gate rather than blindly trusting early unresolved MHD inductance.

KnowledgeReference basis:

- `KnowledgeReference/bosch-hale-1992-fusion-reactivity.md:24-36`, `KnowledgeReference/bosch-hale-1992-fusion-reactivity.md:81-110`.
- `KnowledgeReference/lee_radpf_theory.md:4060-4104`.
- `KnowledgeReference/original-deuteron-beam-fluence-emitted-from-dense-plasma-focus.md:219-272`.
- `KnowledgeReference/plasma-formulary.md:3165-3167`, `KnowledgeReference/plasma-formulary.md:5064-5109`.
- `KnowledgeReference/neutron-generation-dynamics-inside-a-ma-class-dense-plasma-focus-z-pinch-5.md:276-335`.

Explanation:

The codebase is not scientifically empty or unserious. It contains credible pieces that align with the on-file literature. The problem is scope control: source-backed components, empirical closures, smoke tests, and aspirational full-MHD claims are currently mixed together in user-facing descriptions.

Required resolution:

Preserve these modules, but strengthen provenance labels and output metadata so every result says which physical model produced it and whether that model is KR-derived, empirical, or experimental.

Ratchet update 2026-05-05:

- This positive finding is now partially acted on across multiple modules.
- Source-backed or source-limited diagnostics now expose machine-readable provenance metadata in neutron yield, scaling laws, radiation cooling, p-B11 yield, and validation quality assessment.
- The current pattern to keep: preserve useful formulas and solvers, but attach `model_role`, `validation_role`, component decomposition, and validity notes wherever a result might be misread as predictive validation.
- Remaining work: consolidate these metadata patterns into a shared provenance schema once more modules are converted. For now the implementation is intentionally local to each module to avoid a broad refactor during the ratchet loop.

## Remaining High-Fidelity Scientific Accuracy Plan

This is the current plan for what remains before the project can honestly be called a validated end-to-end predictive DPF simulation tool under the `KnowledgeReference`-only rule. Each item has a next ratchet and a done condition so progress can be checked mechanically.

1. Source-authority data curation

Current gap: only PF-1000 is currently validation-ready for tier-1 circuit waveform evidence. POSEIDON-60kV and UNU-ICTP still fail the KR-only waveform-source rule because their waveform arrays are external or reconstructed records, not verified `KnowledgeReference` waveform records.

KnowledgeReference basis: the current findings and source gates are tied to the verified local corpus requirement, and the device registry now enforces measured, KR-verified waveform provenance before promoting circuit evidence.

Next ratchet: extract every device/shot waveform, device geometry, bank parameter, fill pressure, timing marker, uncertainty, and provenance note from `KnowledgeReference` into a line-referenced target registry.

Done condition: every registered validation device has source path, line range, measurement/provenance status, waveform authority, uncertainty class, and a pass/fail reason. No device can enter tier-1 evidence without that registry entry.

2. Tier-2 snowplow phase validation

Current gap: the code can compare phase histories when supplied with phase targets, but ordinary runs do not yet carry device/shot-specific KR phase targets.

KnowledgeReference basis: Lee/RADPF sources define axial/radial/pinch phase semantics and current-rollover/dip fitting, while PF-1000 timing work uses derivative/dip timing as an observed event.

Next ratchet: build source-backed phase target records for each validation device: axial rundown end, radial transit duration, pinch/stagnation time, and the diagnostic used to identify each marker.

Done condition: production simulations attach tier-2 `snowplow_validation` only from same-device, KR-backed phase targets; targetless phase labels remain candidates.

3. Tier-3 high-fidelity numerical verification

Current gap: the gate can accept Sod and Brio-Wu style MHD verification evidence, but that is still generic code verification. It does not prove cylindrical DPF source terms, resistive diffusion, wall/electrode boundary handling, circuit-MHD coupling, or backend parity.

KnowledgeReference basis: local numerical-method references support shock-tube verification as standard MHD checks, while DPF HEDP papers require EOS, conductivity, radiation, and more complete coupling for high-fidelity modeling.

Next ratchet: expand verification to cylindrical source-term preservation, magnetic diffusion/resistive heating, energy accounting with circuit work, backend parity, spatial/time convergence, Orszag-Tang/rotor-style MHD cases, and restart/reproducibility checks.

Done condition: tier-3 evidence includes named analytic/regression tests for ideal and resistive MHD, circuit-energy coupling, cylindrical geometry, and all supported backends, with documented tolerances.

4. Tier-4 spatial DPF experimental validation

Current gap: the code can safely combine density, magnetic-field/EM, and temperature components only when they share one validation scope. Existing components are still partial or cross-scope.

KnowledgeReference basis: PF-1000 imaging supports a density-proxy geometry target, LLNL EM probes support fluctuation-band comparisons, and the DPF review gives broad temperature/magnetic regimes, but these cannot be merged unless they describe the same device/shot/scope.

Next ratchet: ingest same-scope spatial diagnostics from the local corpus: density or emission geometry, magnetic-field/EM diagnostic, temperature or spectrum-derived temperature, timing alignment, and diagnostic uncertainty.

Done condition: tier-4 support requires density, magnetic-field, and temperature evidence from one KR-backed validation scope, with shared device/shot/timing metadata and component source authority.

5. Tier-5 neutron validation

Current gap: tier 5 now requires mechanism timing, spectrum, and anisotropy evidence, but production runs do not yet produce calibrated detector/activation outputs for that full set.

KnowledgeReference basis: the MJOLNIR neutron-generation paper separates thermonuclear stagnation emission from later beam-target emission, reports spectrum broadening, detector time-of-flight context, and anisotropy trends.

Next ratchet: add production outputs for mechanism-separated neutron histories, neutron spectrum samples, angular yield/anisotropy, time-of-flight/detector response, and activation/yield uncertainty.

Done condition: tier-5 support is produced only when the simulation compares mechanism timing, spectrum, anisotropy, and detector/activation observables against one KR-backed neutron validation scope.

6. Missing high-fidelity physics

Current gap: the project still lacks or only approximates several physics blocks identified by the local corpus: tabulated EOS, ionization/dissociation/charge-state kinetics, two-temperature physics, multigroup radiation/opacities, high-Z material mixing, electrode ablation/impurities, Hall/FLR/kinetic/PIC effects, 3D instabilities, flashover/sheath initiation, anomalous resistance, restrike, and post-pinch disruption physics.

KnowledgeReference basis: ALEGRA and Seyler HEDP/Kr-doped DPF sources call out EOS, advanced conductivities, two-temperature and radiation transport; DPF review and MJOLNIR sources call out kinetic, beam-target, instability, finite-Larmor-radius, strong-field, and 3D effects.

Next ratchet: keep existing reduced models but create a physics-fidelity evidence schema that records which of these effects are implemented, verified, validated, empirical, or absent for each run.

Done condition: user-facing predictive claims are allowed only for scopes whose required physics effects are either implemented and validated or explicitly shown not to control the target observable under the relevant KR-backed regime.

7. Circuit-field coupling fidelity

Current gap: the current implementation has better metadata and some MHD feedback, but several paths still use density-weighted or Lee-style inductance/back-EMF closures rather than a validated field-derived Poynting/circuit coupling.

KnowledgeReference basis: the corpus supports circuit and Lee/RADPF waveform fitting for some regimes, but also warns that late pinch and beam-target phases require effects outside ordinary MHD.

Next ratchet: define a `field_coupling_validation` evidence record for inductance, dL/dt, back-EMF, Poynting flux, circuit-energy balance, and transition timing from snowplow to resolved MHD.

Done condition: circuit current in MHD mode is driven by validated field-derived coupling, and the exported result distinguishes snowplow-loaded, blended, and fully field-coupled intervals.

8. Uncertainty quantification and statistical validation

Current gap: waveform uncertainty tools exist, but high-fidelity predictive readiness is not yet tied to a complete uncertainty budget for inputs, numerics, diagnostics, model-form error, and shot-to-shot variability.

KnowledgeReference basis: the local validation code already references ASME V&V and GUM-style uncertainty for waveform validation, but spatial/neutron/model-form budgets are still missing.

Next ratchet: extend UQ records from circuit waveform comparisons into phase, spatial, neutron, and numerical evidence; propagate uncertainty through validation decisions instead of using only point tolerances.

Done condition: every supported validation tier reports experimental, numerical, input, and model-form uncertainty contributions, plus the acceptance rule used for pass/fail.

9. Export and claim hygiene

Current gap: documentation and app outputs now expose readiness blockers, but the remaining scientific-accuracy roadmap should also be machine-readable so downstream UI/API layers cannot hide the scientific blockers.

KnowledgeReference basis: this is a governance/control requirement derived from the KR-only validation rule, not a new physics claim.

Next ratchet: add a structured scientific-accuracy gap report beside `validation_tiers` and `predictive_readiness` in app results.

Done condition: every app result exposes the current blockers, next ratchet, and done condition for the main scientific-accuracy areas.

Ratchet update 2026-05-05, scientific-accuracy gap export:

- Modules touched: `src/dpf/validation/quality_assessment.py`, `src/dpf/validation/__init__.py`, `app_mhd.py`, `tests/test_quality_assessment.py`, and `tests/test_mhd_physics_integration.py`.
- Added `ScientificAccuracyGap` and `scientific_accuracy_gap_report()` as a machine-readable version of the remaining high-fidelity scientific-accuracy plan.
- The report currently covers source-authority data, snowplow phase validation, high-fidelity MHD numerical verification, spatial DPF validation, neutron validation, missing physics fidelity, circuit-field coupling, uncertainty quantification, and export/claim hygiene.
- App post-processing now exports `scientific_accuracy_gaps` beside `validation_tiers` and `predictive_readiness`.
- Ordinary DPF runs now carry not only the readiness failure, but also the next ratchet and done condition for each major blocker.
- Verification status: `python3 -m pytest tests/test_quality_assessment.py -q` passed; `python3 -m pytest tests/test_mhd_physics_integration.py::test_predictive_readiness_exported_and_blocks_unvalidated_claims tests/test_mhd_physics_integration.py::test_app_exports_circuit_waveform_validation_for_registered_device tests/test_mhd_physics_integration.py::test_app_records_validation_errors_when_evidence_generation_fails -q` passed; `python3 -m py_compile src/dpf/validation/quality_assessment.py src/dpf/validation/__init__.py app_mhd.py tests/test_quality_assessment.py tests/test_mhd_physics_integration.py` passed.
- Remaining scientific limit: this ratchet does not add new physics or new validation data. It makes the remaining work unavoidable in result metadata so downstream code cannot present an ordinary run as scientifically complete.

Ratchet update 2026-05-05, tier-2 phase target registry:

- Modules touched: `src/dpf/validation/kr_targets.py`, `src/dpf/validation/__init__.py`, and `tests/test_kr_targets.py`.
- Added `lee_snowplow_phase_semantics_targets()` to record the KR-backed Lee/RADPF phase semantics for axial, radial, and pinch timing. This is a semantics target, not device-specific validation evidence.
- Added `pf1000_16kv_shot12581_phase_targets()` from the PF-1000 2021 Radiation Physics and Chemistry source. It records the 16 kV / 170.5 kJ / 1.2 Torr shot context, current-dip end around 8 us, pinch duration about 212 ns, derivative-dip timing context, fitted Lee factors, and missing full-tier-2 fields.
- Added `pf1000_16kv_phase_candidate_evidence_from_history()` to compare a simulated phase history against that partial PF-1000 target.
- The candidate evidence is intentionally `passed=False`: it can mark pinch timing/duration agreement, but it cannot support tier 2 until same-shot axial rundown end and radial transit targets are also present.
- KnowledgeReference basis: Lee/RADPF phase fitting semantics from `KnowledgeReference/a-course-on-plasma-focus-numerical-experiments-s-lee-and-s-h-saw-part-1-basic-course.md:886-891`, `14922-14936`, `16239-16244`, and `16298-16304`; PF-1000 phase/timing context from `KnowledgeReference/radiation-physics-and-chemistry-188-2021-109633.md:111-124`, `132-137`, `219-235`, `250-285`, and `332-346`.
- Verification status: `python3 -m pytest tests/test_kr_targets.py -q` passed; `python3 -m py_compile src/dpf/validation/kr_targets.py src/dpf/validation/__init__.py tests/test_kr_targets.py` passed.
- Remaining scientific limit: this is a partial target registry ratchet. It does not yet provide full tier-2 snowplow validation because axial rundown end and radial transit targets for the same PF-1000 shot are still missing.

Ratchet update 2026-05-05, app-level partial PF-1000 phase candidate:

- Modules touched: `app_mhd.py` and `tests/test_mhd_physics_integration.py`.
- App post-processing now recognizes PF-1000 phase histories run near 16 kV and compares them against the partial shot-12581 phase target.
- The result is exported as `snowplow_validation_candidate`, not `snowplow_validation`, because the KR record currently supplies current-dip/pinch timing and pinch duration but not complete same-shot axial/radial phase targets.
- Tests prove the app emits the PF-1000 candidate evidence, keeps `passed=False`, marks pinch timing agreement, and leaves tier 2 at `partial`.
- Verification status: `python3 -m pytest tests/test_mhd_physics_integration.py::test_predictive_readiness_exported_and_blocks_unvalidated_claims tests/test_mhd_physics_integration.py::test_app_uses_pf1000_16kv_partial_phase_target_as_candidate tests/test_mhd_physics_integration.py::test_app_exports_target_backed_snowplow_validation tests/test_kr_targets.py -q` passed; `python3 -m py_compile app_mhd.py tests/test_mhd_physics_integration.py src/dpf/validation/kr_targets.py src/dpf/validation/__init__.py tests/test_kr_targets.py` passed.
- Remaining scientific limit: this makes one partial KR phase target flow through production post-processing. It still cannot certify snowplow validation until the missing axial rundown and radial transit targets are extracted for the same validation scope.

Ratchet update 2026-05-05, strict high-fidelity readiness gate:

- Modules touched: `src/dpf/validation/quality_assessment.py`, `src/dpf/validation/__init__.py`, `app_mhd.py`, `tests/test_quality_assessment.py`, and `tests/test_mhd_physics_integration.py`.
- Added `HighFidelityReadiness` and `high_fidelity_readiness_report()`.
- `predictive_readiness_report()` remains the five-tier evidence gate. The new high-fidelity gate is stricter: it requires predictive readiness plus all `scientific_accuracy_gap_report()` areas to be `supported`.
- App results now export `high_fidelity_readiness` beside `validation_tiers`, `predictive_readiness`, and `scientific_accuracy_gaps`.
- Tests prove that a synthetic result can satisfy the five-tier predictive gate while still failing high-fidelity readiness because missing physics fidelity, circuit-field coupling, UQ, and other scientific-accuracy blockers remain open.
- Verification status: `python3 -m pytest tests/test_quality_assessment.py tests/test_mhd_physics_integration.py::test_predictive_readiness_exported_and_blocks_unvalidated_claims tests/test_mhd_physics_integration.py::test_app_uses_pf1000_16kv_partial_phase_target_as_candidate tests/test_mhd_physics_integration.py::test_app_exports_target_backed_snowplow_validation tests/test_kr_targets.py -q` passed; `python3 -m py_compile src/dpf/validation/quality_assessment.py src/dpf/validation/__init__.py app_mhd.py tests/test_quality_assessment.py tests/test_mhd_physics_integration.py src/dpf/validation/kr_targets.py tests/test_kr_targets.py` passed.
- Remaining scientific limit: this is still a gate, not physics implementation. It prevents overclaiming high-fidelity readiness until the roadmap items are actually closed.

Ratchet update 2026-05-05, high-fidelity physics audit:

- Modules touched: `src/dpf/validation/physics_fidelity.py`, `src/dpf/validation/__init__.py`, `app_mhd.py`, `tests/test_physics_fidelity.py`, and `tests/test_mhd_physics_integration.py`.
- Added `physics_fidelity_evidence_from_result()` to produce a conservative run-level audit for required high-fidelity physics effects.
- The audit covers tabulated EOS/conductivity, ionization and charge-state kinetics, two-temperature energy partition, radiation transport/opacities, ablation/impurity mixing, Hall/FLR/kinetic/PIC effects, 3D instabilities, flashover/sheath initiation, restrike/anomalous resistance, and beam-generation/beam-target coupling.
- App post-processing now exports `physics_fidelity_evidence` before computing `scientific_accuracy_gaps` and `high_fidelity_readiness`.
- Active advanced modules such as FLD radiation, ablation, CR ionization, sheath BCs, or Nernst are marked as implemented or empirical where applicable, but still `validated=False` until KR-backed evidence is attached.
- KnowledgeReference basis: ALEGRA HEDP limitations and required EOS/conductivity/radiation physics from `KnowledgeReference/unlimited-release-printed-september-2009-alegra-hedp-simulations-of-the-dense-plasma-focus.md:287-326` and `332-362`; kinetic transition and neutron mechanism requirements from `KnowledgeReference/neutron-generation-dynamics-inside-a-ma-class-dense-plasma-focus-z-pinch-5.md:174-215` and `405-448`; final-pinch kinetic/instability limits from `KnowledgeReference/the-dense-plasma-focus-a-versatile-dense-pinch-for-diverse-applications.md:184-190`; high-Z/dopant EOS/opacity/material requirements from `KnowledgeReference/seyler-2021-kr-doped-dpf-mhd.md:184-190` and `488-517`.
- Verification status: `python3 -m pytest tests/test_physics_fidelity.py tests/test_quality_assessment.py tests/test_mhd_physics_integration.py::test_predictive_readiness_exported_and_blocks_unvalidated_claims -q` passed; `python3 -m py_compile src/dpf/validation/physics_fidelity.py src/dpf/validation/__init__.py app_mhd.py tests/test_physics_fidelity.py tests/test_mhd_physics_integration.py tests/test_quality_assessment.py` passed.
- Remaining scientific limit: this ratchet exposes missing physics as structured data. It does not implement or validate tabulated EOS, kinetic/PIC closure, multigroup opacity/radiation transport, 3D instabilities, startup flashover, or post-pinch restrike physics.

Ratchet update 2026-05-05, circuit-field coupling audit:

- Modules touched: `src/dpf/validation/circuit_field_coupling.py`, `src/dpf/validation/__init__.py`, `src/dpf/validation/quality_assessment.py`, `app_mhd.py`, `tests/test_circuit_field_coupling.py`, and `tests/test_mhd_physics_integration.py`.
- Added `field_coupling_evidence_from_result()` to export a conservative run-level audit for circuit/MHD coupling.
- The audit checks for plasma inductance series, field-derived inductance, dL/dt or back-EMF, Poynting/interface power balance, circuit energy balance, snowplow-to-MHD handoff metadata, and KR-backed experimental comparison evidence.
- App post-processing now exports `field_coupling_validation` before computing `scientific_accuracy_gaps` and `high_fidelity_readiness`.
- The scientific gap report now marks `circuit_field_coupling` as `partial` when coupling audit evidence or exported inductance/back-EMF signals exist, but it still requires a KR-sourced passing evidence record before the area can be `supported`.
- KnowledgeReference basis: Auluck's circuit-element analysis states that time-varying inductance interpretations create conceptual difficulties and that anomalous impedance is needed to reconcile motional impedance with a Poynting-theorem description (`KnowledgeReference/auluck-2021-dpf-circuit-element.md:35-38`), identifies interface power as `I(t)V(t)` (`KnowledgeReference/auluck-2021-dpf-circuit-element.md:435-455`), and describes terms with no ordinary circuit-theory analog tied to 3D velocity and magnetic-field structure (`KnowledgeReference/auluck-2021-dpf-circuit-element.md:957-991`). Lee/RADPF derives `V=d(LI)/dt=I dL/dt + L dI/dt` and the dynamic-resistance power term from changing inductance (`KnowledgeReference/a-course-on-plasma-focus-numerical-experiments-s-lee-and-s-h-saw-part-1-basic-course.md:12098-12128`).
- Verification status: `python3 -m pytest tests/test_circuit_field_coupling.py tests/test_mhd_physics_integration.py::test_predictive_readiness_exported_and_blocks_unvalidated_claims -q` passed; `python3 -m pytest tests/test_circuit_field_coupling.py tests/test_physics_fidelity.py tests/test_quality_assessment.py tests/test_mhd_physics_integration.py::test_predictive_readiness_exported_and_blocks_unvalidated_claims -q` passed; `python3 -m py_compile src/dpf/validation/circuit_field_coupling.py src/dpf/validation/quality_assessment.py src/dpf/validation/__init__.py app_mhd.py tests/test_circuit_field_coupling.py tests/test_mhd_physics_integration.py` passed.
- Remaining scientific limit: this ratchet does not replace reduced or density-weighted coupling with a validated field-derived model. It makes the current coupling status explicit and keeps MHD current-prediction claims blocked until inductance, dL/dt/back-EMF, Poynting power, energy balance, handoff timing, and KR comparison evidence are validated for the claimed scope.

Ratchet update 2026-05-05, uncertainty-budget audit:

- Modules touched: `src/dpf/validation/uncertainty_budget.py`, `src/dpf/validation/__init__.py`, `src/dpf/validation/quality_assessment.py`, `app_mhd.py`, `tests/test_uncertainty_budget.py`, and `tests/test_mhd_physics_integration.py`.
- Added `uncertainty_evidence_from_result()` to export a conservative run-level audit for high-fidelity uncertainty quantification.
- The audit checks for experimental measurement uncertainty, input/parameter uncertainty, numerical discretization uncertainty, model-form uncertainty, shot-to-shot variability, propagated uncertainty on observables, validation acceptance rules, and same-scope KR uncertainty targets.
- App post-processing now exports `uncertainty_validation` before computing `scientific_accuracy_gaps` and `high_fidelity_readiness`.
- The scientific gap report now marks `uncertainty_quantification` as `partial` when an uncertainty audit exists, but it still requires a KR-sourced passing evidence record before the area can be `supported`.
- KnowledgeReference basis: the plasma-science review defines UQ as uncertainty in model inputs, parameters, structure, and forward propagation to model outputs (`KnowledgeReference/2022-review-of-data-driven-plasma-science.md:1118-1138`) and notes shot-to-shot driver/plasma fluctuations plus diagnostic uncertainty/noise as a control and prediction issue (`KnowledgeReference/2022-review-of-data-driven-plasma-science.md:2580-2618`). DPF-specific sources identify shot-to-shot variation as a known DPF limitation (`KnowledgeReference/paper-open-access-dense-plasma-focus-from-alternative-fusion-source-to-versatile-high-energy.md:399-406`), use error bars and standard deviations for neutron timing/anisotropy comparisons (`KnowledgeReference/neutron-generation-dynamics-inside-a-ma-class-dense-plasma-focus-z-pinch-5.md:565-604`), discuss voltage measurement uncertainty in MHD validation (`KnowledgeReference/beresnyak_2022_pulsed_power_ideal_mhd.md:2689-2695`), and quantify density/profile uncertainty plus shot-to-shot variation in interferometry comparisons (`KnowledgeReference/malir-2024-interferometry-dpf.md:381-390`, `825-831`, and `983-986`).
- Verification status: `python3 -m pytest tests/test_uncertainty_budget.py tests/test_mhd_physics_integration.py::test_predictive_readiness_exported_and_blocks_unvalidated_claims -q` passed; `python3 -m pytest tests/test_uncertainty_budget.py tests/test_circuit_field_coupling.py tests/test_physics_fidelity.py tests/test_quality_assessment.py tests/test_mhd_physics_integration.py::test_predictive_readiness_exported_and_blocks_unvalidated_claims -q` passed; `python3 -m py_compile src/dpf/validation/uncertainty_budget.py src/dpf/validation/quality_assessment.py src/dpf/validation/__init__.py app_mhd.py tests/test_uncertainty_budget.py tests/test_mhd_physics_integration.py` passed.
- Remaining scientific limit: this ratchet does not propagate uncertainty through the solver or validation stack. It makes missing UQ components explicit and keeps high-fidelity predictive claims blocked until experimental, input, numerical, model-form, shot-to-shot, propagated-observable, and acceptance-rule uncertainties are validated for the claimed scope.

Ratchet update 2026-05-05, MHD numerical-fidelity audit:

- Modules touched: `src/dpf/validation/mhd_numerical_fidelity.py`, `src/dpf/validation/__init__.py`, `src/dpf/validation/quality_assessment.py`, `app_mhd.py`, `tests/test_mhd_numerical_fidelity.py`, and `tests/test_mhd_physics_integration.py`.
- Added `mhd_numerical_fidelity_evidence_from_result()` to export a conservative run-level audit for tier-3 numerical fidelity.
- The audit checks for finite-volume MHD verification, cylindrical-geometry verification, circuit-coupled energy verification, resistive/non-ideal verification, convergence studies, backend parity, and explicit DPF phase/scope limits.
- App post-processing now exports `mhd_numerical_fidelity` before computing `scientific_accuracy_gaps` and `high_fidelity_readiness`.
- The scientific gap report now marks `mhd_numerical_fidelity` as `partial` when this audit exists, but it still requires a KR-sourced passing evidence record before the area can be `supported`.
- KnowledgeReference basis: Beresnyak's pulsed-power ideal-MHD paper describes finite-volume MHD with Riemann solvers, HLLD flux, PLM reconstruction, second-order time stepping, Cartesian/cylindrical coordinates, and circuit/MHD unit/voltage feedback (`KnowledgeReference/beresnyak_2022_pulsed_power_ideal_mhd.md:336-356`), describes the cylindrical/circuit boundary and voltage coupling (`KnowledgeReference/beresnyak_2022_pulsed_power_ideal_mhd.md:383-414`), reports cylindrical Mag-Noh-style verification and convergence (`KnowledgeReference/beresnyak_2022_pulsed_power_ideal_mhd.md:1900-1955`), and states that after disruption the ideal-MHD description breaks down (`KnowledgeReference/beresnyak_2022_pulsed_power_ideal_mhd.md:2506-2519` and `2690-2711`). Malir's PF-1000 interferometry paper highlights resistivity-distribution sensitivity in DPF current-density results (`KnowledgeReference/malir-2024-interferometry-dpf.md:511-541` and `912-930`).
- Verification status: `python3 -m pytest tests/test_mhd_numerical_fidelity.py tests/test_mhd_physics_integration.py::test_predictive_readiness_exported_and_blocks_unvalidated_claims -q` passed; `python3 -m pytest tests/test_mhd_numerical_fidelity.py tests/test_uncertainty_budget.py tests/test_circuit_field_coupling.py tests/test_physics_fidelity.py tests/test_quality_assessment.py tests/test_mhd_physics_integration.py::test_predictive_readiness_exported_and_blocks_unvalidated_claims -q` passed; `python3 -m py_compile src/dpf/validation/mhd_numerical_fidelity.py src/dpf/validation/quality_assessment.py src/dpf/validation/__init__.py app_mhd.py tests/test_mhd_numerical_fidelity.py tests/test_mhd_physics_integration.py` passed.
- Remaining scientific limit: this ratchet does not add new MHD solvers, convergence studies, backend comparisons, or validated resistive/circuit-coupled tests. It makes the missing tier-3 numerical evidence explicit and keeps MHD numerical fidelity below high-fidelity support until those checks are actually run and tied to KR-scoped tolerances.

Ratchet update 2026-05-05, PF-1000 interferometry density target:

- Modules touched: `src/dpf/validation/kr_targets.py`, `src/dpf/validation/__init__.py`, `app_mhd.py`, `tests/test_kr_targets.py`, and `tests/test_mhd_physics_integration.py`.
- Added `pf1000_interferometry_density_targets()` from the Malir 2024 PF-1000 Mach-Zehnder interferometry paper.
- Added `pf1000_interferometry_density_evidence_from_profile()` to compare a radial electron-density profile against the KR peak-density and peak-radius targets for shots 13317 and 13328.
- App post-processing now emits this density-profile component when a PF-1000 result carries `density_profile_radius_cm` plus `electron_density_profile_cm3`, or equivalent meter/SI-density keys.
- This evidence is intentionally density-only. It can contribute to a tier-4 `spatial_validation_candidate`, but it cannot promote tier 4 unless same-scope magnetic-field and temperature components are also present.
- KnowledgeReference basis: PF-1000 device and interferometer setup from `KnowledgeReference/malir-2024-interferometry-dpf.md:190-205`; shot context from `208-239`; profile selection at about 1 cm above the anode and 6 mm averaging band from `301-330`; density/radius features from `331-348`; uncertainty from axis/fringe/AIM errors and about 20 percent relative density error away from the axis from `381-397`; comparison limitations and high shot-to-shot variation from `945-990`.
- Verification status: `python3 -m pytest tests/test_kr_targets.py tests/test_mhd_physics_integration.py::test_app_exports_pf1000_interferometry_density_profile_component tests/test_mhd_physics_integration.py::test_app_exports_pf1000_xray_geometry_as_density_component tests/test_mhd_physics_integration.py::test_predictive_readiness_exported_and_blocks_unvalidated_claims -q` passed; `python3 -m pytest tests/test_kr_targets.py tests/test_mhd_numerical_fidelity.py tests/test_uncertainty_budget.py tests/test_circuit_field_coupling.py tests/test_physics_fidelity.py tests/test_quality_assessment.py tests/test_mhd_physics_integration.py::test_predictive_readiness_exported_and_blocks_unvalidated_claims tests/test_mhd_physics_integration.py::test_app_exports_pf1000_interferometry_density_profile_component tests/test_mhd_physics_integration.py::test_app_exports_pf1000_xray_geometry_as_density_component -q` passed; `python3 -m py_compile src/dpf/validation/kr_targets.py src/dpf/validation/__init__.py app_mhd.py tests/test_kr_targets.py tests/test_mhd_physics_integration.py` passed.
- Remaining scientific limit: this ratchet adds a real KR-backed density-profile target, but still does not close tier 4. The project needs same-scope PF-1000 magnetic-field and temperature targets, plus production simulation outputs that export comparable radial profiles with uncertainty, before spatial DPF validation can be supported.

Ratchet update 2026-05-05, tier-4 same-scope closure report:

- Modules touched: `src/dpf/validation/quality_assessment.py`, `src/dpf/validation/__init__.py`, `app_mhd.py`, `tests/test_quality_assessment.py`, and `tests/test_mhd_physics_integration.py`.
- Added `spatial_validation_scope_closure_report()` to group partial tier-4 components by validation scope and report which of density, magnetic field, and temperature remain missing in each scope.
- App post-processing now exports `spatial_validation_scope_closure` whenever it combines `spatial_validation_components`.
- This closes a metadata ambiguity: a candidate may contain density, magnetic-field, and temperature components from different KR scopes, but the closure report now shows that no single scope is complete.
- KnowledgeReference basis: the PF-1000 interferometry paper provides density evolution as the present comparable diagnostic and explicitly identifies temperature and magnetic-field evolution as poorly diagnosed or future work (`KnowledgeReference/malir-2024-interferometry-dpf.md:1003-1018`). The same paper supports density-profile comparison from interferometry, but not same-scope temperature or magnetic-field validation (`KnowledgeReference/malir-2024-interferometry-dpf.md:190-205`, `301-330`, and `945-990`).
- Verification status: `python3 -m pytest tests/test_quality_assessment.py tests/test_mhd_physics_integration.py::test_app_exports_pf1000_interferometry_density_profile_component tests/test_mhd_physics_integration.py::test_app_promotes_complete_same_scope_spatial_components -q` passed; `python3 -m pytest tests/test_kr_targets.py tests/test_mhd_numerical_fidelity.py tests/test_uncertainty_budget.py tests/test_circuit_field_coupling.py tests/test_physics_fidelity.py tests/test_quality_assessment.py tests/test_mhd_physics_integration.py::test_predictive_readiness_exported_and_blocks_unvalidated_claims tests/test_mhd_physics_integration.py::test_app_exports_pf1000_interferometry_density_profile_component tests/test_mhd_physics_integration.py::test_app_exports_pf1000_xray_geometry_as_density_component tests/test_mhd_physics_integration.py::test_app_promotes_complete_same_scope_spatial_components -q` passed; `python3 -m py_compile src/dpf/validation/quality_assessment.py src/dpf/validation/__init__.py app_mhd.py tests/test_quality_assessment.py tests/test_mhd_physics_integration.py` passed.
- Remaining scientific limit: this ratchet does not add magnetic-field or temperature measurements. It makes the same-scope tier-4 gap explicit so PF-1000 density evidence cannot be accidentally combined with unrelated magnetic or temperature evidence to support spatial validation.

Ratchet update 2026-05-05, tier-5 same-scope neutron closure:

- Modules touched: `src/dpf/validation/quality_assessment.py`, `src/dpf/validation/kr_targets.py`, `src/dpf/validation/__init__.py`, `app_mhd.py`, `tests/test_quality_assessment.py`, `tests/test_kr_targets.py`, and `tests/test_mhd_physics_integration.py`.
- Added `neutron_validation_scope_closure_report()` to require neutron mechanism/timing, spectrum, and anisotropy evidence from the same validation scope before tier 5 can be `supported`.
- Updated the MJOLNIR neutron timing, spectrum, and anisotropy helpers so all three evidence records carry `validation_scope="mjolnir_neutron_timing_2025_goyon"`.
- App post-processing now exports `neutron_validation_scope_closure` whenever neutron validation components are present.
- The tier-5 gate now rejects independently sourced timing, spectrum, and anisotropy evidence even when all three components pass individually.
- KnowledgeReference basis: the MJOLNIR target derives timing, mechanism, spectrum, and anisotropy expectations from one source/scope: mechanism discussion from `KnowledgeReference/neutron-generation-dynamics-inside-a-ma-class-dense-plasma-focus-z-pinch-5.md:405-448`, time-of-flight and pulse-shape processing from `474-530`, and spectrum/anisotropy targets from `548-616`.
- Verification status: `python3 -m pytest tests/test_quality_assessment.py tests/test_kr_targets.py tests/test_mhd_physics_integration.py::test_mjolnir_neutron_timing_evidence_is_exported_when_phase_timed tests/test_mhd_physics_integration.py::test_mjolnir_inferred_neutron_timing_remains_candidate_only -q` passed; `python3 -m pytest tests/test_kr_targets.py tests/test_mhd_numerical_fidelity.py tests/test_uncertainty_budget.py tests/test_circuit_field_coupling.py tests/test_physics_fidelity.py tests/test_quality_assessment.py tests/test_mhd_physics_integration.py::test_predictive_readiness_exported_and_blocks_unvalidated_claims tests/test_mhd_physics_integration.py::test_mjolnir_neutron_timing_evidence_is_exported_when_phase_timed tests/test_mhd_physics_integration.py::test_mjolnir_inferred_neutron_timing_remains_candidate_only tests/test_mhd_physics_integration.py::test_app_exports_pf1000_interferometry_density_profile_component tests/test_mhd_physics_integration.py::test_app_exports_pf1000_xray_geometry_as_density_component tests/test_mhd_physics_integration.py::test_app_promotes_complete_same_scope_spatial_components -q` passed; `python3 -m py_compile src/dpf/validation/quality_assessment.py src/dpf/validation/kr_targets.py src/dpf/validation/__init__.py app_mhd.py tests/test_quality_assessment.py tests/test_kr_targets.py tests/test_mhd_physics_integration.py` passed.
- Remaining scientific limit: this ratchet tightens the neutron gate; it does not add new neutron diagnostics. Tier 5 still depends on production outputs with validated mechanism-separated neutron histories, spectra, anisotropy, detector response, and uncertainty for the same KR-backed scope.

Ratchet update 2026-05-05, MJOLNIR detector/activation response audit:

- Modules touched: `src/dpf/validation/kr_targets.py`, `src/dpf/validation/__init__.py`, `src/dpf/validation/quality_assessment.py`, `app_mhd.py`, `tests/test_kr_targets.py`, `tests/test_quality_assessment.py`, and `tests/test_mhd_physics_integration.py`.
- Added `mjolnir_neutron_detector_response_targets()` and `mjolnir_neutron_detector_response_evidence()` for the detector/activation side of MJOLNIR neutron validation.
- The audit requires Be/Y/Br activation channels, Be absolute calibration, LaBr/Y cross-calibration to Be, 10/70 degree anisotropy angles, 45 degree Be reference yield, 2.2 m and 6.6 m scintillator TOF distances, relative timing within 1 ns, propagation broadening, detector temporal response, x-ray peak co-timing, beam-target energy spread, and explicit room-scatter/background assessment.
- App post-processing now emits `neutron_detector_response_validation` for passing MJOLNIR detector-response evidence and `neutron_detector_response_validation_candidate` for incomplete evidence.
- The five-tier predictive gate remains focused on neutron timing, spectrum, and anisotropy. The stricter scientific-accuracy gap report now keeps `neutron_validation` at `partial` unless detector/activation response evidence is KR-sourced and passing.
- KnowledgeReference basis: MJOLNIR diagnostics and activation channels from `KnowledgeReference/neutron-generation-dynamics-inside-a-ma-class-dense-plasma-focus-z-pinch-5.md:132-149`; TOF detector distances and timing calibration from `160-168`; synthetic detector response, x-ray co-timing, temporal response, and unresolved room/equipment scattering from `449-509`; activation anisotropy channels and 10/70/45 degree geometry from `595-607`.
- Verification status: `python3 -m pytest tests/test_kr_targets.py tests/test_quality_assessment.py tests/test_mhd_physics_integration.py::test_app_exports_mjolnir_detector_response_validation tests/test_mhd_physics_integration.py::test_app_keeps_incomplete_mjolnir_detector_response_candidate_only tests/test_mhd_physics_integration.py::test_mjolnir_neutron_timing_evidence_is_exported_when_phase_timed -q` passed (`77 passed in 1.24s`); `python3 -m py_compile src/dpf/validation/kr_targets.py src/dpf/validation/quality_assessment.py src/dpf/validation/__init__.py app_mhd.py tests/test_kr_targets.py tests/test_quality_assessment.py tests/test_mhd_physics_integration.py` passed.
- Remaining scientific limit: this ratchet audits detector-response metadata; it does not generate calibrated detector signals from the production neutron source model. High-fidelity neutron validation still needs production simulation outputs for mechanism-separated birth histories, spectra, anisotropy, detector response, activation response, and uncertainty in the same KR-backed scope.

Ratchet update 2026-05-05, PF-1000 16 kV derived-output candidate:

- Modules touched: `src/dpf/validation/kr_targets.py`, `src/dpf/validation/__init__.py`, `app_mhd.py`, `tests/test_kr_targets.py`, and `tests/test_mhd_physics_integration.py`.
- Added `pf1000_16kv_derived_output_candidate_evidence()` to compare production observables against the PF-1000 16 kV Lee-fitted outputs already extracted from the Akel 2021 KR record.
- The candidate checks peak current, pinch current, axial speed, radial shock speed, radial piston speed, final pinch radius, pinch length, and maximum induced voltage against the KR target values.
- App post-processing now exports `snowplow_dynamics_validation_candidate` for matching PF-1000 16 kV phase histories when direct or derivable observables are present.
- This remains `passed=False` by design. It audits Lee-derived dynamics but cannot support tier 2 until the same validation scope has complete measured axial rundown, radial transit, and pinch timing targets.
- KnowledgeReference basis: PF-1000 shot/context and Lee fitting procedure from `KnowledgeReference/radiation-physics-and-chemistry-188-2021-109633.md:111-137` and `219-235`; shot-12581 fitted context and outputs from `250-285`; table context from `332-346`.
- Verification status: `python3 -m pytest tests/test_kr_targets.py tests/test_mhd_physics_integration.py::test_app_uses_pf1000_16kv_partial_phase_target_as_candidate tests/test_mhd_physics_integration.py::test_app_keeps_targetless_snowplow_phase_history_as_candidate tests/test_quality_assessment.py::TestQualityAssessment::test_scientific_accuracy_gap_report_lists_remaining_work -q` passed (`36 passed in 0.62s`); `python3 -m py_compile src/dpf/validation/kr_targets.py src/dpf/validation/__init__.py app_mhd.py tests/test_kr_targets.py tests/test_mhd_physics_integration.py` passed.
- Remaining scientific limit: the KR corpus still has not yielded complete same-shot PF-1000 axial/radial/pinch timing targets. This ratchet makes additional PF-1000 dynamic comparisons visible, but tier 2 remains partial.

Ratchet update 2026-05-05, validation-observable uncertainty coverage:

- Modules touched: `src/dpf/validation/uncertainty_budget.py`, `src/dpf/validation/__init__.py`, `app_mhd.py`, `tests/test_uncertainty_budget.py`, and `tests/test_mhd_physics_integration.py`.
- Added `validation_uncertainty_coverage_from_result()` to enumerate every present validation evidence record and report whether it carries uncertainty metadata.
- The coverage report includes circuit, snowplow phase, PF-1000 snowplow dynamics candidates, MHD verification, MHD numerical fidelity, spatial validation/candidates/components, neutron timing, spectrum, anisotropy, and detector-response evidence.
- App post-processing now exports `validation_uncertainty_coverage` before `uncertainty_validation`, so the uncertainty audit can distinguish generic UQ output from observable-level uncertainty coverage.
- `uncertainty_evidence_from_result()` now marks observable-level coverage separately as `observable_coverage_present`, but still keeps the full UQ evidence `passed=False` unless all required uncertainty components and KR uncertainty targets are validated.
- KnowledgeReference basis: UQ propagation from inputs/model structure to outputs from `KnowledgeReference/2022-review-of-data-driven-plasma-science.md:1118-1138`; data uncertainty reporting expectations from `6889-6892`; DPF shot-to-shot and diagnostic uncertainty from the already-cited PF-1000, MJOLNIR, Malir, and Beresnyak KR lines in the uncertainty-budget audit.
- Verification status: `python3 -m pytest tests/test_uncertainty_budget.py tests/test_mhd_physics_integration.py::test_predictive_readiness_exported_and_blocks_unvalidated_claims -q` passed (`8 passed in 0.69s`); `python3 -m py_compile src/dpf/validation/uncertainty_budget.py src/dpf/validation/__init__.py app_mhd.py tests/test_uncertainty_budget.py tests/test_mhd_physics_integration.py` passed.
- Remaining scientific limit: this ratchet identifies which validation observables still lack uncertainty metadata. It does not perform Monte Carlo, Bayesian, multifidelity, or shot-statistical propagation through the solver, and it does not create same-scope KR uncertainty targets.

Ratchet update 2026-05-05, Lee dynamic-inductance power accounting:

- Modules touched: `src/dpf/validation/circuit_field_coupling.py`, `src/dpf/validation/__init__.py`, `app_mhd.py`, `tests/test_circuit_field_coupling.py`, and `tests/test_mhd_physics_integration.py`.
- Added `dynamic_inductance_power_balance_from_waveforms()` to compute the Lee dynamic-inductance power partition from exported time, current, and plasma-inductance waveforms.
- The diagnostic computes `V=d(LI)/dt`, interface power `VI`, magnetic-energy derivative, dynamic-resistance power, and the residual of the identity `VI = d(0.5LI^2)/dt + 0.5 I^2 dL/dt`.
- App post-processing now exports `dynamic_inductance_power_balance` when `t_us`, `I_MA`, and a plasma-inductance series are available, before the circuit/field coupling audit runs.
- The circuit-field coupling audit now treats this diagnostic as a circuit-energy-balance channel, but it remains `diagnostic_not_validated` and does not support field-coupling validation without KR-backed current/voltage or Poynting comparison evidence.
- KnowledgeReference basis: Lee dynamic inductance and power partition from `KnowledgeReference/a-course-on-plasma-focus-numerical-experiments-s-lee-and-s-h-saw-part-1-basic-course.md:12103-12127`; Auluck's warning that time-varying inductance does not by itself close the Poynting-theorem coupling description from `KnowledgeReference/auluck-2021-dpf-circuit-element.md:35-39` and `1027-1031`.
- Verification status: `python3 -m pytest tests/test_circuit_field_coupling.py tests/test_mhd_physics_integration.py::test_predictive_readiness_exported_and_blocks_unvalidated_claims -q` passed (`7 passed in 0.67s`); `python3 -m py_compile src/dpf/validation/circuit_field_coupling.py src/dpf/validation/__init__.py app_mhd.py tests/test_circuit_field_coupling.py tests/test_mhd_physics_integration.py` passed.
- Remaining scientific limit: this is an internal reduced-model accounting diagnostic. It does not prove first-principles circuit/MHD coupling, Poynting flux through the plasma boundary, 3D post-stagnation magnetic-field effects, or experimental voltage/current agreement.

Ratchet update 2026-05-05, MHD numerical-method metadata export:

- Modules touched: `app_mhd.py`, `src/dpf/validation/mhd_numerical_fidelity.py`, `tests/test_mhd_numerical_fidelity.py`, and `tests/test_mhd_physics_integration.py`.
- App results now export `mhd_numerical_method` with backend, finite-volume flag, coordinates, grid shape, grid spacing, reconstruction, Riemann solver, time integrator, and precision.
- The MHD numerical-fidelity audit now distinguishes method metadata from verification evidence. A finite-volume PLM/HLL/HLLD method declaration is reported as `method_metadata_only`, not as validated numerical fidelity.
- Cylindrical-coordinate metadata now contributes to the cylindrical-geometry evidence channel, but remains `diagnostic_not_validated` until analytic cylindrical verification and convergence evidence are attached.
- KnowledgeReference basis: Beresnyak's pulsed-power MHD paper identifies finite-volume MHD, Riemann solvers, HLLD flux, PLM reconstruction, second-order time stepping, Cartesian/cylindrical coordinates, SI unit exchange, and circuit feedback as required numerical-method context (`KnowledgeReference/beresnyak_2022_pulsed_power_ideal_mhd.md:341-354`, `383-414`), and separately reports cylindrical convergence evidence (`1900-1955`).
- Verification status: `python3 -m pytest tests/test_mhd_numerical_fidelity.py tests/test_mhd_physics_integration.py::test_predictive_readiness_exported_and_blocks_unvalidated_claims -q` passed (`6 passed in 0.68s`); `python3 -m py_compile src/dpf/validation/mhd_numerical_fidelity.py app_mhd.py tests/test_mhd_numerical_fidelity.py tests/test_mhd_physics_integration.py` passed.
- Remaining scientific limit: this ratchet makes the numerical method auditable in result metadata. It still does not run or validate cylindrical Mag-Noh convergence, circuit-coupled energy verification, resistive/non-ideal verification, backend parity, or DPF phase-scope applicability.

Ratchet update 2026-05-05, cylindrical z-pinch convergence evidence:

- Modules touched: `src/dpf/validation/mhd_numerical_fidelity.py`, `src/dpf/validation/__init__.py`, and `tests/test_mhd_numerical_fidelity.py`.
- Added `cylindrical_convergence_evidence_from_results()` to turn local cylindrical z-pinch convergence output into explicit tier-3 code-verification evidence.
- The evidence requires at least three resolutions, finite positive `Btheta_errors`, strictly decreasing `Btheta_errors`, and measured convergence order at or above the KR-supported first-order threshold.
- The MHD numerical-fidelity audit now recognizes only this KR-scoped evidence role as validated cylindrical/convergence support. Arbitrary `cylindrical_verification`, backend labels, method metadata, or generic `grid_convergence` dictionaries still remain `diagnostic_not_validated`.
- A passing cylindrical convergence record now marks only `cylindrical_geometry_verification` and `convergence_study` as `supported`; the full MHD numerical-fidelity audit still fails until finite-volume analytic tests, circuit-coupled energy verification, resistive/non-ideal verification, backend parity, and DPF phase-scope limits are also validated.
- KnowledgeReference basis: Beresnyak's pulsed-power MHD paper describes cylindrical self-similar MHD verification and reports numerical convergence slightly better than first order (`KnowledgeReference/beresnyak_2022_pulsed_power_ideal_mhd.md:1900-1955`). The Bennett/vorticity KR reference supplies the z-pinch force-balance basis: axial current, azimuthal magnetic field, radial pressure gradient, and analytic pressure profile (`KnowledgeReference/bennett-vorticity-analytic-solutions-to-a-flowing-nonlinear-shear-flow-stabilized-z-pinch.md:285-299`, `386-397`, and `452-485`).
- Verification status: `python3 -m pytest tests/test_mhd_numerical_fidelity.py -q` passed (`8 passed in 0.70s`); `python3 -m py_compile src/dpf/validation/mhd_numerical_fidelity.py src/dpf/validation/__init__.py tests/test_mhd_numerical_fidelity.py` passed.
- Remaining scientific limit: this ratchet adds an evidence adapter and audit recognition, not a production convergence campaign. It does not run the cylindrical solver across production DPF regimes, reproduce Beresnyak's Mag-Noh boundary problem, validate circuit-coupled Poynting flux, validate non-ideal terms, compare backends, or prove DPF-observable convergence.

Ratchet update 2026-05-05, resistive magnetic-diffusion verification evidence:

- Modules touched: `src/dpf/validation/mhd_numerical_fidelity.py`, `src/dpf/validation/__init__.py`, and `tests/test_mhd_numerical_fidelity.py`.
- Added `resistive_diffusion_convergence_evidence_from_results()` to turn local magnetic-diffusion convergence output into explicit tier-3 code-verification evidence.
- The evidence requires a recognized local diffusion method (`explicit`, `implicit`, or `sts`), at least three resolutions, finite positive errors, strictly decreasing errors, positive resistivity, and convergence order at or above the conservative first-order threshold.
- The MHD numerical-fidelity audit now recognizes only this KR-scoped evidence role as support for `resistive_or_nonideal_verification`. Generic `resistivity`, `eta`, `ohmic_heating`, `R_anom`, or diffusion dictionaries remain `implemented_not_validated`.
- A passing diffusion record supports only the resistive magnetic-diffusion operator. It does not mark the full convergence-study channel as validated, because the KR gap is still convergence of claimed DPF observables and production backends.
- KnowledgeReference basis: the modeling text defines generalized Ohm's law resistivity and the magnetic-field evolution equation's resistive diffusion operator (`KnowledgeReference/modeling-and-simulation-in-science-engineering-and-technology-mathematical-models-and.md:1288-1295` and `1341-1358`). Malir's PF-1000 paper includes resistive terms in the generalized Ohm law and a Spitzer resistivity choice (`KnowledgeReference/malir-2024-interferometry-dpf.md:511-545`), then warns that current-region differences are probably caused by the resistivity distribution and 1D limitations (`908-930`). The gas-puff z-pinch KR record shows sheath structure can depend on the chosen anomalous-resistivity model (`KnowledgeReference/the-hall-term-and-anomalous-resistivity-effects-in-neon-gas-puff-z-pinches.md:17-38` and `402-410`).
- Verification status: `python3 -m pytest tests/test_mhd_numerical_fidelity.py -q` passed (`11 passed in 0.45s`); `python3 -m py_compile src/dpf/validation/mhd_numerical_fidelity.py src/dpf/validation/__init__.py tests/test_mhd_numerical_fidelity.py` passed.
- Remaining scientific limit: this ratchet verifies an operator-level numerical convergence record if supplied. It does not validate the Spitzer closure, anomalous/LHDI resistivity, Hall terms, low-density-floor behavior, 2D/3D current redistribution, or PF-1000/MJOLNIR current-density profiles.

Ratchet update 2026-05-05, circuit-coupled Poynting/energy evidence:

- Modules touched: `src/dpf/validation/circuit_field_coupling.py`, `src/dpf/validation/mhd_numerical_fidelity.py`, `src/dpf/validation/__init__.py`, `tests/test_circuit_field_coupling.py`, and `tests/test_mhd_numerical_fidelity.py`.
- Added `circuit_coupled_energy_evidence_from_history()` to verify, for a supplied time history, that `voltage * current` matches a field/interface Poynting-power series and that integrated interface energy is accounted for by stored plus dissipated energy.
- The field-coupling audit now recognizes only this KR-scoped role as support for the `poynting_power_balance` and `circuit_energy_balance` channels. Plain Poynting dictionaries, energy arrays, or Lee dynamic-inductance identity checks remain diagnostics.
- The MHD numerical-fidelity audit now recognizes the same evidence as support for `circuit_coupled_energy_verification`, while the overall audit still fails until finite-volume, cylindrical, resistive, convergence, backend-parity, and scope-limit evidence are complete.
- KnowledgeReference basis: Beresnyak describes the MHD boundary voltage being determined from the `-v x B` electric field, subsequent circuit solving from MHD dynamics, and simultaneous MHD/circuit integration (`KnowledgeReference/beresnyak_2022_pulsed_power_ideal_mhd.md:383-414`). Auluck states that a Poynting-theorem field view exposes missing motional-impedance terms when plasma inductance is inferred only from magnetic energy (`KnowledgeReference/auluck-2021-dpf-circuit-element.md:1026-1031`).
- Verification status: `python3 -m pytest tests/test_circuit_field_coupling.py tests/test_mhd_numerical_fidelity.py -q` passed (`21 passed in 0.45s`); `python3 -m py_compile src/dpf/validation/circuit_field_coupling.py src/dpf/validation/mhd_numerical_fidelity.py src/dpf/validation/__init__.py tests/test_circuit_field_coupling.py tests/test_mhd_numerical_fidelity.py` passed.
- Remaining scientific limit: this ratchet validates only consistency of a supplied power/energy history. It does not generate field-derived Poynting flux from the production solver, validate the electrode/vacuum boundary, compare current/voltage to experiment, or resolve late-pinch 3D magnetic activity.

Ratchet update 2026-05-05, backend parity evidence:

- Modules touched: `src/dpf/validation/mhd_numerical_fidelity.py`, `src/dpf/validation/__init__.py`, and `tests/test_mhd_numerical_fidelity.py`.
- Added `backend_parity_evidence_from_results()` to compare per-backend observables against a reference backend with explicit relative tolerances.
- The evidence requires at least two backends, common or required finite observables, and all relative errors within the configured tolerance.
- The MHD numerical-fidelity audit now recognizes only this KR-scoped parity evidence role as support for the `backend_parity` channel. Bare `backend_parity`, `backend_comparison`, or `backend_validation` dictionaries remain `diagnostic_not_validated`.
- KnowledgeReference basis: Beresnyak reports cylindrical MHD verification against theoretical solutions and notes those solutions have been used to test stability and accuracy of multiple codes such as Athena, Mach2, and Flash (`KnowledgeReference/beresnyak_2022_pulsed_power_ideal_mhd.md:1900-1903` and `1939-1955`). This supports a code-verification requirement for cross-backend agreement, not an experimental validation claim.
- Verification status: `python3 -m pytest tests/test_mhd_numerical_fidelity.py -q` passed (`15 passed in 0.44s`); `python3 -m py_compile src/dpf/validation/mhd_numerical_fidelity.py src/dpf/validation/__init__.py tests/test_mhd_numerical_fidelity.py` passed.
- Remaining scientific limit: this ratchet can validate numerical parity only for supplied observables and tolerances. It does not validate the observables against KR experiments, prove convergence inside each backend, cover GPU/Metal runtime differences automatically, or guarantee parity for all DPF phases.

Ratchet update 2026-05-05, finite-volume MHD verification channel:

- Modules touched: `src/dpf/validation/mhd_numerical_fidelity.py` and `tests/test_mhd_numerical_fidelity.py`.
- The MHD numerical-fidelity audit now marks `finite_volume_mhd_verification` as `supported` only when finite-volume method metadata is present and tier-3 code-verification evidence passes the required Sod and Brio-Wu analytic tests.
- Method metadata alone remains `method_metadata_only`. Generic MHD verification without finite-volume method metadata remains `implemented_not_complete`.
- This closes only the generic finite-volume MHD verification channel. The audit still requires separate cylindrical, circuit-coupled, resistive/non-ideal, convergence, backend-parity, and DPF scope-limit evidence before numerical fidelity can pass.
- KnowledgeReference basis: Beresnyak's pulsed-power MHD paper identifies a finite-volume MHD code with Riemann solvers, HLLD flux, PLM reconstruction, second-order time stepping, Cartesian/cylindrical coordinates, and circuit feedback as the numerical context for its verification work (`KnowledgeReference/beresnyak_2022_pulsed_power_ideal_mhd.md:341-354` and `383-414`).
- Verification status: `python3 -m pytest tests/test_mhd_numerical_fidelity.py -q` passed (`16 passed in 0.45s`); `python3 -m py_compile src/dpf/validation/mhd_numerical_fidelity.py tests/test_mhd_numerical_fidelity.py` passed.
- Remaining scientific limit: Sod and Brio-Wu are generic MHD code checks. They do not validate cylindrical DPF source terms, electrode/vacuum boundary coupling, resistivity choices, backend parity, or experimental DPF observables.

Ratchet update 2026-05-05, MHD phase/scope-limit evidence:

- Modules touched: `src/dpf/validation/mhd_numerical_fidelity.py`, `src/dpf/validation/__init__.py`, and `tests/test_mhd_numerical_fidelity.py`.
- Added `mhd_scope_limit_evidence_from_phases()` to require explicit MHD applicability and invalidity phases before `dpf_scope_limit` can be supported.
- The evidence requires a pre-disruption or first-collapse applicability phase, an excluded post-collapse/post-disruption phase, and a stated limitation reason such as instability, non-ideal fields, disruption, or beyond-ideal-MHD effects.
- The MHD numerical-fidelity audit now treats arbitrary `plasma_regime`, `validity_notes`, or `physics_fidelity_evidence` as `scope_limiter_reported` only. A passing KR-scoped phase-limit evidence record is required for `supported`.
- KnowledgeReference basis: Beresnyak says the MHD description may apply during plasma column/z-pinch formation before disruption, and that after disruption non-ideal fields break MHD applicability (`KnowledgeReference/beresnyak_2022_pulsed_power_ideal_mhd.md:2506-2519`). The same paper reports that dynamics before and during first collapse are reasonably described by ideal MHD, while after first collapse they are not well described because of Rayleigh-Taylor sensitivity and effects beyond ideal MHD (`2689-2711`).
- Verification status: `python3 -m pytest tests/test_mhd_numerical_fidelity.py -q` passed (`19 passed in 0.75s`); `python3 -m py_compile src/dpf/validation/mhd_numerical_fidelity.py src/dpf/validation/__init__.py tests/test_mhd_numerical_fidelity.py` passed.
- Remaining scientific limit: this ratchet bounds claims; it does not validate the solver within the bounded phase. It also does not implement post-disruption kinetic, beam, anomalous-resistivity, or 3D instability physics.

Ratchet update 2026-05-05, MHD numerical-fidelity closure path:

- Module touched: `tests/test_mhd_numerical_fidelity.py`.
- Added a regression that assembles the full MHD numerical-fidelity evidence packet: finite-volume Sod/Brio-Wu verification with method metadata, cylindrical z-pinch convergence, circuit-coupled Poynting/energy evidence, resistive diffusion convergence, backend parity, and phase/scope-limit evidence.
- The audit now has an explicit passing path only when all required evidence channels are validated. The scientific-accuracy gap report promotes `mhd_numerical_fidelity` to `supported` only for that complete packet.
- Verification status: `python3 -m pytest tests/test_mhd_numerical_fidelity.py tests/test_quality_assessment.py::TestQualityAssessment::test_scientific_accuracy_gap_report_lists_remaining_work -q` passed (`21 passed in 0.55s`); `python3 -m py_compile tests/test_mhd_numerical_fidelity.py` passed.
- Remaining scientific limit: this is a synthetic evidence-packet regression, not a production run. The app still has to generate or attach these evidence records from actual solver outputs for a claimed device/phase before MHD numerical fidelity is supported in real results.

Ratchet update 2026-05-05, high-fidelity physics-effect evidence path:

- Modules touched: `src/dpf/validation/physics_fidelity.py`, `src/dpf/validation/__init__.py`, and `tests/test_physics_fidelity.py`.
- Added `physics_effect_validation_evidence()` to create line-referenced evidence for one required high-fidelity physics effect.
- The physics-fidelity audit now accepts only passing `physics_effect_validation` / `physics_effect_validations` evidence, or effect-specific `*_validation` evidence, when the effect name is known, the source is in `KnowledgeReference/`, a validation scope is stated, and the effect is either implemented/validated or explicitly bounded out of the claim scope.
- Added a complete synthetic evidence-packet regression showing `physics_fidelity_evidence_from_result()` can pass and the scientific-accuracy gap can mark `missing_physics_fidelity` as `supported`, but only when all required effects have KR-backed evidence.
- KnowledgeReference basis: each effect keeps the existing per-effect source lines already used by the audit: HEDP EOS/transport/ionization/two-temperature/restrike blockers from the ALEGRA DPF KR source, radiation/opacity and impurity/mixing requirements from the Seyler krypton-doped DPF MHD source, kinetic/beam-target limits from the MJOLNIR neutron dynamics source, and 3D/startup limits from Krishnan's DPF review.
- Verification status: `python3 -m pytest tests/test_physics_fidelity.py -q` passed (`6 passed in 0.69s`); `python3 -m py_compile src/dpf/validation/physics_fidelity.py src/dpf/validation/__init__.py tests/test_physics_fidelity.py` passed.
- Remaining scientific limit: this ratchet creates the evidence path; it does not supply real EOS, ionization, two-temperature, radiation, impurity, kinetic/PIC, 3D, startup, restrike, or beam-target validation evidence from production runs.

Ratchet update 2026-05-05, uncertainty-budget component evidence path:

- Modules touched: `src/dpf/validation/uncertainty_budget.py`, `src/dpf/validation/__init__.py`, and `tests/test_uncertainty_budget.py`.
- Added `uncertainty_component_evidence()` to create line-referenced evidence for one required uncertainty-budget component.
- The uncertainty audit now accepts only passing `uncertainty_component_validation` / `uncertainty_component_validations` evidence, or component-specific `*_validation` evidence, when the component name is known, the source is in `KnowledgeReference/`, and a validation scope is stated.
- Added a complete synthetic uncertainty packet regression showing `uncertainty_evidence_from_result()` can pass and the scientific-accuracy gap can mark `uncertainty_quantification` as `supported`, but only when all required uncertainty components have KR-backed evidence.
- KnowledgeReference basis: each component keeps the existing per-component source lines already used by the audit: plasma UQ propagation requirements from the 2022 data-driven plasma science review, DPF shot variability from the open-access DPF review, MJOLNIR error-bar/standard-deviation handling, Beresnyak voltage uncertainty, and Malir PF-1000 density uncertainty.
- Verification status: `python3 -m pytest tests/test_uncertainty_budget.py -q` passed (`9 passed in 0.46s`); `python3 -m py_compile src/dpf/validation/uncertainty_budget.py src/dpf/validation/__init__.py tests/test_uncertainty_budget.py` passed.
- Remaining scientific limit: this creates the UQ evidence path; it does not run Monte Carlo, Bayesian, polynomial-chaos, ensemble, discretization, model-form, shot-statistical, or acceptance-rule propagation for a production DPF result.

Ratchet update 2026-05-05, field-coupling component evidence path:

- Modules touched: `src/dpf/validation/circuit_field_coupling.py`, `src/dpf/validation/__init__.py`, and `tests/test_circuit_field_coupling.py`.
- Added `field_coupling_component_evidence()` to create line-referenced evidence for one field-coupling audit component.
- The field-coupling audit now accepts only passing `field_coupling_component_validation` / `field_coupling_component_validations` evidence, or component-specific `*_validation` evidence, when the component name is known, the source is in `KnowledgeReference/`, and a validation scope is stated.
- Added a complete synthetic field-coupling packet regression showing `field_coupling_evidence_from_result()` can pass only when plasma inductance, field-derived inductance, dL/dt/back-EMF, Poynting power, circuit energy, handoff metadata, and KR experimental comparison components all have KR-backed evidence.
- KnowledgeReference basis: the component source lines remain those already used by the audit: Lee/Saw dynamic inductance and dynamic-resistance accounting, Auluck's Poynting-theorem circuit-element critique, and the KR requirement to compare coupling signals against field/circuit targets.
- Verification status: `python3 -m pytest tests/test_circuit_field_coupling.py -q` passed (`11 passed in 0.43s`); `python3 -m py_compile src/dpf/validation/circuit_field_coupling.py tests/test_circuit_field_coupling.py` passed.
- Remaining scientific limit: this creates the field-coupling evidence path; it does not produce field-derived inductance, back-EMF, Poynting flux, handoff timing, or KR experimental coupling comparisons from production solver output.

Ratchet update 2026-05-05, result-level source-authority evidence:

- Modules touched: `src/dpf/validation/quality_assessment.py`, `src/dpf/validation/__init__.py`, and `tests/test_quality_assessment.py`.
- Added `source_authority_evidence()` to create result-level KR source-authority evidence for a stated validation scope.
- The scientific-accuracy gap report now marks `source_authority_data` as `supported` when the result carries passing `source_authority_validation` evidence with KnowledgeReference sources and line ranges. Without that result-level evidence, the report keeps the previous registry-level behavior, where the gap is only partial until all registered devices have KR-verified measured waveform authority.
- Verification status: `python3 -m pytest tests/test_quality_assessment.py::TestQualityAssessment::test_scientific_accuracy_gap_report_lists_remaining_work tests/test_quality_assessment.py::TestQualityAssessment::test_result_level_source_authority_evidence_can_support_gap -q` passed (`2 passed in 0.58s`); `python3 -m py_compile src/dpf/validation/quality_assessment.py src/dpf/validation/__init__.py tests/test_quality_assessment.py` passed.
- Remaining scientific limit: this supports source authority for a scoped result only. It does not make POSEIDON-60kV, UNU-ICTP, or any other registry device validation-ready until their actual KR waveform/diagnostic authority is extracted.

Ratchet update 2026-05-05, high-fidelity readiness closure path:

- Module touched: `tests/test_quality_assessment.py`.
- Added an end-to-end readiness regression that combines supported predictive tiers, result-level source authority, passed MHD numerical fidelity, passed field-coupling validation, passed physics-fidelity evidence, passed uncertainty-budget evidence, neutron detector-response evidence, and export hygiene.
- The high-fidelity readiness gate now has an explicit passing path and still fails by default when any scientific-accuracy gap remains open.
- Verification status: `python3 -m pytest tests/test_quality_assessment.py::TestQualityAssessment::test_high_fidelity_readiness_can_pass_with_complete_evidence_packet -q` passed (`1 passed in 0.50s`); `python3 -m py_compile tests/test_quality_assessment.py` passed.
- Remaining scientific limit: this is a synthetic closure regression. It proves the gate semantics, not that production DPF runs currently generate the required KR-backed evidence packets.

Ratchet update 2026-05-05, app-level MHD scope limiter:

- Modules touched: `app_mhd.py` and `tests/test_mhd_physics_integration.py`.
- App post-processing now exports `mhd_scope_limit` for MHD results before building `mhd_numerical_fidelity`.
- The evidence states the KR-supported ideal-MHD applicability boundary: formation/first collapse are in scope, while post-first-collapse/post-disruption behavior is out of ideal-MHD scope because of Rayleigh-Taylor sensitivity and non-ideal electric fields.
- The MHD numerical-fidelity audit now marks the app-generated `dpf_scope_limit` channel as `supported` while keeping the overall numerical-fidelity audit blocked until the other channels are supplied by real evidence.
- KnowledgeReference basis: `KnowledgeReference/beresnyak_2022_pulsed_power_ideal_mhd.md:2506-2519` and `2689-2711`.
- Verification status: `python3 -m pytest tests/test_mhd_physics_integration.py::test_predictive_readiness_exported_and_blocks_unvalidated_claims -q` passed (`1 passed in 0.73s`); `python3 -m py_compile app_mhd.py tests/test_mhd_physics_integration.py` passed.
- Remaining scientific limit: this is a claim-boundary evidence record. It does not validate MHD dynamics, late-pinch disruption, kinetic beams, or post-collapse neutron production.

Ratchet update 2026-05-05, result-derived source-authority audit:

- Modules touched: `src/dpf/validation/quality_assessment.py`, `src/dpf/validation/__init__.py`, `app_mhd.py`, and `tests/test_quality_assessment.py`.
- Added `source_authority_evidence_from_result()` to audit the passed validation records already attached to a result. It accepts only KnowledgeReference paths plus explicit line ranges, including nested `required_evidence`, `required_effects`, and `required_components` records used by the MHD numerical-fidelity, field-coupling, physics-fidelity, and uncertainty-budget audits.
- App post-processing now exports `source_authority_validation` automatically when the caller has not supplied one, before computing validation tiers, scientific-accuracy gaps, and high-fidelity readiness.
- This closes a metadata gap: production results can no longer rely only on a hand-built synthetic source-authority packet. The result must carry traceable KR line authority for each passed validation claim it wants covered.
- KnowledgeReference rule enforced: any validation record without a `KnowledgeReference/` source and line range is listed under `missing_source_authority`; failed or candidate evidence is not promoted into the source-authority claim.
- Verification status: `python3 -m py_compile src/dpf/validation/quality_assessment.py src/dpf/validation/__init__.py app_mhd.py tests/test_quality_assessment.py` passed; `python3 -m pytest tests/test_quality_assessment.py::TestQualityAssessment::test_source_authority_evidence_from_result_collects_kr_lines tests/test_quality_assessment.py::TestQualityAssessment::test_source_authority_evidence_from_result_rejects_unlined_claim tests/test_quality_assessment.py::TestQualityAssessment::test_result_level_source_authority_evidence_can_support_gap -q` passed (`3 passed in 0.84s`); app/readiness smoke slice passed (`3 passed in 0.74s`); `python3 -m pytest tests/test_quality_assessment.py tests/test_kr_targets.py -q` passed (`80 passed in 0.65s`); `python3 -m pytest tests/test_mhd_physics_integration.py tests/test_mhd_numerical_fidelity.py tests/test_circuit_field_coupling.py tests/test_physics_fidelity.py tests/test_uncertainty_budget.py -q` passed (`87 passed, 3 skipped in 1.50s`); `git diff --check` passed.
- Remaining scientific limit: this is provenance enforcement, not new physics or new experimental evidence. Most ordinary production runs will still export a failing source-authority audit until their actual passed validation records carry KR line ranges.

Ratchet update 2026-05-05, failed source-authority blocks the source gap:

- Modules touched: `src/dpf/validation/quality_assessment.py` and `tests/test_quality_assessment.py`.
- `scientific_accuracy_gap_report()` now treats an explicit failing `source_authority_validation` record as a run-level blocker. It no longer lets a result with failed line authority fall back to the registry-wide partial status from `get_validation_ready_devices()`.
- The source-authority gap blocker now names the evidence keys missing KR line authority, for example `neutron_spectrum_validation`, so downstream UI/API layers can report the exact provenance failure.
- KnowledgeReference rule enforced: if a run claims passed validation evidence, the result-level source audit must prove that claim from local KR files and line ranges. Registry readiness is only a fallback when no result-level source audit exists.
- Verification status: `python3 -m py_compile src/dpf/validation/quality_assessment.py tests/test_quality_assessment.py` passed; `python3 -m pytest tests/test_quality_assessment.py::TestQualityAssessment::test_gap_report_uses_failed_result_source_authority_as_blocker tests/test_quality_assessment.py::TestQualityAssessment::test_scientific_accuracy_gap_report_lists_remaining_work tests/test_mhd_physics_integration.py::test_predictive_readiness_exported_and_blocks_unvalidated_claims -q` passed (`3 passed in 1.08s`); `python3 -m pytest tests/test_quality_assessment.py tests/test_kr_targets.py -q` passed (`81 passed in 0.60s`); `python3 -m pytest tests/test_mhd_physics_integration.py tests/test_mhd_numerical_fidelity.py tests/test_circuit_field_coupling.py tests/test_physics_fidelity.py tests/test_uncertainty_budget.py -q` passed (`87 passed, 3 skipped in 1.30s`); `git diff --check` passed.
- Remaining scientific limit: this tightens claim hygiene. It still does not generate missing KR waveform, phase, spatial, neutron, physics, field-coupling, or UQ evidence for production simulations.

Ratchet update 2026-05-05, manual source-authority packets are cross-checked:

- Modules touched: `src/dpf/validation/quality_assessment.py` and `tests/test_quality_assessment.py`.
- A manually supplied passing `source_authority_validation` packet no longer overrides missing KR line authority on other passed validation claims. When passed validation evidence is present, `scientific_accuracy_gap_report()` derives its own source-authority audit and blocks the source gap if any passed claim is unlined.
- Standalone source-authority packets are still accepted when no other passed validation evidence is present, preserving the scoped helper use case while preventing high-fidelity closure from a one-line provenance placeholder.
- The synthetic high-fidelity closure regression now carries source lines on every passed validation claim. This keeps the passing path but makes it represent the stricter provenance contract.
- KnowledgeReference rule enforced: the source-authority gap is supported only when the result-level packet and the passed evidence records agree that all claimed evidence is locally KR-sourced and line-referenced.
- Verification status: `python3 -m py_compile src/dpf/validation/quality_assessment.py tests/test_quality_assessment.py` passed; `python3 -m pytest tests/test_quality_assessment.py::TestQualityAssessment::test_gap_report_cross_checks_manual_source_authority_packet tests/test_quality_assessment.py::TestQualityAssessment::test_gap_report_uses_failed_result_source_authority_as_blocker tests/test_quality_assessment.py::TestQualityAssessment::test_high_fidelity_readiness_can_pass_with_complete_evidence_packet -q` passed (`3 passed in 0.72s`); `python3 -m pytest tests/test_quality_assessment.py tests/test_kr_targets.py -q` passed (`82 passed in 0.49s`); `python3 -m pytest tests/test_mhd_physics_integration.py tests/test_mhd_numerical_fidelity.py tests/test_circuit_field_coupling.py tests/test_physics_fidelity.py tests/test_uncertainty_budget.py -q` passed (`87 passed, 3 skipped in 1.34s`); `git diff --check` passed.
- Remaining scientific limit: this prevents provenance placeholders from closing the high-fidelity gate. It does not add the missing line-referenced experimental targets or production validation comparisons themselves.

Ratchet update 2026-05-05, source-authority requires local KR files:

- Modules touched: `src/dpf/validation/quality_assessment.py` and `tests/test_quality_assessment.py`.
- Source-authority helpers now require each claimed `KnowledgeReference/...` source path to resolve to a real local file, not merely start with the right prefix.
- `source_authority_evidence()` and the result-derived audit both reject missing local KR files. The synthetic high-fidelity closure packet was updated to use real KR files for spatial, neutron, MHD, circuit, and source-authority records.
- This is a stricter implementation of the user's source-of-truth rule: a validation claim cannot cite a placeholder such as `KnowledgeReference/neutron.md` and close the provenance gate.
- Verification status: `python3 -m py_compile src/dpf/validation/quality_assessment.py tests/test_quality_assessment.py` passed; `python3 -m pytest tests/test_quality_assessment.py::TestQualityAssessment::test_source_authority_evidence_rejects_missing_kr_file tests/test_quality_assessment.py::TestQualityAssessment::test_result_level_source_authority_evidence_can_support_gap tests/test_quality_assessment.py::TestQualityAssessment::test_source_authority_evidence_from_result_collects_kr_lines tests/test_quality_assessment.py::TestQualityAssessment::test_gap_report_cross_checks_manual_source_authority_packet tests/test_quality_assessment.py::TestQualityAssessment::test_high_fidelity_readiness_can_pass_with_complete_evidence_packet -q` passed (`5 passed in 0.77s`); `python3 -m pytest tests/test_quality_assessment.py tests/test_kr_targets.py -q` passed (`83 passed in 0.53s`); `python3 -m pytest tests/test_mhd_physics_integration.py tests/test_mhd_numerical_fidelity.py tests/test_circuit_field_coupling.py tests/test_physics_fidelity.py tests/test_uncertainty_budget.py -q` passed (`87 passed, 3 skipped in 1.34s`); `git diff --check` passed.
- Remaining scientific limit: existence of the KR file does not prove that the cited line range contains the claimed fact. It does, however, remove placeholder paths from the high-fidelity source-authority path.

Ratchet update 2026-05-05, source-authority validates line ranges:

- Modules touched: `src/dpf/validation/quality_assessment.py` and `tests/test_quality_assessment.py`.
- Source-authority helpers now parse cited line ranges and require every numeric range to fall within the actual local KR file length.
- Accepted formats include single ranges and comma-separated ranges such as `383-398, 1900-1955`; missing, malformed, reversed, or out-of-bounds ranges fail the source-authority check.
- This strengthens the local-source rule from "real file exists" to "real file and plausible line range exist" for any evidence that tries to close the source-authority gap.
- Verification status: `python3 -m py_compile src/dpf/validation/quality_assessment.py tests/test_quality_assessment.py` passed; `python3 -m pytest tests/test_quality_assessment.py::TestQualityAssessment::test_source_authority_evidence_rejects_missing_kr_file tests/test_quality_assessment.py::TestQualityAssessment::test_source_authority_evidence_rejects_out_of_bounds_line_range tests/test_quality_assessment.py::TestQualityAssessment::test_result_level_source_authority_evidence_can_support_gap tests/test_quality_assessment.py::TestQualityAssessment::test_source_authority_evidence_from_result_collects_kr_lines tests/test_quality_assessment.py::TestQualityAssessment::test_high_fidelity_readiness_can_pass_with_complete_evidence_packet -q` passed (`5 passed in 0.51s`); `python3 -m pytest tests/test_quality_assessment.py tests/test_kr_targets.py -q` passed (`84 passed in 0.51s`); `python3 -m pytest tests/test_mhd_physics_integration.py tests/test_mhd_numerical_fidelity.py tests/test_circuit_field_coupling.py tests/test_physics_fidelity.py tests/test_uncertainty_budget.py -q` passed (`87 passed, 3 skipped in 1.24s`); `git diff --check` passed.
- Remaining scientific limit: line-range validity still does not semantically verify that the cited lines support the claim. It does make impossible line references fail automatically.

Ratchet update 2026-05-05, KR target authority manifest:

- Modules touched: `src/dpf/validation/kr_targets.py`, `src/dpf/validation/__init__.py`, `tests/test_kr_targets.py`, `CortexFindings.md`, and `CodexFindings.md`.
- Created `CortexFindings.md` with the detailed reviewed plan from KR target authority through end-to-end high-fidelity demonstration.
- Added `kr_validation_target_manifest()` to enumerate every coded KR validation target with target id, device, validation scope, tier, role, source, and flattened source-line metadata.
- Added `kr_validation_target_source_audit()` to run local source-authority checks over the manifest, using the existing KR file and line-range validation path.
- Exported both helpers from `dpf.validation`.
- This completes plan step 1: the code can now answer which local KR targets are currently available and whether their cited source files and line ranges are locally valid.
- Verification status: `python3 -m py_compile src/dpf/validation/kr_targets.py src/dpf/validation/__init__.py tests/test_kr_targets.py` passed; `python3 -m pytest tests/test_kr_targets.py::test_kr_validation_target_manifest_lists_coded_targets tests/test_kr_targets.py::test_kr_validation_target_source_audit_passes_for_local_targets -q` passed (`2 passed in 0.89s`); `python3 -m pytest tests/test_quality_assessment.py tests/test_kr_targets.py -q` passed (`86 passed in 0.50s`); `python3 -m pytest tests/test_mhd_physics_integration.py tests/test_mhd_numerical_fidelity.py tests/test_circuit_field_coupling.py tests/test_physics_fidelity.py tests/test_uncertainty_budget.py -q` passed (`87 passed, 3 skipped in 1.29s`); `git diff --check` passed.
- Remaining scientific limit: the manifest audits authority and structure only. It does not yet turn all target records into complete typed observable packets, and it does not prove semantic support inside each cited line range.

Ratchet update 2026-05-05, typed KR target coverage report:

- Modules touched: `src/dpf/validation/kr_targets.py`, `src/dpf/validation/__init__.py`, `tests/test_kr_targets.py`, `CortexFindings.md`, and `CodexFindings.md`.
- Added `kr_validation_target_coverage_report()` to map coded KR targets into the end-to-end observable groups required by the plan: circuit waveform, phase semantics, phase timing, spatial density, spatial magnetic/EM, spatial temperature, neutron timing, neutron spectrum, neutron anisotropy, neutron detector response, and uncertainty.
- The report intentionally fails today because the current target set is incomplete. It marks `circuit_waveform` as missing and `phase_timing` as partial because the PF-1000 16 kV target lacks all axial/radial/pinch timing observables required for full tier 2.
- This converts part of the extraction roadmap from prose into a machine-readable audit.
- Verification status: `python3 -m py_compile src/dpf/validation/kr_targets.py src/dpf/validation/__init__.py tests/test_kr_targets.py` passed; `python3 -m pytest tests/test_kr_targets.py::test_kr_validation_target_coverage_report_lists_remaining_groups tests/test_kr_targets.py::test_kr_validation_target_source_audit_passes_for_local_targets -q` passed (`2 passed in 0.80s`); `python3 -m pytest tests/test_quality_assessment.py tests/test_kr_targets.py -q` passed (`87 passed in 0.55s`); `python3 -m pytest tests/test_mhd_physics_integration.py tests/test_mhd_numerical_fidelity.py tests/test_circuit_field_coupling.py tests/test_physics_fidelity.py tests/test_uncertainty_budget.py -q` passed (`87 passed, 3 skipped in 1.36s`); `git diff --check` passed.
- Remaining scientific limit: target coverage does not validate simulation outputs or prove same-scope closure. It makes missing target groups visible so they cannot be hidden in a high-fidelity claim.

Ratchet update 2026-05-05, PF-1000 partial circuit waveform target:

- Modules touched: `src/dpf/validation/kr_targets.py`, `src/dpf/validation/__init__.py`, `tests/test_kr_targets.py`, `CortexFindings.md`, and `CodexFindings.md`.
- Added `pf1000_16kv_current_waveform_targets()` as a typed tier-1 KR target for PF-1000 16 kV measured current-waveform context from `KnowledgeReference/radiation-physics-and-chemistry-188-2021-109633.md`.
- The target records device/shot context, 16 kV operation, 1.05-1.2 Torr deuterium fill, measured-current availability, peak-current range, shot-12581 peak and pinch current, and the KR statement that the fit is good only until the end of the current dip.
- The target explicitly lists missing data for full tier-1 validation: digitized current trace points, per-point current uncertainty, and per-point timing uncertainty.
- `kr_validation_target_coverage_report()` now marks `circuit_waveform` as `partial` instead of `missing`.
- KnowledgeReference basis: measured current waveforms and Lee fit context in `KnowledgeReference/radiation-physics-and-chemistry-188-2021-109633.md:111-124`, `217-236`, `247-285`, `294-300`, and `332-346`.
- Verification status: `python3 -m py_compile src/dpf/validation/kr_targets.py src/dpf/validation/__init__.py tests/test_kr_targets.py` passed; `python3 -m pytest tests/test_kr_targets.py::test_pf1000_current_waveform_target_metadata tests/test_kr_targets.py::test_kr_validation_target_manifest_lists_coded_targets tests/test_kr_targets.py::test_kr_validation_target_coverage_report_lists_remaining_groups tests/test_kr_targets.py::test_kr_validation_target_source_audit_passes_for_local_targets -q` passed (`4 passed in 2.62s`); `python3 -m pytest tests/test_quality_assessment.py tests/test_kr_targets.py -q` passed (`88 passed in 0.52s`); `python3 -m pytest tests/test_mhd_physics_integration.py tests/test_mhd_numerical_fidelity.py tests/test_circuit_field_coupling.py tests/test_physics_fidelity.py tests/test_uncertainty_budget.py -q` passed (`87 passed, 3 skipped in 1.43s`); `git diff --check` passed.
- Remaining scientific limit: this target is still not a pointwise waveform dataset. It supports extraction and provenance but cannot by itself close tier-1 waveform validation for production runs.

Ratchet update 2026-05-05, Lee-course full phase timing example target:

- Modules touched: `src/dpf/validation/kr_targets.py`, `src/dpf/validation/__init__.py`, `tests/test_kr_targets.py`, `CortexFindings.md`, and `CodexFindings.md`.
- Added `lee_course_nx2_neon_phase_timing_example_targets()` with typed axial, radial, and pinch endpoint timing from the Lee/RADPF course worksheet.
- The target records axial end/radial start at 1.172 us, radial end at 1.407 us, radial duration 0.235 us, pinch start at 1.38 us, pinch duration 26.2 ns, radial shock axis time 178 ns after radial start, and reflected-shock/piston timing around 210 ns after radial start.
- The target explicitly lists predictive limitations: it is an NX2 neon fitted worksheet example, not a same-shot deuterium experimental validation target with uncertainty.
- `kr_validation_target_coverage_report()` continues to mark `phase_timing` as partial, now with both the PF-1000 partial current-dip target and the Lee-course full endpoint example visible.
- KnowledgeReference basis: `KnowledgeReference/a-course-on-plasma-focus-numerical-experiments-s-lee-and-s-h-saw-part-1-basic-course.md:1938-1958`, `1978-1994`, and `2038-2048`.
- Verification status: `python3 -m py_compile src/dpf/validation/kr_targets.py src/dpf/validation/__init__.py tests/test_kr_targets.py` passed; `python3 -m pytest tests/test_kr_targets.py::test_lee_course_nx2_phase_timing_example_metadata tests/test_kr_targets.py::test_kr_validation_target_coverage_report_lists_remaining_groups tests/test_kr_targets.py::test_kr_validation_target_source_audit_passes_for_local_targets -q` passed (`3 passed in 2.62s`); `python3 -m pytest tests/test_quality_assessment.py tests/test_kr_targets.py -q` passed (`89 passed in 0.66s`); `python3 -m pytest tests/test_mhd_physics_integration.py tests/test_mhd_numerical_fidelity.py tests/test_circuit_field_coupling.py tests/test_physics_fidelity.py tests/test_uncertainty_budget.py -q` passed (`87 passed, 3 skipped in 1.44s`); `git diff --check` passed.
- Remaining scientific limit: this improves typed extraction and tests phase endpoint semantics, but it cannot close predictive tier 2 until a same-device/same-shot experimental deuterium phase target with uncertainty is extracted.

Ratchet update 2026-05-05, app exports KR target source and coverage reports:

- Modules touched: `app_mhd.py`, `tests/test_mhd_physics_integration.py`, `CortexFindings.md`, and `CodexFindings.md`.
- App post-processing now exports `kr_validation_target_source_audit` and `kr_validation_target_coverage` before validation tiers, predictive readiness, scientific-accuracy gaps, and high-fidelity readiness.
- Ordinary result payloads now show both that the currently coded target sources are locally valid and that the target set is still incomplete for end-to-end predictive validation.
- The app smoke test asserts that source audit passes but target coverage does not, and that `phase_timing` remains in `missing_or_partial_groups`.
- Verification status: `python3 -m py_compile app_mhd.py tests/test_mhd_physics_integration.py` passed; `python3 -m pytest tests/test_mhd_physics_integration.py::test_predictive_readiness_exported_and_blocks_unvalidated_claims tests/test_kr_targets.py::test_kr_validation_target_coverage_report_lists_remaining_groups tests/test_kr_targets.py::test_kr_validation_target_source_audit_passes_for_local_targets -q` passed (`3 passed in 0.73s`); `python3 -m pytest tests/test_quality_assessment.py tests/test_kr_targets.py -q` passed (`89 passed in 0.55s`); `python3 -m pytest tests/test_mhd_physics_integration.py tests/test_mhd_numerical_fidelity.py tests/test_circuit_field_coupling.py tests/test_physics_fidelity.py tests/test_uncertainty_budget.py -q` passed (`87 passed, 3 skipped in 1.24s`); `git diff --check` passed.
- Remaining scientific limit: this exposes target coverage; it does not add missing same-shot waveform, phase, spatial, neutron, or uncertainty targets.

Ratchet update 2026-05-05, KR target semantic source-window audit:

- Modules touched: `src/dpf/validation/kr_targets.py`, `src/dpf/validation/__init__.py`, `app_mhd.py`, `tests/test_kr_targets.py`, `tests/test_mhd_physics_integration.py`, `CortexFindings.md`, and `CodexFindings.md`.
- Added `kr_validation_target_semantic_audit()` to verify that each coded KR target's cited local line windows contain domain markers matching the extracted observable.
- App post-processing now exports `kr_validation_target_semantic_audit` alongside the target source audit and target coverage report.
- The audit currently passes for the coded target set, including Lee/RADPF phase semantics, NX2 phase timing, PF-1000 current waveform and phase timing, MJOLNIR neutron timing/detector response, PF-1000 spatial/density targets, LLNL EM fluctuation targets, and DPF pinch-temperature targets.
- The Malir PF-1000 density target marker was aligned to the source-window wording: the cited lines use `interferometer` / `interferometric` diagnostic language, not the title-form word `interferometry`.
- Verification status: `python3 -m py_compile app_mhd.py src/dpf/validation/kr_targets.py src/dpf/validation/__init__.py tests/test_kr_targets.py tests/test_mhd_physics_integration.py` passed; `python3 -m pytest tests/test_mhd_physics_integration.py::test_predictive_readiness_exported_and_blocks_unvalidated_claims tests/test_kr_targets.py::test_kr_validation_target_semantic_audit_passes_for_coded_targets -q` passed (`2 passed in 1.19s`); `python3 -m pytest tests/test_quality_assessment.py tests/test_kr_targets.py -q` passed (`90 passed in 0.50s`); `python3 -m pytest tests/test_mhd_physics_integration.py tests/test_mhd_numerical_fidelity.py tests/test_circuit_field_coupling.py tests/test_physics_fidelity.py tests/test_uncertainty_budget.py -q` passed (`87 passed, 3 skipped in 1.68s`); `git diff --check` passed.
- Remaining scientific limit: this proves only local cited-window plausibility. It does not prove complete extraction, same-scope closure, digitized waveform availability, or uncertainty-bearing simulation agreement.

Ratchet update 2026-05-05, KR target coverage is now a high-fidelity gap:

- Modules touched: `src/dpf/validation/quality_assessment.py`, `tests/test_quality_assessment.py`, `tests/test_mhd_physics_integration.py`, `CortexFindings.md`, and `CodexFindings.md`.
- Added a `kr_target_coverage` area to `scientific_accuracy_gap_report()`.
- High-fidelity readiness now requires both a passing `kr_validation_target_coverage_report` and a passing `kr_validation_target_semantic_audit`.
- Ordinary app results expose this as a partial blocker today because the target coverage report still lists partial `circuit_waveform`, `phase_timing`, and `spatial_temperature` groups.
- The synthetic complete high-fidelity test now has to provide explicit passing target-coverage and semantic-audit packets, so a complete evidence packet cannot bypass target extraction.
- Verification status: `python3 -m py_compile src/dpf/validation/quality_assessment.py tests/test_quality_assessment.py tests/test_mhd_physics_integration.py` passed; `python3 -m pytest tests/test_quality_assessment.py::TestQualityAssessment::test_scientific_accuracy_gap_report_lists_remaining_work tests/test_quality_assessment.py::TestQualityAssessment::test_high_fidelity_readiness_can_pass_with_complete_evidence_packet tests/test_mhd_physics_integration.py::test_predictive_readiness_exported_and_blocks_unvalidated_claims -q` passed (`3 passed in 0.73s`); `python3 -m pytest tests/test_quality_assessment.py tests/test_kr_targets.py -q` passed (`90 passed in 0.46s`); `python3 -m pytest tests/test_mhd_physics_integration.py tests/test_mhd_numerical_fidelity.py tests/test_circuit_field_coupling.py tests/test_physics_fidelity.py tests/test_uncertainty_budget.py -q` passed (`87 passed, 3 skipped in 1.25s`); `git diff --check` passed.
- Remaining scientific limit: this closes a readiness-gate loophole. It does not yet add digitized current traces, same-shot phase timing, or same-device temperature targets.

Ratchet update 2026-05-05, same-scope KR target coverage audit:

- Modules touched: `src/dpf/validation/kr_targets.py`, `src/dpf/validation/__init__.py`, `src/dpf/validation/quality_assessment.py`, `app_mhd.py`, `tests/test_kr_targets.py`, `tests/test_quality_assessment.py`, `tests/test_mhd_physics_integration.py`, `CortexFindings.md`, and `CodexFindings.md`.
- Added `kr_validation_same_scope_target_report()` to report whether any one KR validation scope covers every end-to-end target group.
- App post-processing now exports `kr_validation_same_scope_targets`.
- The scientific-accuracy `kr_target_coverage` gap now requires target coverage, same-scope target coverage, and semantic source-window audit before high-fidelity readiness can pass.
- Current result: no same-scope target set passes. The best scope is `mjolnir_neutron_timing_2025_goyon`, which currently combines MJOLNIR neutron timing, spectrum, anisotropy, and detector response targets, but lacks circuit waveform, phase timing, spatial density, spatial magnetic/EM, spatial temperature, and uncertainty groups.
- Verification status: `python3 -m py_compile app_mhd.py src/dpf/validation/kr_targets.py src/dpf/validation/__init__.py src/dpf/validation/quality_assessment.py tests/test_kr_targets.py tests/test_quality_assessment.py tests/test_mhd_physics_integration.py` passed; `python3 -m pytest tests/test_kr_targets.py::test_kr_validation_same_scope_target_report_requires_one_scope tests/test_quality_assessment.py::TestQualityAssessment::test_high_fidelity_readiness_can_pass_with_complete_evidence_packet tests/test_mhd_physics_integration.py::test_predictive_readiness_exported_and_blocks_unvalidated_claims -q` passed (`3 passed in 3.03s`); `python3 -m pytest tests/test_quality_assessment.py tests/test_kr_targets.py -q` passed (`91 passed in 0.49s`); `python3 -m pytest tests/test_mhd_physics_integration.py tests/test_mhd_numerical_fidelity.py tests/test_circuit_field_coupling.py tests/test_physics_fidelity.py tests/test_uncertainty_budget.py -q` passed (`87 passed, 3 skipped in 1.32s`); `git diff --check` passed.
- Remaining scientific limit: this prevents cross-device target aggregation from satisfying an end-to-end claim. It does not create missing same-scope experimental waveform, phase, spatial, or uncertainty targets.

Ratchet update 2026-05-05, MJOLNIR stagnation temperature target context:

- Modules touched: `src/dpf/validation/kr_targets.py`, `src/dpf/validation/__init__.py`, `tests/test_kr_targets.py`, `CortexFindings.md`, and `CodexFindings.md`.
- Added `mjolnir_stagnation_temperature_targets()` from `KnowledgeReference/neutron-generation-dynamics-inside-a-ma-class-dense-plasma-focus-z-pinch-5.md`.
- The target is same-scope with `mjolnir_neutron_timing_2025_goyon` and adds partial spatial-temperature context to the MJOLNIR target set.
- The target records the KR stagnation-temperature scaling reference of 21 keV, the `(Te + Ti) / 2` average-temperature definition, the several-keV MJOLNIR stagnation context, and the explicit missing items for full tier 4: direct temperature diagnostic, experimental uncertainty, and same-scope density/magnetic-field targets.
- `kr_validation_target_coverage_report()` and `kr_validation_same_scope_target_report()` still mark spatial temperature as partial because this is shock-theory/MHD-kinetic context, not a direct measured temperature diagnostic.
- Verification status: `python3 -m py_compile src/dpf/validation/kr_targets.py src/dpf/validation/__init__.py tests/test_kr_targets.py` passed; `python3 -m pytest tests/test_kr_targets.py::test_mjolnir_stagnation_temperature_target_is_partial_context tests/test_kr_targets.py::test_kr_validation_target_semantic_audit_passes_for_coded_targets tests/test_kr_targets.py::test_kr_validation_target_coverage_report_lists_remaining_groups tests/test_kr_targets.py::test_kr_validation_same_scope_target_report_requires_one_scope -q` passed (`4 passed in 0.50s`); `python3 -m pytest tests/test_quality_assessment.py tests/test_kr_targets.py -q` passed (`92 passed in 0.58s`); `python3 -m pytest tests/test_mhd_physics_integration.py tests/test_mhd_numerical_fidelity.py tests/test_circuit_field_coupling.py tests/test_physics_fidelity.py tests/test_uncertainty_budget.py -q` passed (`87 passed, 3 skipped in 1.46s`); `git diff --check` passed.
- Remaining scientific limit: this improves same-scope target context but still cannot close spatial DPF validation without direct measured temperature, density, magnetic/EM, timing, and uncertainty evidence from one compatible scope.

Ratchet update 2026-05-05, corpus review status saved and audited:

- Modules touched: `src/dpf/validation/kr_corpus.py`, `src/dpf/validation/__init__.py`, `app_mhd.py`, `tests/test_kr_corpus.py`, `tests/test_mhd_physics_integration.py`, `CortexFindings.md`, and `CodexFindings.md`.
- Saved the explicit status that the complete `KnowledgeReference/` corpus has not yet been line-by-line review-closed.
- Added `kr_corpus_inventory()` to count the local source-of-truth tree and `kr_corpus_review_status()` to compare coded target sources against DPF-named markdown files.
- Current corpus inventory: 827 total files, 398 markdown files, 396 JSON files, and 54 DPF-named markdown files.
- Current coded review closure: 11 coded KR target records from 7 unique KR source files; 6 of 54 DPF-named markdown files are represented by coded targets; 48 DPF-named markdown files remain unreviewed under the coded-target rule.
- App post-processing now exports `kr_corpus_review_status`, so ordinary result payloads show that corpus review remains open.
- Verification status: `python3 -m py_compile src/dpf/validation/kr_corpus.py src/dpf/validation/__init__.py app_mhd.py tests/test_kr_corpus.py tests/test_mhd_physics_integration.py` passed; `python3 -m pytest tests/test_kr_corpus.py tests/test_mhd_physics_integration.py::test_predictive_readiness_exported_and_blocks_unvalidated_claims -q` passed (`3 passed in 1.14s`).
- Remaining scientific limit: this makes corpus-review incompleteness auditable. It does not extract the remaining waveform, phase, spatial, neutron, or uncertainty data. The next ratchet is to review the 48 unreviewed DPF-named markdown files and either extract targets with source lines or mark them non-extractable with reasons.

Ratchet update 2026-05-05, unreviewed DPF source triage queue:

- Modules touched: `src/dpf/validation/kr_corpus.py`, `src/dpf/validation/__init__.py`, `tests/test_kr_corpus.py`, `CortexFindings.md`, and `CodexFindings.md`.
- Added `kr_unreviewed_dpf_source_triage()` to rank the 48 unreviewed DPF-named markdown files by observable keyword categories.
- Current triage counts among the 48 unreviewed DPF-named markdown files: 30 circuit waveform candidates, 31 phase timing candidates, 17 spatial density candidates, 33 spatial magnetic/EM candidates, 42 spatial temperature candidates, 42 neutron validation candidates, and 18 uncertainty candidates.
- Top broad-category candidates are `focus-fusion-overview-of-progress-towards-p-b11-fusion-with-the-dense-plasma-focus.md`, `gribkov-2007-pf1000-jphysd-part2.md`, `regular-article-deuterium-argon-admixture-for-plasma-focus-neutron-generation-muhammad-luqman.md`, `scholz-2007-pf1000-part2-jphysd.md`, and `characterising-the-plasma-focus-pinch-and-speed-enhancing-the-neutron-yield.md`.
- Verification status: `python3 -m py_compile src/dpf/validation/kr_corpus.py src/dpf/validation/__init__.py tests/test_kr_corpus.py` passed; `python3 -m pytest tests/test_kr_corpus.py -q` passed (`3 passed in 0.53s`).
- Remaining scientific limit: this is a review queue, not extraction. Each candidate must still be reviewed line by line and converted into typed KR targets or marked non-extractable with reasons.

Ratchet update 2026-05-05, PF-1000 full-energy 2007 target bundle:

- Modules touched: `src/dpf/validation/kr_targets.py`, `src/dpf/validation/__init__.py`, `tests/test_kr_targets.py`, `CortexFindings.md`, and `CodexFindings.md`.
- Reviewed two high-priority local PF-1000 papers from the source-of-truth queue: `KnowledgeReference/gribkov-2007-pf1000-jphysd-part2.md` and `KnowledgeReference/scholz-2007-pf1000-part2-jphysd.md`.
- Added `pf1000_full_energy_phase_context_targets()` from paper I with the PF-1000 full-energy phase semantics, 2-4 Torr / up to 850 kJ / 2.5-3 MA context, maximum compression about 100 ns before current dip, maximum compression about 2 us after current maximum, about 150 ns confinement/neutron-pulse timing, and explicit missing digitized phase endpoints.
- Added `pf1000_full_energy_neutron_spatial_targets()` from paper II with 810 kJ operation, shot 3121 at 465 Pa and 35 kV, typical current 2.5-2.6 MA, best current near 3 MA, estimated average pinch current about 2 MA, neutron anisotropy ratios, yield range and maximum, TOF correction, first-pulse 2.45 MeV context, density and magnetic-field estimates, temperature estimates, and detector/temperature limitations.
- Both targets share `validation_scope="pf1000_full_energy_2007_gribkov_scholz"`. This gives the project its broadest same-scope PF-1000 target packet so far: circuit waveform, phase semantics, phase timing, spatial density, spatial magnetic/EM, spatial temperature, neutron timing, neutron spectrum, neutron anisotropy, neutron detector response, and uncertainty are present in one scope.
- The packet remains explicitly partial and does not close readiness. It is still missing digitized current/neutron traces, direct pinch-current measurement, direct ion-temperature measurement, full room-scatter/detector transport response, and quantitative uncertainty.
- Current corpus closure after this ratchet: 13 coded KR target records from 9 unique local KR source files; 8 of 54 DPF-named markdown files are represented by coded targets; 46 DPF-named markdown files remain unreviewed under the coded-target rule.
- Updated triage counts after removing these two sources from the queue: 28 circuit waveform candidates, 29 phase timing candidates, 15 spatial density candidates, 31 spatial magnetic/EM candidates, 40 spatial temperature candidates, 40 neutron validation candidates, and 16 uncertainty candidates.
- KnowledgeReference basis: PF-1000 paper I lines `36-50`, `59-145`, `642-678`, `720-750`, `760-780`, and `1238-1260`; PF-1000 paper II lines `320-386`, `392-467`, `465-532`, `530-565`, `585-635`, `716-820`, and `1467-1558`.
- Verification status: `python3 -m py_compile src/dpf/validation/kr_targets.py src/dpf/validation/__init__.py tests/test_kr_targets.py` passed; `python3 -m pytest tests/test_kr_targets.py -q` passed (`43 passed in 0.45s`).
- Remaining scientific limit: this improves target authority and same-scope structure. It still does not produce a validated end-to-end simulation because the source itself identifies missing direct ion-temperature and pinch-current measurements, scatter-limited neutron pulse interpretation, and non-digitized waveforms/histories.

Ratchet update 2026-05-05, PF-1000 same-scope detector-response context:

- Modules touched: `src/dpf/validation/kr_targets.py`, `tests/test_kr_targets.py`, `CortexFindings.md`, and `CodexFindings.md`.
- Extended `pf1000_full_energy_neutron_spatial_targets()` with same-scope activation-counter, indium/bubble-detector cross-check, AmBe calibration, scintillator-PM, time-of-flight, and room-scatter response requirements from paper II.
- The PF-1000 full-energy scope now has every required end-to-end target group present in one validation scope.
- Same-scope closure still fails because the detector-response group is partial. The KR source provides calibration, TOF, detector geometry, and scatter limitations, but not a complete neutron-field transport or detector-response model.
- Verification status: `python3 -m py_compile src/dpf/validation/kr_targets.py tests/test_kr_targets.py` passed; `python3 -m pytest tests/test_kr_targets.py -q` passed (`43 passed in 0.50s`).
- Remaining scientific limit: this changes the scope from "missing detector-response group" to "partial detector-response group." It does not digitize traces, model hall scatter, or compare production neutron histories against detectors.

Ratchet update 2026-05-05, deuterium-argon admixture neutron target:

- Modules touched: `src/dpf/validation/kr_targets.py`, `src/dpf/validation/__init__.py`, `tests/test_kr_targets.py`, `CortexFindings.md`, and `CodexFindings.md`.
- Reviewed `KnowledgeReference/regular-article-deuterium-argon-admixture-for-plasma-focus-neutron-generation-muhammad-luqman.md` and added `deuterium_argon_admixture_neutron_targets()`.
- The target records the 2.7 kJ Mather-type PF context: 30 uF capacitor, up to 14 kV charge, 4 mbar fill, 10-70% argon mass mixtures, 30 shots per mixture, measured current/voltage waveforms, Rogowski conversion of 36 kA/V, voltage-probe calibration factor 0.71, Lee-model current fitting, and indium activation at 28 cm.
- Extracted target values include focus-time shift from 2.7 to 3.3 us, voltage-spike FWHM values for pure D2 and 10% argon, pure-D2 yield `(4.7 +/- 0.8)e6` n/shot, 50% argon yield `(3.0 +/- 0.6)e7` n/shot, recorded 50% argon yield `3.9e7` n/shot, pure-D2 energy into pinch `(83 +/- 17)` J, 50% argon energy into pinch `(139 +/- 16)` J, and computed pinch-current/temperature context.
- Current corpus closure after this ratchet: 14 coded KR target records from 10 unique local KR source files; 9 of 54 DPF-named markdown files are represented by coded targets; 45 DPF-named markdown files remain unreviewed under the coded-target rule.
- Updated triage counts after removing this source from the queue: 27 circuit waveform candidates, 28 phase timing candidates, 14 spatial density candidates, 30 spatial magnetic/EM candidates, 39 spatial temperature candidates, 39 neutron validation candidates, and 15 uncertainty candidates.
- KnowledgeReference basis: lines `44-58`, `124-139`, `139-193`, `220-235`, `245-249`, `344-408`, `410-441`, `509-527`, `584-600`, and `660-733` in the local deuterium-argon paper.
- Verification status: `python3 -m py_compile src/dpf/validation/kr_targets.py src/dpf/validation/__init__.py tests/test_kr_targets.py` passed; `python3 -m pytest tests/test_kr_targets.py -q` passed (`44 passed in 0.85s`); `python3 -m pytest tests/test_quality_assessment.py tests/test_kr_targets.py tests/test_kr_corpus.py -q` passed (`98 passed in 0.67s`); `python3 -m pytest tests/test_mhd_physics_integration.py tests/test_mhd_numerical_fidelity.py tests/test_circuit_field_coupling.py tests/test_physics_fidelity.py tests/test_uncertainty_budget.py -q` passed (`87 passed, 3 skipped in 1.84s`); `git diff --check` passed.
- Remaining scientific limit: this is an admixture-yield and activation target, not a full DPF validation packet. It lacks digitized waveform points, time-resolved neutron histories, direct temperature diagnostics, and same-scope spatial field/density comparisons.

Ratchet update 2026-05-05, FF-1 Focus Fusion plasmoid and p-B11 context target:

- Modules touched: `src/dpf/validation/kr_targets.py`, `src/dpf/validation/__init__.py`, `tests/test_kr_targets.py`, `CortexFindings.md`, and `CodexFindings.md`.
- Reviewed `KnowledgeReference/focus-fusion-overview-of-progress-towards-p-b11-fusion-with-the-dense-plasma-focus.md` and added `ff1_focus_fusion_plasmoid_targets()`.
- The target records FF-1/FF-2B device context, diagnostic suite, main and beam Rogowski context, ion-beam energy-transfer measurements, confined-ion energy from neutron TOF, isotropy support from bubble detectors, best 2016 neutron yield, wall-plug efficiency, estimated density, n-tau-T product, beryllium impurity/deposition measurements, QMF/p-B11 magnetic-field constraints, and current oscillation/yield-plateau limitations.
- Extracted values include 113 uF total capacitance, 115 kJ maximum stored energy, slightly over 1 MA operation, 1.8 us rise time, 2 kJ / 3 MeV / 5 ns ion-beam event, 240 +/- 20 keV confined-ion energy, best 2016 yield `2.5 +/- 0.25e11` neutrons, density estimate `3e19-4e19 cm^-3`, `n tau T = 3.4 +/- 0.8e20 keV-s/m^3`, and `Zeff` reduced to about 1.004.
- Current corpus closure after this ratchet: 15 coded KR target records from 11 unique local KR source files; 10 of 54 DPF-named markdown files are represented by coded targets; 44 DPF-named markdown files remain unreviewed under the coded-target rule.
- Updated triage counts after removing this source from the queue: 26 circuit waveform candidates, 27 phase timing candidates, 13 spatial density candidates, 29 spatial magnetic/EM candidates, 38 spatial temperature candidates, 38 neutron validation candidates, and 14 uncertainty candidates.
- KnowledgeReference basis: lines `66-98`, `121-164`, `169-198`, `233-250`, `259-323`, `420-433`, `770-800`, `808-849`, `856-929`, `930-1065`, `1159-1197`, and `1245-1367` in the local Focus Fusion overview.
- Verification status: `python3 -m py_compile src/dpf/validation/kr_targets.py src/dpf/validation/__init__.py tests/test_kr_targets.py` passed; `python3 -m pytest tests/test_kr_targets.py -q` passed (`45 passed in 0.80s`).
- Remaining scientific limit: this target is not p-B11 net-energy validation. It includes measured deuterium FF-1 values, but the p-B11/QMF/net-energy content is constraint, projection, or reduced simulation context. It lacks digitized waveforms, full detector response, shot-series distributions, and direct advanced-fuel yield validation.

Ratchet update 2026-05-05, Lee drive-parameter speed-enhancement target:

- Modules touched: `src/dpf/validation/kr_targets.py`, `src/dpf/validation/__init__.py`, `tests/test_kr_targets.py`, `CortexFindings.md`, and `CodexFindings.md`.
- Reviewed `KnowledgeReference/characterising-the-plasma-focus-pinch-and-speed-enhancing-the-neutron-yield.md` and added `lee_drive_parameter_speed_enhancement_targets()`.
- The target records generic Lee axial snowplow/radial slug semantics, deuterium and neon pinch radius/length/lifetime scaling with anode radius, the neutron-optimized drive parameter `Ip/a/sqrt(p_D2) = 89.0 +/- 7.7 kA/cm/sqrt(torr)`, typical axial and radial speeds, constant-speed `Y ~ I^4` scaling, speed-enhanced thermonuclear and beam-target scaling, and operational speed limits where focus quality deteriorates.
- Current corpus closure after this ratchet: 16 coded KR target records from 12 unique local KR source files; 11 of 54 DPF-named markdown files are represented by coded targets; 43 DPF-named markdown files remain unreviewed under the coded-target rule.
- Updated triage counts after removing this source from the queue: 25 circuit waveform candidates, 26 phase timing candidates, 13 spatial density candidates, 28 spatial magnetic/EM candidates, 37 spatial temperature candidates, 37 neutron validation candidates, and 13 uncertainty candidates.
- KnowledgeReference basis: lines `17-28`, `32-102`, `194-197`, `201-239`, `249-333`, `337-346`, `351-405`, `411-445`, and `480-481` in the local speed-enhancement paper.
- Verification status: `python3 -m py_compile src/dpf/validation/kr_targets.py src/dpf/validation/__init__.py tests/test_kr_targets.py` passed; `python3 -m pytest tests/test_kr_targets.py -q` passed (`46 passed in 0.49s`).
- Remaining scientific limit: this is a generic scaling/regime target. It is useful for checking speed and scaling assumptions, but cannot close same-shot validation without a device-specific waveform, pressure, geometry, phase timing, neutron history, and detector-response packet.

Ratchet update 2026-05-05, PFZ-200 hybrid X-pinch proton/neutron target:

- Modules touched: `src/dpf/validation/kr_targets.py`, `src/dpf/validation/__init__.py`, `tests/test_kr_targets.py`, `CortexFindings.md`, and `CodexFindings.md`.
- Reviewed `KnowledgeReference/deuterium-hybrid-x-pinch-driven-by-small-dense-plasma-focus-2.md` and added `pfz200_hybrid_xpinch_proton_neutron_targets()`.
- The target records PFZ-200 current/geometry/gas context, Rogowski current diagnostics, silver activation and nTOF detector setup, schlieren and CR-39 diagnostic details, neutron FWHM timing for 3 mm and 5 mm A-K gaps versus unmodified DPF operation, neutron-yield ranges, localized proton-source dimensions, proton spectrum/yield values, and anisotropy/shot-to-shot interpretation limits.
- Extracted values include 3 kJ energy, current above 200 kA, 1.6 us rise time, 360 Pa deuterium pressure, neutron production FWHM `(20 +/- 7)` ns for 3 mm gap, `(27 +/- 8)` ns for 5 mm gap by table, `(38 +/- 9)` ns for unmodified PFZ-200, hybrid average neutron yield about `6e7` after first-shot rejection, 3 mm source diameter `1.1-1.5` mm, maximum proton energy `3.6 MeV`, and inferred deuteron energy up to `1.3 MeV`.
- Current corpus closure after this ratchet: 17 coded KR target records from 13 unique local KR source files; 12 of 54 DPF-named markdown files are represented by coded targets; 42 DPF-named markdown files remain unreviewed under the coded-target rule.
- Updated triage counts after removing this source from the queue: 24 circuit waveform candidates, 26 phase timing candidates, 12 spatial density candidates, 27 spatial magnetic/EM candidates, 36 spatial temperature candidates, 36 neutron validation candidates, and 12 uncertainty candidates.
- KnowledgeReference basis: lines `59-69`, `134-149`, `151-183`, `204-220`, `223-269`, `271-305`, `308-317`, `410-459`, `463-475`, and `609-645,681-704` in the local PFZ-200 hybrid X-pinch paper.
- Verification status: `python3 -m py_compile src/dpf/validation/kr_targets.py src/dpf/validation/__init__.py tests/test_kr_targets.py` passed; `python3 -m pytest tests/test_kr_targets.py -q` passed (`47 passed in 0.48s`).
- Remaining scientific limit: this is a modified hybrid X-pinch load, not an ordinary DPF end-to-end target. It contributes useful particle-source and detector-response constraints, but not same-scope density, magnetic-field, or temperature validation for a standard DPF pinch.

Ratchet update 2026-05-05, LLNL fully kinetic DPF benchmark and duplicate review decisions:

- Modules touched: `src/dpf/validation/kr_targets.py`, `src/dpf/validation/kr_corpus.py`, `src/dpf/validation/__init__.py`, `tests/test_kr_targets.py`, `tests/test_kr_corpus.py`, `CortexFindings.md`, and `CodexFindings.md`.
- Reviewed the three local Schmidt/Tang/Welch fully kinetic DPF paper copies and selected `KnowledgeReference/fully-kinetic-simulations-of-dense-plasma-focus-z-pinch-8.md` as canonical for extraction. The base file and `-9` file are now explicit duplicate review decisions rather than duplicate scientific targets.
- Added `llnl_fully_kinetic_dpf_targets()` for the LLNL low-current DPF benchmark. It records LSP implicit-PIC setup, 2D cylindrical geometry, grid/domain/electrode dimensions, 1 torr deuterium density context, 4 kV/180 kA drive context, current dip and impedance targets, lower-hybrid-frequency field-fluctuation context, 12 keV ion and 3 keV electron hot-pinch temperatures, MeV-ion spectrum context, and fluid/hybrid/fully kinetic neutron-yield comparison.
- Extracted values include a 322-by-151 `r,z` grid, 5 cm anode, 1.5 cm cathode radius, 10 cm domain length, 1 mm initial sheath, neutral density `6.7e16 cm^-3`, sheath density `3.3e17 cm^-3`, fully kinetic current dip `15 kA` or `8%`, LLNL experimental current dips up to `40 kA` near 1 torr, impedance from `20 mOhm` to `1 Ohm`, hot-pinch temperatures `Ti ~ 12 keV` and `Te ~ 3 keV`, fully kinetic neutron yield `0.86e7`, LLNL experimental yield up to `2e7` at 180 kA, hybrid yield `3.6e4`, and fluid yield `0`.
- Current corpus closure after this ratchet: 18 coded KR target records from 14 unique local KR source files; 13 of 54 DPF-named markdown files are represented by coded targets; 2 additional DPF-named markdown files are review-closed by duplicate decisions; 39 DPF-named markdown files remain unreviewed.
- Updated triage counts after removing the three fully kinetic local copies from the review queue: 21 circuit waveform candidates, 23 phase timing candidates, 12 spatial density candidates, 24 spatial magnetic/EM candidates, 33 spatial temperature candidates, 33 neutron validation candidates, and 9 uncertainty candidates.
- KnowledgeReference basis: lines `20-32`, `35-66`, `70-99`, `102-112`, `113-142`, `143-151`, `155-159`, and `242-274` in the canonical local fully kinetic manuscript.
- Verification status: `python3 -m py_compile src/dpf/validation/kr_targets.py src/dpf/validation/kr_corpus.py src/dpf/validation/__init__.py tests/test_kr_targets.py tests/test_kr_corpus.py` passed; `python3 -m pytest tests/test_kr_targets.py tests/test_kr_corpus.py -q` passed (`52 passed in 0.91s`).
- Remaining scientific limit: this target is simulation-to-experiment context, not direct experimental validation. It supports the argument that kinetic physics is required for MeV ions and low-current neutron yield, but it cannot close detector response, shot-ensemble uncertainty, 3D kinetic validation, or a same-scope end-to-end predictive claim.

Ratchet update 2026-05-05, NSTec/Gemini fully 3D MHD rundown benchmark:

- Modules touched: `src/dpf/validation/kr_targets.py`, `src/dpf/validation/__init__.py`, `tests/test_kr_targets.py`, `CortexFindings.md`, and `CodexFindings.md`.
- Reviewed `KnowledgeReference/fully-three-dimensional-simulation-and-modeling-of-a-dense-plasma-focus.md` and added `nstec_3d_mhd_rundown_targets()`.
- The target records NSTec/Gemini device geometry, bank/circuit context, Faraday rotator current diagnostic setup, 37-shot waveform repeatability at 37.5 kV and 7.28 Torr, measured peak current and rundown time, 2D/3D ALEGRA current and rundown comparisons, 3D cathode-bar flow/inductance context, density-floor and artificial hot-start limits, and explicit MHD scope limits near Z-pinch.
- Extracted values include 432 uF bank capacitance, 70 kV maximum bank voltage, 1 MJ maximum stored energy, 36 coaxial cables, 8 rail-gap switches, 24 cathode bars, Faraday loop turns `5.25`, measured peak current `2.17 MA`, 2D predicted peak current `2.08 MA`, 3D predicted peak current `1.82 MA`, experimental rundown `6.96 us`, 3D rundown `6.69 us`, 2D rundown `5.59 us`, nominal series inductance `25 nH`, tweaked 2D series inductance `28.2 nH` without experimental justification, density floor `2.5e-4 kg/m^3`, artificial startup layer `1e6 K`, and startup stabilization near `1e4 K` in about `20 ns`.
- Current corpus closure after this ratchet: 19 coded KR target records from 15 unique local KR source files; 14 of 54 DPF-named markdown files are represented by coded targets; 2 additional DPF-named markdown files are review-closed by duplicate decisions; 38 DPF-named markdown files remain unreviewed.
- Updated triage counts after removing this source from the review queue: 20 circuit waveform candidates, 22 phase timing candidates, 12 spatial density candidates, 23 spatial magnetic/EM candidates, 32 spatial temperature candidates, 32 neutron validation candidates, and 8 uncertainty candidates.
- KnowledgeReference basis: lines `27-48`, `86-124`, `140-183`, `184-247`, `267-279`, `280-313,339-350`, `351-380`, `381-421`, `432-511`, `514-566`, `587-599`, and `600-624` in the local fully 3D simulation paper.
- Verification status: `python3 -m py_compile src/dpf/validation/kr_targets.py src/dpf/validation/__init__.py tests/test_kr_targets.py` passed; `python3 -m pytest tests/test_kr_targets.py tests/test_kr_corpus.py -q` passed (`53 passed in 0.56s`).
- Remaining scientific limit: this is a current/rundown and 3D-MHD scope target, not a neutron-yield validation packet. It lacks digitized Faraday trace points, per-shot uncertainty, direct spatial density/temperature/field diagnostics, detector response, and validated late-pinch kinetic/PIC closure.

Ratchet update 2026-05-05, MJOLNIR high/low-yield parasitic-current target:

- Modules touched: `src/dpf/validation/kr_targets.py`, `src/dpf/validation/__init__.py`, `tests/test_kr_targets.py`, `CortexFindings.md`, and `CodexFindings.md`.
- Reviewed `KnowledgeReference/goyon-2022-mjolnir-high-low.md` and added `mjolnir_high_low_parasitic_current_targets()`.
- The target records MJOLNIR 1-MJ and 2-MJ configurations, highest reported neutron yields, Rogowski current and anode-cathode voltage diagnostics, photodiode and framing-camera timing diagnostics, CHICAGO/BERTHA/PIC setup, snow-plow alternate-current-path modeling, sheath phase sequence, conditioning behavior, run-down/run-in velocity-yield correlations, current-dip and voltage-yield correlations, parasitic-current rBtheta/PIC mechanism, pressure degradation, and detector/trace/uncertainty gaps.
- Extracted values include up to `4.1e11` neutrons/pulse at about `3.3 MA`, 1-MJ current up to `2.5 MA` at 100 kV erected, 2-MJ commissioned current up to `3.25 MA` at 70 kV erected, 1-MJ lumped parameters `204 uF`, `67.4 nH`, `12.5 mOhm`, 2-MJ estimated parameters `408 uF`, `46.7 nH`, `6.3 mOhm`, 8-24 torr deuterium fill range, 16-frame camera with 3 ns exposures, voltage probe `955.5 ohm` / `8 nH` / `50 MHz` corrected to `200 MHz`, high-yield voltage spike about `180 kV`, high-yield alternate path `50 ns` after stagnation, low-yield conditioning alternate path `200 ns` before stagnation, and a 48-shot 16 torr current-dip/yield dataset.
- Current corpus closure after this ratchet: 20 coded KR target records from 16 unique local KR source files; 15 of 54 DPF-named markdown files are represented by coded targets; 2 additional DPF-named markdown files are review-closed by duplicate decisions; 37 DPF-named markdown files remain unreviewed.
- Updated triage counts after removing this source from the review queue: 19 circuit waveform candidates, 21 phase timing candidates, 12 spatial density candidates, 22 spatial magnetic/EM candidates, 31 spatial temperature candidates, 31 neutron validation candidates, and 7 uncertainty candidates.
- KnowledgeReference basis: lines `25-41`, `44-131`, `144-216`, `217-245`, `246-286`, `287-311`, `312-332`, `341-394`, `397-486`, `488-557`, `565-645`, `646-701`, `713-781`, `783-832`, and `836-908` in the local MJOLNIR high/low-yield paper.
- Verification status: `python3 -m py_compile src/dpf/validation/kr_targets.py src/dpf/validation/__init__.py tests/test_kr_targets.py` passed; `python3 -m pytest tests/test_kr_targets.py tests/test_kr_corpus.py -q` passed (`54 passed in 0.53s`).
- Remaining scientific limit: this target adds strong same-device mechanism constraints, but it is not a complete same-shot neutron validation packet. It lacks digitized traces, shot-resolved uncertainty, activation detector response, neutron timing/spectrum/anisotropy, direct spatial density/temperature/field diagnostics, and validated production coupling to the solver.

Ratchet update 2026-05-05, PF-400J x-ray diagnostic inference target:

- Modules touched: `src/dpf/validation/kr_targets.py`, `src/dpf/validation/__init__.py`, `tests/test_kr_targets.py`, `CortexFindings.md`, and `CodexFindings.md`.
- Reviewed `KnowledgeReference/inference-of-x-ray-emission-from-a-plasma-focus-discharge-comparison-between-characteristic.md` and added `pf400j_xray_inference_targets()`.
- The target records PF-400J device/bank/fill conditions, Rogowski/ILS/voltage-divider/Vivaldi diagnostics, scintillator-PMT x-ray detector context, data-acquisition details, 959-shot campaign size, breakdown and pinch feature definitions, x-ray feature-selection/ML inference results, normalization limits, and the explicit gap between x-ray diagnostic inference and neutron validation.
- Extracted values include 850 nF capacitance, 39 nH external inductance, 42 mOhm external resistance, 291 ns quarter period, 26 kV charging voltage, 287 J stored energy, hydrogen at 9 mbar, 13 mm anode effective length, 23 mm alumina insulator, BC-408 scintillator, Hamamatsu R1828-01 PMT, 5 mm aluminum casing, response above 20 keV, 1.4 kV PMT bias, 500 mA linearity threshold, 4 mm Pb filter with 250 keV cutoff, 959 recorded discharges, 5625-sample/900 ns signal windows, and a 75-by-75 CNN reshape.
- Current corpus closure after this ratchet: 21 coded KR target records from 17 unique local KR source files; 16 of 54 DPF-named markdown files are represented by coded targets; 2 additional DPF-named markdown files are review-closed by duplicate decisions; 36 DPF-named markdown files remain unreviewed.
- Updated triage counts after removing this source from the review queue: 18 circuit waveform candidates, 20 phase timing candidates, 12 spatial density candidates, 21 spatial magnetic/EM candidates, 30 spatial temperature candidates, 30 neutron validation candidates, and 6 uncertainty candidates.
- KnowledgeReference basis: lines `38-56`, `87-188`, `190-220`, `221-235`, `236-290`, `393-430`, `461-565`, `628-681`, `774-823`, `837-882`, and `922-952` in the local PF-400J x-ray inference paper.
- Verification status: `python3 -m py_compile src/dpf/validation/kr_targets.py src/dpf/validation/__init__.py tests/test_kr_targets.py` passed; `python3 -m pytest tests/test_kr_targets.py tests/test_kr_corpus.py -q` passed (`55 passed in 0.55s`).
- Remaining scientific limit: this is a hydrogen x-ray diagnostic target for a hundreds-of-joules device. It must not be promoted into deuterium neutron yield validation, same-scope high-fidelity spatial validation, or production solver readiness without absolute x-ray response, digitized traces, uncertainty, and cross-device validation.

Ratchet update 2026-05-05, Reuben 2024 thesis review decision:

- Modules touched: `src/dpf/validation/kr_corpus.py`, `tests/test_kr_corpus.py`, `CortexFindings.md`, and `CodexFindings.md`.
- Reviewed `KnowledgeReference/modification-and-numerical-modelling-of-dense-plasma-focus.md` and added an explicit `insufficient_extractable_validation_data` corpus review decision rather than a coded target.
- Reason: the local markdown has useful abstract, introduction, scaling-table, and figure-caption context for a 1 kJ / 1.3 uF / 40 kV modified DPF thesis, but the Experimental System, Numerical Modelling, Results and Discussion, and Conclusion sections are empty page stubs in this text extraction. The current waveform, radial trajectory, neutron production, pinch-temperature, and scaling values appear only as figure-list captions, not line-referenced validation data.
- Current corpus closure after this decision: 21 coded KR target records from 17 unique local KR source files; 16 of 54 DPF-named markdown files are represented by coded targets; 3 additional DPF-named markdown files are review-closed by explicit decisions; 35 DPF-named markdown files remain unreviewed.
- Updated triage counts after removing this source from the review queue: 17 circuit waveform candidates, 19 phase timing candidates, 12 spatial density candidates, 20 spatial magnetic/EM candidates, 29 spatial temperature candidates, 29 neutron validation candidates, and 5 uncertainty candidates.
- Verification status: `python3 -m py_compile src/dpf/validation/kr_corpus.py tests/test_kr_corpus.py src/dpf/validation/__init__.py` passed; `python3 -m pytest tests/test_kr_corpus.py -q` passed (`4 passed in 0.59s`).
- Remaining scientific limit: this source may be useful after PDF re-ingestion, but this markdown cannot support a coded validation target without risking target extraction from captions rather than the actual results text.

Ratchet update 2026-05-05, Goyon 2025 neutron-generation duplicate decision:

- Modules touched: `src/dpf/validation/kr_corpus.py`, `tests/test_kr_corpus.py`, `CortexFindings.md`, and `CodexFindings.md`.
- Reviewed `KnowledgeReference/neutron-generation-dynamics-inside-a-ma-class-dense-plasma-focus-z-pinch.md` and added an explicit duplicate review decision pointing to `KnowledgeReference/neutron-generation-dynamics-inside-a-ma-class-dense-plasma-focus-z-pinch-5.md`.
- Reason: the canonical `-5` file already backs the coded MJOLNIR neutron timing, stagnation-temperature, and neutron detector-response targets. This local copy is the same Phys. Plasmas 2025 Goyon MA-class MJOLNIR neutron-generation paper and should not produce duplicate targets.
- Current corpus closure after this decision: 21 coded KR target records from 17 unique local KR source files; 16 of 54 DPF-named markdown files are represented by coded targets; 4 additional DPF-named markdown files are review-closed by explicit decisions; 34 DPF-named markdown files remain unreviewed.
- Updated triage counts after removing this source from the review queue: 16 circuit waveform candidates, 18 phase timing candidates, 12 spatial density candidates, 19 spatial magnetic/EM candidates, 28 spatial temperature candidates, 28 neutron validation candidates, and 4 uncertainty candidates.
- Verification status: `python3 -m py_compile src/dpf/validation/kr_corpus.py tests/test_kr_corpus.py` passed; `python3 -m pytest tests/test_kr_corpus.py -q` passed (`4 passed in 0.55s`).
- Remaining scientific limit: duplicate closure prevents double-counting. It does not add coverage beyond the existing MJOLNIR Goyon 2025 coded target bundle.

Ratchet update 2026-05-05, Rawat 2015 generic operating-envelope target:

- Modules touched: `src/dpf/validation/kr_targets.py`, `src/dpf/validation/kr_corpus.py`, `src/dpf/validation/__init__.py`, `tests/test_kr_targets.py`, `tests/test_kr_corpus.py`, `CortexFindings.md`, and `CodexFindings.md`.
- Reviewed `KnowledgeReference/paper-open-access-dense-plasma-focus-from-alternative-fusion-source-to-versatile-high-energy-4.md` and the header/PDF-name duplicate `KnowledgeReference/paper-open-access-dense-plasma-focus-from-alternative-fusion-source-to-versatile-high-energy.md`.
- Added `rawat_dpf_operating_envelope_targets()` from the canonical `-4` source and added an explicit duplicate decision for the unsuffixed variant.
- Extracted values include 100-500 ns current-sheath formation, 500-3000 ns quarter-period context, optimized axial sheath speed 2-10 cm/us, radial speed 2-2.5 times axial, pinch density `5e24-1e26 m^-3`, DPF energy density `1.2e10-9.5e10 J/m^3`, pinch electron/ion temperature envelopes, electron energies from tens to hundreds of keV, ion energies from tens of keV to a few MeV, 20 degree forward ion cone context, 10-30 kV typical charge voltage, efficient operation at a few mbar, and shot-to-shot conditioning requirements.
- Current corpus closure after this ratchet: 22 coded KR target records from 18 unique local KR source files; 17 of 54 DPF-named markdown files are represented by coded targets; 5 additional DPF-named markdown files are review-closed by explicit decisions; 32 DPF-named markdown files remain unreviewed.
- Updated triage counts after removing this pair from the review queue: 16 circuit waveform candidates, 16 phase timing candidates, 10 spatial density candidates, 17 spatial magnetic/EM candidates, 26 spatial temperature candidates, 26 neutron validation candidates, and 2 uncertainty candidates.
- KnowledgeReference basis: lines `52-74`, `109-134`, `253-268`, `275-313`, `319-355`, `383-408`, and `720-722,749-754,791-802` in the local Rawat 2015 DPF review.
- Verification status: `python3 -m py_compile src/dpf/validation/kr_targets.py src/dpf/validation/kr_corpus.py src/dpf/validation/__init__.py tests/test_kr_targets.py tests/test_kr_corpus.py` passed; `python3 -m pytest tests/test_kr_targets.py tests/test_kr_corpus.py -q` passed (`56 passed in 0.63s`); broad post-ratchet sweep passed with `107 passed in 0.63s` for quality/KR tests, `87 passed, 3 skipped in 1.54s` for MHD/physics/UQ tests, and `git diff --check` clean.
- Remaining scientific limit: this target is a generic review-derived operating envelope. It can reject simulations that miss basic DPF scale and mechanism context, but it cannot validate a predictive end-to-end device model without same-scope measured current traces, phase endpoints, spatial diagnostics, neutron histories/spectra/anisotropy, detector response, and uncertainty.

Ratchet update 2026-05-05, Petrov/LLNL 2022 MJOLNIR duplicate decision:

- Modules touched: `src/dpf/validation/kr_corpus.py`, `tests/test_kr_corpus.py`, `CortexFindings.md`, and `CodexFindings.md`.
- Reviewed `KnowledgeReference/petrov-2022-mjolnir-high-low-discharges.md` and added an explicit duplicate review decision pointing to `KnowledgeReference/goyon-2022-mjolnir-high-low.md`.
- Reason: this LLNL report extraction is the same Schmidt/Goyon 2022 MJOLNIR high/low-performing discharge paper already represented by the coded `mjolnir_high_low_parasitic_current_2022_goyon` target. Differences are extraction headers/page stamps/line wrapping, not separate validation evidence.
- Current corpus closure after this decision: 22 coded KR target records from 18 unique local KR source files; 17 of 54 DPF-named markdown files are represented by coded targets; 6 additional DPF-named markdown files are review-closed by explicit decisions; 31 DPF-named markdown files remain unreviewed.
- Updated triage counts after removing this source from the review queue: 15 circuit waveform candidates, 15 phase timing candidates, 10 spatial density candidates, 16 spatial magnetic/EM candidates, 25 spatial temperature candidates, 25 neutron validation candidates, and 1 uncertainty candidate.
- Verification status: `python3 -m py_compile src/dpf/validation/kr_corpus.py tests/test_kr_corpus.py` passed; `python3 -m pytest tests/test_kr_corpus.py -q` passed (`4 passed in 0.51s`); broad post-decision sweep passed with `107 passed in 0.59s` for quality/KR tests, `87 passed, 3 skipped in 1.49s` for MHD/physics/UQ tests, and `git diff --check` clean.
- Remaining scientific limit: duplicate closure prevents double-counting. It does not add coverage beyond the existing MJOLNIR high/low parasitic-current coded target.

Ratchet update 2026-05-05, Auluck 2023 Generalized Plasma Focus scaling target:

- Modules touched: `src/dpf/validation/kr_targets.py`, `src/dpf/validation/__init__.py`, `tests/test_kr_targets.py`, `CortexFindings.md`, and `CodexFindings.md`.
- Reviewed `KnowledgeReference/the-generalized-plasma-focus-problem-and-its-application-to-space-propulsion-s-k-h-auluck.md` and added `auluck_gpf_scaling_theory_targets()`.
- The target records the paper as a theory/scaling and validation-requirements source, not a completed experimental benchmark. It explicitly captures the source's warning that conventional DPF fusion output is complex and not fully understood, neutron-yield scaling failure is experimentally observed, and no conventional-DPF workaround is available in the paper.
- Extracted values include a 20 kV laboratory example, `43 uF`, `160 kA`, `8.6 kJ`, `8.45 us` quarter period, hydrogen density `0.00342 kg/m^3` or about `43 mbar`, power-density amplification about `9000`, magnetic field rise from `20 T` to about `200 T` in about `40 ns`, wire current about `80 kA`, current density `1.8e12 A/m^2`, wire travel time about `8.4 ns`, radial Alfven transit about `17 ns`, explosion timescale about `3 ps`, jet Alfven velocity about `1450 m/s`, and impulse about `0.002 kg m/s`.
- The target also records required validation work from the source: plasma voltage/current measurement, inductance-variation comparison, profile sweeps, jet momentum/velocity measurement, energy-deposition validation, gas-distribution/breakdown validation, and separate deuterium-tube neutron tests.
- Current corpus closure after this ratchet: 23 coded KR target records from 19 unique local KR source files; 18 of 54 DPF-named markdown files are represented by coded targets; 6 additional DPF-named markdown files are review-closed by explicit decisions; 30 DPF-named markdown files remain unreviewed.
- Updated triage counts after removing this source from the review queue: 14 circuit waveform candidates, 14 phase timing candidates, 9 spatial density candidates, 15 spatial magnetic/EM candidates, 24 spatial temperature candidates, 24 neutron validation candidates, and 1 uncertainty candidate.
- KnowledgeReference basis: lines `29-48`, `92-150`, `152-188`, `190-221`, `2123-2244`, `2249-2417`, `5367-5531`, `5796-5845`, `5957-5978`, `6248-6336`, and `6346-6369` in the local Auluck 2023 GPF paper.
- Verification status: `python3 -m py_compile src/dpf/validation/kr_targets.py src/dpf/validation/__init__.py tests/test_kr_targets.py` passed; `python3 -m pytest tests/test_kr_targets.py tests/test_kr_corpus.py -q` passed (`57 passed in 0.60s`); broad post-ratchet sweep passed with `108 passed in 0.64s` for quality/KR tests, `87 passed, 3 skipped in 1.54s` for MHD/physics/UQ tests, and `git diff --check` clean.
- Remaining scientific limit: this source helps define first-principles scaling requirements and blocks unvalidated propulsion/neutron-source claims, but it does not provide same-shot measured current, phase, spatial, neutron, or uncertainty data for predictive DPF validation.

Ratchet update 2026-05-05, Sandia 2009 ALEGRA-HEDP MHD target:

- Modules touched: `src/dpf/validation/kr_targets.py`, `src/dpf/validation/__init__.py`, `tests/test_kr_targets.py`, `CortexFindings.md`, and `CodexFindings.md`.
- Reviewed `KnowledgeReference/unlimited-release-printed-september-2009-alegra-hedp-simulations-of-the-dense-plasma-focus.md` and added `alegra_hedp_dpf_mhd_validation_targets()`.
- The target records the report as a partial early-MHD/circuit benchmark and a neutron-yield scope limiter. It captures that MHD can reproduce early current/timing/sheath behavior but only predicts the thermonuclear neutron component and must stop when charge separation and instabilities invalidate the approximation.
- Extracted values include Bernard Long `135 uF`, `20 kV`, `27 kJ`, `27 nH`, `3.3 mOhm`, `3 Torr`, experiment/simulation peak current `0.6 MA`/`0.5-0.6 MA`, neutron yield `1.5e9` experiment vs `1.2e5` ALEGRA; Bernard Short `120 uF`, `40 kV`, `96 kJ`, `10 Torr`, peak current `1.5 MA` experiment and ALEGRA, neutron yield `3e10` experiment vs `1.5e6` ALEGRA; and Tallboy `216 uF`, `50 kV`, `270 kJ`, `50 nH`, peak current `2.3 MA` experiment vs `1.8 MA` ALEGRA, neutron yield `3.5e11` experiment vs `3.7e7` ALEGRA.
- The target also records generic pinch scale `1 mm` by a few mm, `1e19-1e20 cm^-3`, Bernard Long measured density `1e18-5e19 cm^-3`, simulated density `1.4e19 cm^-3`, pre-pinch ion/electron temperature agreement, unresolved simulated `9 keV` pinch ion temperature, QEOS/SESAME low-density limitations, arbitrary `1 eV` seed layer, `0.5 mm` cell size, and the 3D/PIC follow-up requirement.
- Current corpus closure after this ratchet: 24 coded KR target records from 20 unique local KR source files; 19 of 54 DPF-named markdown files are represented by coded targets; 6 additional DPF-named markdown files are review-closed by explicit decisions; 29 DPF-named markdown files remain unreviewed.
- Updated triage counts after removing this source from the review queue: 13 circuit waveform candidates, 13 phase timing candidates, 8 spatial density candidates, 14 spatial magnetic/EM candidates, 23 spatial temperature candidates, 23 neutron validation candidates, and 1 uncertainty candidate.
- KnowledgeReference basis: lines `118-131`, `251-261`, `265-301`, `305-326`, `331-341`, `347-387`, `399-459`, `470-522`, `523-547,590-597`, and `549-577` in the local Sandia ALEGRA-HEDP report.
- Verification status: `python3 -m py_compile src/dpf/validation/kr_targets.py src/dpf/validation/__init__.py tests/test_kr_targets.py` passed; `python3 -m pytest tests/test_kr_targets.py tests/test_kr_corpus.py -q` passed (`58 passed in 0.62s`); broad post-ratchet sweep passed with `109 passed in 0.60s` for quality/KR tests, `87 passed, 3 skipped in 1.51s` for MHD/physics/UQ tests, and `git diff --check` clean.
- Remaining scientific limit: this target validates only early MHD/circuit behavior. It explicitly does not validate total neutron yield, beam-target production, neutron timing, spectrum, anisotropy, detector response, or kinetic post-pinch evolution.

Ratchet update 2026-05-05, Auluck 2021 circuit-element/Poynting target:

- Modules touched: `src/dpf/validation/kr_targets.py`, `src/dpf/validation/__init__.py`, `tests/test_kr_targets.py`, `CortexFindings.md`, and `CodexFindings.md`.
- Reviewed `KnowledgeReference/auluck-2021-dpf-circuit-element.md` and added `auluck_circuit_element_poynting_targets()`.
- The target encodes the source's core warning that DPF post-stagnation circuit behavior cannot be reduced to a scalar time-varying inductance unless 3D magnetic/velocity structures and Poynting-power terms are modeled or bounded. The unaccounted difference appears as anomalous impedance.
- Extracted values include PF-1000 magnetic probe radii `40`, `13`, and `0 mm`, probe height `10 mm`, interferogram intervals `10-15 ns`, current-carrying layer thickness `1.6-2.6 cm`, sheath velocity about `2.1e5 m/s` with `25%` shot-to-shot variation, density fall by at least two orders within less than `1 mm`, illustrative probe times `-68`, `-38`, and `22 ns`, diagnostic propagation delay `10-20 ns` over about `2 m`, and neon current-derivative minimum more than `200 ns` after dense-column breakup.
- Encoded requirements include terminal voltage from the volume field-power integral, circuit power accounting for all chamber processes, anomalous impedance for terms without scalar-circuit analogs, 3D magnetic and velocity diagnostics, and volume-integrated `J dot E`.
- Current corpus closure after this ratchet: 25 coded KR target records from 21 unique local KR source files; 20 of 54 DPF-named markdown files are represented by coded targets; 6 additional DPF-named markdown files are review-closed by explicit decisions; 28 DPF-named markdown files remain unreviewed.
- Updated triage counts after removing this source from the review queue: 13 circuit waveform candidates, 12 phase timing candidates, 7 spatial density candidates, 13 spatial magnetic/EM candidates, 23 spatial temperature candidates, 22 neutron validation candidates, and 0 uncertainty candidates.
- KnowledgeReference basis: lines `26-41`, `44-70`, `72-119`, `121-149`, `151-201`, `211-224`, `762-786,788-910`, `950-1019`, and `1021-1045` in the local Auluck 2021 circuit-element paper.
- Verification status: `python3 -m py_compile src/dpf/validation/kr_targets.py src/dpf/validation/__init__.py tests/test_kr_targets.py` passed; `python3 -m pytest tests/test_kr_targets.py tests/test_kr_corpus.py -q` passed (`59 passed in 0.56s`); broad post-ratchet sweep passed with `110 passed in 0.64s` for quality/KR tests, `87 passed, 3 skipped in 1.53s` for MHD/physics/UQ tests, and `git diff --check` clean.
- Remaining scientific limit: this is a circuit-field coupling constraint, not a complete validation packet. It still needs digitized `dI/dt`/voltage traces, 3D field and velocity measurements, volume-integrated `J dot E`, and neutron-response linkage.

Ratchet update 2026-05-05, Esaulov 2003 2D MHRDR DPF target:

- Modules touched: `src/dpf/validation/kr_targets.py`, `src/dpf/validation/__init__.py`, `tests/test_kr_targets.py`, `CortexFindings.md`, and `CodexFindings.md`.
- Reviewed `KnowledgeReference/esaulov_2003_2d_mhd_dpf.md` and added `esaulov_2d_mhrdr_dpf_targets()`.
- The target records a partial LANL Begay DPF 2D multi-temperature MHD context, including MHRDR model ingredients and thermal neutron-rate computation with Maxwell-averaged D-D cross sections.
- Extracted values include inner/outer electrode radii `1.18 cm`/`3.65 cm`, inner electrode length `15.7 cm`, deuterium fill `1 Torr`, capacitance `36.4 uF`, charging voltage `14 kV`, series inductance `178 nH`, formation examples `0.9` and `2.0 us`, acceleration slices `1.0` and `2.0 us`, collapse contours `2.6` and `2.65 us`, local neutron-rate peaks `2.74` and `2.92 us`, current during acceleration `50-100 kA`, voltage drop `1-2 kV`, abstract density above `1e19 cm^-3`, and axis-history temperature scale to `5 keV`.
- Current corpus closure after this ratchet: 26 coded KR target records from 22 unique local KR source files; 21 of 54 DPF-named markdown files are represented by coded targets; 6 additional DPF-named markdown files are review-closed by explicit decisions; 27 DPF-named markdown files remain unreviewed.
- Updated triage counts after removing this source from the review queue: 12 circuit waveform candidates, 12 phase timing candidates, 6 spatial density candidates, 12 spatial magnetic/EM candidates, 22 spatial temperature candidates, 21 neutron validation candidates, and 0 uncertainty candidates.
- KnowledgeReference basis: lines `26-49`, `79-121`, `200-257`, `272-314`, `334-459`, `461-529`, `623-660,664-724`, `727-791`, and `830-879` in the local Esaulov 2003 paper.
- Verification status: `python3 -m py_compile src/dpf/validation/kr_targets.py src/dpf/validation/__init__.py tests/test_kr_targets.py` passed; `python3 -m pytest tests/test_kr_targets.py tests/test_kr_corpus.py -q` passed (`60 passed in 0.58s`); broad post-ratchet sweep passed with `111 passed in 0.57s` for quality/KR tests, `87 passed, 3 skipped in 1.43s` for MHD/physics/UQ tests, and `git diff --check` clean.
- Remaining scientific limit: this is thermal-MHD context, not a complete neutron validation packet. It lacks digitized current/voltage traces, uncertainty, absolute neutron yield, neutron timing, spectrum, anisotropy, detector response, and kinetic beam-target closure.

Ratchet update 2026-05-05, FAETON-I 2025 high-voltage DPF target:

- Modules touched: `src/dpf/validation/kr_targets.py`, `src/dpf/validation/__init__.py`, `tests/test_kr_targets.py`, `CortexFindings.md`, and `CodexFindings.md`.
- Reviewed `KnowledgeReference/faeton-i-investigation-of-plasma-dynamics-and-radiation-output-of-a-100-kv-plasma-focus-device.md` and added `faeton_i_high_voltage_dpf_targets()`.
- The local markdown extraction only exposes the references/conclusion/Table 3 region, so the target is explicitly partial and must not be used as a complete validation packet.
- Extracted Table 3 values include shot `1062`: `fcr=0.4`, `fcr2=0.35`, `Vp=37.3 kV`, code yield `2.77e9`, measured yield `3e9`; shot `1036`: `fcr=0.72`, `fcr2=0.35`, `Vp=101.4 kV`, code yield `2.54e10`, measured yield `2.21e10`; shot `1027`: `fcr=0.8`, `fcr2=0.58`, `Vp=160.5 kV`, code yield `5.5e10`, measured yield `5.44e10`; and shot `895`: `fcr=0.9`, `fcr2=0.7`, `Vp=194 kV`, code yield `4.1e10`, measured yield `6e10`.
- Encoded interpretation constraints: `fcr=0.7` indicates good current sheath formation, exceptional shots are `fcr=0.8-0.9`, peak inductive voltage `Vmax` is preferred over current-dip severity for high-voltage large PF devices with restrikes, and the voltage peak is pre-stagnation and dynamics-induced.
- Encoded diagnostics and neutron evidence: consistent D-D yield `2.5e10` over five shots without gas refill, exceptional D-D yield up to `8e10`, forward anisotropy factor `1.6`, neutron energy peak `2.5 MeV` with `0.3 MeV` uncertainty, PMT scintillators at `5`, `10`, `20`, and `40 m`, `40 m` nTOF, `30 cm` lead shielding for gamma photons above `3 MeV`, and Faraday-cup deuteron energy about `350 keV`.
- D-T Faeton-X values were recorded only as projections, not validation targets: `2e14` neutrons for `65 kV`, `1 MJ`, `4 MA`; and `2e15` neutrons for `150 kV`, `5 MJ`, `7 MA`.
- Current corpus closure after this ratchet: 27 coded KR target records from 23 unique local KR source files; 22 of 54 DPF-named markdown files are represented by coded targets; 6 additional DPF-named markdown files are review-closed by explicit decisions; 26 DPF-named markdown files remain unreviewed.
- Updated triage counts after removing this source from the review queue: 11 circuit waveform candidates, 11 phase timing candidates, 6 spatial density candidates, 11 spatial magnetic/EM candidates, 21 spatial temperature candidates, 20 neutron validation candidates, and 0 uncertainty candidates.
- KnowledgeReference basis: lines `11-46,191-195`, `47-55`, `56-62`, `63-73`, `74-85`, and `90-99` in the local FAETON-I 2025 paper extract.
- Verification status: `python3 -m py_compile src/dpf/validation/kr_targets.py src/dpf/validation/__init__.py tests/test_kr_targets.py` passed; `python3 -m pytest tests/test_kr_targets.py tests/test_kr_corpus.py -q` passed (`61 passed in 0.57s`); broad post-ratchet sweep passed with `112 passed in 0.62s` for quality/KR tests, `87 passed, 3 skipped in 1.51s` for MHD/physics/UQ tests, and `git diff --check` clean.
- Remaining scientific limit: this target adds high-voltage waveform/yield/spectrum/anisotropy/detector-response constraints, but it still lacks digitized current and voltage traces, absolute phase times, same-shot spatial density/temperature/magnetic-field diagnostics, full detector response and calibration uncertainty, full neutron histories/spectra, and complete shot data.

Ratchet update 2026-05-05, Lee/RADPF theory model-scope target:

- Modules touched: `src/dpf/validation/kr_targets.py`, `src/dpf/validation/__init__.py`, `tests/test_kr_targets.py`, `CortexFindings.md`, and `CodexFindings.md`.
- Reviewed `KnowledgeReference/lee_radpf_theory.md` and added `lee_radpf_theory_model_scope_targets()`.
- The source was classified as a reduced-model theoretical-scope target, not as same-shot experimental validation evidence.
- Encoded model structure: circuit and sheath motion are coupled; the equation of motion is current-driven and the circuit equation depends on sheath motion/position; plasma resistance is ignored for electromagnetic drive; and the axial/radial tube voltage is treated as inductive in the reduced model.
- Encoded phase assumptions: axial phase uses snowplow current-sheath trajectory/speed for current-profile fitting; radial phase replaces the singular thin-snowplow limit with a slug model where a shock front opens space for the magnetic piston; reflected shock begins when the inward radial shock reaches the axis; and breakup becomes an expanded uniform current column.
- Extracted timing and radiation constraints include `alpha` as electrical time over axial transit time, `alpha1` as axial transit over radial transit time, axial transit about `20` times radial shock transit, typical axial/radial characteristic time ratio about `40`, reflected-shock speed `0.3` of the on-axis inward radial shock speed, communication delay `(rp - rs) / SDS`, deuterium radiation-collapse critical current `1.6 MA`, and neon/argon line-radiation critical current below `100 kA`.
- Encoded neutron-model limits: thermonuclear yield uses density, volume, thermal `sigma v`, and time; beam-target yield is phenomenological; beam deuterons are produced by diode action near the anode; beam voltage is tied to `Vmax`; the cross section uses beam energy `3 * Vmax`; the source reports code `Vmax` order `20-50 kV`, relevant experimental beam energy `50-150 keV`, lower-voltage range `30-60 keV`, empirical fit `Yn = 9e10 * Ipinch^3.8` for `0.1-1 MA`, and calibration point `0.5 MA`, `7e9` neutrons.
- Current corpus closure after this ratchet: 28 coded KR target records from 24 unique local KR source files; 23 of 54 DPF-named markdown files are represented by coded targets; 6 additional DPF-named markdown files are review-closed by explicit decisions; 25 DPF-named markdown files remain unreviewed.
- Updated triage counts after removing this source from the review queue: 10 circuit waveform candidates, 10 phase timing candidates, 6 spatial density candidates, 10 spatial magnetic/EM candidates, 20 spatial temperature candidates, 19 neutron validation candidates, and 0 uncertainty candidates.
- KnowledgeReference basis: lines `13-166`, `169-209,710-718`, `1268-1296`, `1401-1403,1443-1450`, `2312-2387`, `3292-3364`, `3763-4004`, `4048-4104`, and `5323-5344` in the local Lee/RADPF theory file.
- Verification status: `python3 -m py_compile src/dpf/validation/kr_targets.py src/dpf/validation/__init__.py tests/test_kr_targets.py` passed; `python3 -m pytest tests/test_kr_targets.py tests/test_kr_corpus.py -q` passed (`62 passed in 0.59s`); broad post-ratchet sweep passed with `113 passed in 0.63s` for quality/KR tests, `87 passed, 3 skipped in 1.58s` for MHD/physics/UQ tests, and `git diff --check` clean.
- Remaining scientific limit: this target defines what the Lee/RADPF reduced model assumes and calibrates. It does not validate same-shot current waveforms, phase endpoints, spatial profiles, detector response, neutron spectra/anisotropy, or independent beam-target yield calibration.

Ratchet update 2026-05-05, Blagoev 2025 electric-flux formation diagnostic target:

- Modules touched: `src/dpf/validation/kr_targets.py`, `src/dpf/validation/kr_corpus.py`, `src/dpf/validation/__init__.py`, `tests/test_kr_targets.py`, `tests/test_kr_corpus.py`, `CortexFindings.md`, and `CodexFindings.md`.
- Reviewed `KnowledgeReference/measurement-of-electric-flux-emission-a-new-diagnostic-for-the-dense-plasma-focus-a-b-blagoev12aa-v.md`, added `blagoev_electric_flux_diagnostic_targets()`, and closed the `-4` variant as a duplicate header/PDF-name extraction.
- Extracted device context: University of Sofia `3 kJ` Mather plasma focus, `20 uF`, up to `40 kV`, hollow copper anode `2 cm` diameter and `14.5 cm` length, six cathode rods with `0.8 cm` diameter and `16 cm` length on `3.5 cm` radius, chamber inner diameter `15.5 cm`, chamber height `35 cm`, and operation with air, argon, or deuterium.
- Extracted shot examples: shot `665` argon `0.95 Torr`, `19.0 kV`; shot `668` argon `0.83 Torr`, `19.1 kV`; shot `667` argon `0.77 Torr`, `19.0 kV`, with a reference singularity time `3.03 us`.
- Encoded diagnostic constraints: three symmetric identical D-dot probes, SMA central pins as floating conductors, `50 ohm` coax terminations, CH2/CH3/CH4 channels, `1 ns` sampling, `10` point smoothing, baseline-corrected integration, central-conductor calibration, voltage-divider resistances `1306 ohm` and `13.2 ohm`, applied voltage `5.34 kV`, integrated D-dot maxima within `3%` of mean, and `C1` estimate `0.006 pF`.
- Encoded phase/symmetry interpretation: current maximum marks end of rundown, current maximum-to-singularity is radial phase, lower pressure gives earlier singularity, formation/rundown D-dot similarity indicates adequate symmetry, radial-phase signal divergence indicates changing azimuthal behavior, hidden hollow-anode deformation can be detected by electric-flux asymmetry even when a pinch still forms, and Rogowski `dI/dt` can be contaminated by electric-flux pickup.
- Current corpus closure after this ratchet: 29 coded KR target records from 25 unique local KR source files; 24 of 54 DPF-named markdown files are represented by coded targets; 7 additional DPF-named markdown files are review-closed by explicit decisions; 23 DPF-named markdown files remain unreviewed.
- Updated triage counts after removing these sources from the review queue: 8 circuit waveform candidates, 8 phase timing candidates, 6 spatial density candidates, 8 spatial magnetic/EM candidates, 18 spatial temperature candidates, 17 neutron validation candidates, and 0 uncertainty candidates.
- KnowledgeReference basis: lines `30-44`, `47-62,83-95`, `102-116`, `117-170`, `174-194`, `367-412`, `426-455`, `466-497`, and `506-522,538-542` in the local Blagoev 2025 electric-flux diagnostic source.
- Verification status: `python3 -m py_compile src/dpf/validation/kr_targets.py src/dpf/validation/kr_corpus.py src/dpf/validation/__init__.py tests/test_kr_targets.py tests/test_kr_corpus.py` passed; `python3 -m pytest tests/test_kr_targets.py tests/test_kr_corpus.py -q` passed (`63 passed in 0.57s`); broad post-ratchet sweep passed with `114 passed in 0.59s` for quality/KR tests, `87 passed, 3 skipped in 1.48s` for MHD/physics/UQ tests, and `git diff --check` clean.
- Remaining scientific limit: this target constrains formation symmetry and diagnostic interpretation only. It lacks digitized probe/current traces, per-point uncertainty, independent phase endpoint diagnostics, calibrated electric-field reconstruction, same-shot density/temperature/magnetic profiles, and same-shot neutron outputs.

Ratchet update 2026-05-05, Auluck 2024 poloidal magnetic-field dynamo target:

- Modules touched: `src/dpf/validation/kr_targets.py`, `src/dpf/validation/kr_corpus.py`, `src/dpf/validation/__init__.py`, `tests/test_kr_targets.py`, `tests/test_kr_corpus.py`, `CortexFindings.md`, and `CodexFindings.md`.
- Reviewed `KnowledgeReference/poloidal-magnetic-field-in-the-dense-plasma-focus.md`, added `auluck_poloidal_magnetic_field_targets()`, and closed `KnowledgeReference/poloidal-magnetic-field-in-the-dense-plasma-focus-5.md` as a duplicate header/PDF-name extraction.
- Encoded the source's magnetic-diagnostic warning: point measurement of axial magnetic field inside the plasma with a magnetic probe is rejected because the probe has finite `1-2 mm` spatial resolution, perturbs plasma flow/current, and forms a Langmuir sheath; Faraday-rotation Abel inversion does not apply to the axial component.
- Encoded the simple-dynamo hypothesis: a curved plasma armature in a geomagnetic seed field generates azimuthal electric field through generalized Ohm's law; the Hall term is neglected; the zero-resistivity limit is used; and magnetic Reynolds number is assumed much greater than one.
- Encoded GPF/GV and circuit implications: magnetic-field scaling `B0 = mu0 * I(t) / (2*pi*a*r_tilde)`, flux function evolution in Hamilton-Jacobi form, Mather-type GV surfaces resembling experimental plasma shapes, MHD current overestimation if the dynamo is omitted, possible azimuthal circulating current behind apparent current loss, and a Lee radial-current-fraction response to external axial-field sweep.
- Encoded proposed test requirements: a Helmholtz coil with DC variable polarity, uniform axial field over a small DPF, field amplitude no more than `2` times local geomagnetic field, monitoring current derivative/integrated current/poloidal flux emission, and looking for response near geomagnetic null. Nonuniform or excessively high applied fields are explicitly invalid tests.
- Encoded supporting observation: the Nikulin `2.5 kJ` plasma-focus cone was twisted rather than radially imploded; the source argues that a purely azimuthal magnetic field cannot produce the torque.
- Current corpus closure after this ratchet: 30 coded KR target records from 26 unique local KR source files; 25 of 54 DPF-named markdown files are represented by coded targets; 8 additional DPF-named markdown files are review-closed by explicit decisions; 21 DPF-named markdown files remain unreviewed.
- Updated triage counts after removing these sources from the review queue: 6 circuit waveform candidates, 8 phase timing candidates, 4 spatial density candidates, 6 spatial magnetic/EM candidates, 16 spatial temperature candidates, 15 neutron validation candidates, and 0 uncertainty candidates.
- KnowledgeReference basis: lines `80-110`, `113-141`, `145-163`, `166-192`, `260-292`, `317-390,415-456`, `478-528`, `529-590`, and `595-604,760-763` in the local Auluck 2024 poloidal magnetic-field source.
- Verification status: `python3 -m py_compile src/dpf/validation/kr_targets.py src/dpf/validation/kr_corpus.py src/dpf/validation/__init__.py tests/test_kr_targets.py tests/test_kr_corpus.py` passed; `python3 -m pytest tests/test_kr_targets.py tests/test_kr_corpus.py -q` passed (`64 passed in 0.57s`); broad post-ratchet sweep passed with `115 passed in 0.58s` for quality/KR tests, `87 passed, 3 skipped in 1.47s` for MHD/physics/UQ tests, and `git diff --check` clean.
- Remaining scientific limit: this target blocks purely toroidal-field assumptions and defines a proposed experiment, but it lacks the completed field-sweep dataset, calibrated poloidal-flux signals, radial-current-fraction response, 3D field reconstruction, and same-shot neutron yield/anisotropy response.

Ratchet update 2026-05-05, Wante 2025 UNU/ICTP nitrogen-ion irradiation target:

- Modules touched: `src/dpf/validation/kr_targets.py`, `src/dpf/validation/__init__.py`, `tests/test_kr_targets.py`, `CortexFindings.md`, and `CodexFindings.md`.
- Reviewed `KnowledgeReference/regular-article-nitrogen-ion-irradiation-of-carbon-thin-lms-using-a-dense-plasma-focus-enhanced.md` and added `wante_nitrogen_ion_irradiation_targets()`.
- Classified the source as an ion-beam/material-processing validation target, not a neutron or end-to-end DPF validation target.
- Extracted UNU/ICTP PF configuration: nominal `3.0 kJ` device operated at `2.54 kJ`, `30 uF`, `13 kV`, `156 nH`, `21.4 mOhm`, anode radius `0.95 cm`, cathode radius `3.2 cm`, anode length `16 cm`, anode diameter `1.9 cm`, six copper cathode rods, Pyrex insulator, nitrogen purity `99.999%`, optimal pressure `1.5 mbar`, initial vacuum `5e-3 mbar`, four preliminary shots for stable pinch, sample distance `38 cm`, and irradiation shot counts `6`, `12`, and `24` at `5 min` intervals.
- Encoded diagnostics and Lee fit: Yokogawa `DL7480` current/voltage/ion acquisition, Faraday cup biased ion collector at `-45 V`, ion TOF from X-ray peak to ion peak, X-ray peak aligned with voltage peak, Lee fit parameters `fm=0.03`, `fc=0.7`, `fmr=0.18`, `fcr=0.85`, measured nitrogen ion energy `72.40 keV`, Lee model ion energy `71.0 keV`, ion flux `7.2e27 ions m^-2 s^-1`, and ion fluence `6.4e19 ions m^-2`.
- Encoded material response: nitrogen doping `7.06%`, `5.96%`, and `7.93%` for `6`, `12`, and `24` shots; deposition rates `1.18%`, `0.50%`, and `0.33%` per shot; copper impurity from anode ablation increasing to `2.11%` at `24` shots; fluorine content falling from `12.06%` to `4.94%`; crystallite size rising from `6.27 nm` to `11.16 nm`; new XRD peaks at `52` and `76` degrees; and interlayer spacing falling from `0.37 nm` to `0.340 nm`.
- Current corpus closure after this ratchet: 31 coded KR target records from 27 unique local KR source files; 26 of 54 DPF-named markdown files are represented by coded targets; 8 additional DPF-named markdown files are review-closed by explicit decisions; 20 DPF-named markdown files remain unreviewed.
- Updated triage counts after removing this source from the review queue: 5 circuit waveform candidates, 7 phase timing candidates, 3 spatial density candidates, 6 spatial magnetic/EM candidates, 15 spatial temperature candidates, 14 neutron validation candidates, and 0 uncertainty candidates.
- KnowledgeReference basis: lines `47-60`, `170-205`, `207-240`, `241-257,268-305`, `348-365,374-375`, `419-462`, and `605-637` in the local Wante 2025 nitrogen-ion irradiation source.
- Verification status: `python3 -m py_compile src/dpf/validation/kr_targets.py src/dpf/validation/__init__.py tests/test_kr_targets.py` passed; `python3 -m pytest tests/test_kr_targets.py tests/test_kr_corpus.py -q` passed (`65 passed in 0.49s`); broad post-ratchet sweep passed with `116 passed in 0.62s` for quality/KR tests, `87 passed, 3 skipped in 1.54s` for MHD/physics/UQ tests, and `git diff --check` clean.
- Remaining scientific limit: this target constrains ion beam energy/flux/fluence and material response only. It lacks digitized current/voltage/ion waveforms, Faraday-cup response uncertainty, absolute peak times, same-shot plasma profiles, and all neutron evidence.

Ratchet update 2026-05-05, Kiai 2025 double 3 MJ DPF/ICF concept target:

- Modules touched: `src/dpf/validation/kr_targets.py`, `src/dpf/validation/kr_corpus.py`, `src/dpf/validation/__init__.py`, `tests/test_kr_targets.py`, `tests/test_kr_corpus.py`, `CortexFindings.md`, and `CodexFindings.md`.
- Reviewed `KnowledgeReference/2025-double-3mj-dense-plasma-focus-thermonuclear-icf.md`, added `kiai_double_dpf_icf_concept_targets()`, and closed `KnowledgeReference/double-3-mj-dense-plasma-focus-for-thermonuclear-drive-inertial-confinement-fusion-5.md` plus `KnowledgeReference/double-3-mj-dense-plasma-focus-for-thermonuclear-drive-inertial-confinement-fusion.md` as duplicate extractions of the same paper.
- Classified the source as a theoretical double-DPF/ICF concept and validation roadmap, not as experimental validation evidence.
- Extracted full-scale design parameters: two `3 MJ` DPF banks, `6 MJ` total, deuterium at `10 torr`, `12.5 mOhm`, peak current `20 MA`, charging voltage `200 kV`, capacitance `150 uF`, inductance `35 nH`, circuit period `17.5 us`, anode radius `15 cm`, anode length `80 cm`, cathode radius `22.5 cm`, axial speed `29.5 cm/us`, radial speed `42.4 cm/us`, pinch radius `1.8 cm`, pinch lifetime `300 ns` each DPF, pinch length `12 cm`, current loss factor `0.7`, mass sweep factor `0.13`, and induced voltage `20 MV`.
- Extracted proposed `30 kJ` prototype parameters: `50-60 kV`, `500 uF`, plasma/deuteron density `6e25 ions/m^3`, projected fusion neutron yield `1e10 neutrons/shot`, pinch efficiency `20-30%`, peak current `3.54-4.24 MA`, maximum pinch current `0.71-1.06 MA`, pinch radius `3.0 mm`, pinch length `2.0 cm`, and pinch lifetime `50 ns`.
- Encoded HTS/power/pellet values as projections only: HTS field `10-15 T`, pellet ignition range `10-20 keV`, simplified with-HTS comparison `75 MW` fusion and `30 MW` electric output, without-HTS comparison `25 MW` fusion and `10 MW` electric output, and extreme pellet-model projection `3.61 PW` fusion and `613 TW` electric.
- Encoded the three-stage validation plan: single `30 kJ` DPF prototype, synchronized double `30 kJ` DPF, and full-scale fusion testing with plasma diagnostics, neutron-yield measurements, and high-speed imaging.
- Current corpus closure after this ratchet: 32 coded KR target records from 28 unique local KR source files; 27 of 54 DPF-named markdown files are represented by coded targets; 10 additional DPF-named markdown files are review-closed by explicit decisions; 17 DPF-named markdown files remain unreviewed.
- Updated triage counts after removing these sources from the review queue: 2 circuit waveform candidates, 7 phase timing candidates, 3 spatial density candidates, 3 spatial magnetic/EM candidates, 12 spatial temperature candidates, 11 neutron validation candidates, and 0 uncertainty candidates.
- KnowledgeReference basis: lines `16-40`, `83-106`, `392-453`, `464-530`, `942-996`, `997-1013`, `1200-1239,1275-1321,1349-1363,1373-1419`, `1478-1514`, and `1519-1584` in the local Kiai 2025 double-DPF/ICF source.
- Verification status: `python3 -m py_compile src/dpf/validation/kr_targets.py src/dpf/validation/kr_corpus.py src/dpf/validation/__init__.py tests/test_kr_targets.py tests/test_kr_corpus.py` passed; `python3 -m pytest tests/test_kr_targets.py tests/test_kr_corpus.py -q` passed (`66 passed in 0.57s`); broad post-ratchet sweep passed with `117 passed in 0.61s` for quality/KR tests, `87 passed, 3 skipped in 1.55s` for MHD/physics/UQ tests, and `git diff --check` clean.
- Remaining scientific limit: this is a theoretical proposal with future validation requirements. It lacks measured current/voltage traces, synchronized double-DPF phase timing, same-shot density/temperature/HTS-field profiles, DT pellet coupling diagnostics, measured neutron yield/timing/spectrum/anisotropy, detector response, full energy accounting, and validated `30 kJ` to `6 MJ` scaling.

Ratchet update 2026-05-05, Beresnyak 2018 HAWK 3D MHD model-scope target:

- Modules touched: `src/dpf/validation/kr_targets.py`, `src/dpf/validation/__init__.py`, `tests/test_kr_targets.py`, `CortexFindings.md`, and `CodexFindings.md`.
- Reviewed `KnowledgeReference/beresnyak_2018_dpf_hawk_simulations.md` and added `beresnyak_hawk_3d_mhd_targets()`.
- Classified the source as a HAWK-specific 3D MHD model-scope target, not as an experimental validation packet.
- Encoded HAWK setup: `665 kA` generator, `1.2 us` rise time, `720 nH` high-impedance generator inductance, local plasma injection by plasma guns into an evacuated interelectrode space, and fully ionized deuterium assumption.
- Encoded circuit coupling: `720 nH`, `0.15 ohm`, `1.07 uF`, initial capacitor voltage `640 kV`, zero initial current, current and `dI/dt` as simulation inputs, azimuthal magnetic boundary from current, velocity-gradient boundary from `dI/dt`, and device voltage from integrated electric field.
- Encoded geometry/injection: anode radius `6.33 cm`, anode length `4 cm`, cathode radius `8.57 cm`, high-to-low injected-density ratio `2`, background density `1/4 rho0`, azimuthal modes `m=0`, `m=3`, and `m=6`, and characteristic density `1e-7 g/cc` or `3e16 cm^-3`.
- Encoded phase/current behavior: pinch time `0.95 us` at `3e16 cm^-3`, target-density device voltage below `10 kV`, short-circuit sine period `5.2 us`, and example 3D grid `480 x 480 x 288`.
- Encoded model outputs and limits: total thermal-yield metric peaks at `9e15 cm^-3`, thermal fusion is subdominant and not a projected HAWK yield, Hall-MHD positive-polarity runs give faster/tighter pinch near the anode, Spitzer resistivity does not qualitatively change dynamics, maximum plasma temperature is about `3 keV`, and stochastic ion acceleration produces a mostly isotropic tail to about `200 keV`.
- Current corpus closure after this ratchet: 33 coded KR target records from 29 unique local KR source files; 28 of 54 DPF-named markdown files are represented by coded targets; 10 additional DPF-named markdown files are review-closed by explicit decisions; 16 DPF-named markdown files remain unreviewed.
- Updated triage counts after removing this source from the review queue: 2 circuit waveform candidates, 6 phase timing candidates, 2 spatial density candidates, 3 spatial magnetic/EM candidates, 11 spatial temperature candidates, 10 neutron validation candidates, and 0 uncertainty candidates.
- KnowledgeReference basis: lines `26-42`, `91-147`, `155-199`, `202-218`, `223-253`, `255-301`, `305-333`, and `338-389,393-398` in the local Beresnyak 2018 HAWK source.
- Verification status: `python3 -m py_compile src/dpf/validation/kr_targets.py src/dpf/validation/kr_corpus.py src/dpf/validation/__init__.py tests/test_kr_targets.py tests/test_kr_corpus.py` passed; `python3 -m pytest tests/test_kr_targets.py tests/test_kr_corpus.py -q` passed (`67 passed in 0.61s`); broad post-ratchet sweep passed with `118 passed in 0.64s` for quality/KR tests, `87 passed, 3 skipped in 1.62s` for MHD/physics/UQ tests, and `git diff --check` clean.
- Remaining scientific limit: HAWK experiments were planned, current disruption was not modeled, and the local extract lacks measured HAWK current/voltage traces, measured phase endpoints, same-shot spatial profile diagnostics, measured neutron yield/timing/spectrum/anisotropy, detector response, and uncertainty.

Ratchet update 2026-05-05, Wang/Yang 1999 DPF-16 metallic-vapor interferometry target:

- Modules touched: `src/dpf/validation/kr_targets.py`, `src/dpf/validation/__init__.py`, `tests/test_kr_targets.py`, `CortexFindings.md`, and `CodexFindings.md`.
- Reviewed `KnowledgeReference/observation-of-the-metallic-vapor-from-a-plasma-focus-wang-xinxin-3-yang-jinji-department-of.md` and added `wang_metallic_vapor_interferometry_targets()`.
- Classified the source as a qualitative interferometry/anode-material-vapor target, not as neutron or complete DPF validation evidence.
- Extracted DPF-16 operating context: `16 kJ`, `20 kV`, `380 kA`, Mather type, hydrogen fill pressure `70-650 Pa`, typical interferograms at `200 Pa`, and metallic-vapor development frames at `330 Pa`.
- Extracted geometry: oxygen-free copper anode, anode diameter `66 mm`, anode and cathode length `265 mm`, tungsten target `10 mm` diameter and `6 mm` high, and `60 mm` interferometer field of view.
- Encoded phase timing: `t=0` is the pinch spike in the `dI/dt` waveform and maximum compression above the anode; compression frames at `-200`, `-140`, and `-60 ns`; expansion beginning at `40 ns`; post-focus expansion at `200 ns`; metallic vapor visible at `280 ns`; and higher-pressure vapor frames at `220` and `300 ns`.
- Encoded evidence interpretation: the high-density volume emerges from the anode target after focus, target erosion supports material evaporation, the high-density volume is absent when a hollow anode replaces the solid target, and the delayed metallic plasma is linked to hard X-ray emission several hundred nanoseconds after focus in cited context.
- Current corpus closure after this ratchet: 34 coded KR target records from 30 unique local KR source files; 29 of 54 DPF-named markdown files are represented by coded targets; 10 additional DPF-named markdown files are review-closed by explicit decisions; 15 DPF-named markdown files remain unreviewed.
- Updated triage counts after removing this source from the review queue: 2 circuit waveform candidates, 5 phase timing candidates, 1 spatial density candidate, 3 spatial magnetic/EM candidates, 10 spatial temperature candidates, 9 neutron validation candidates, and 0 uncertainty candidates.
- KnowledgeReference basis: lines `15-21`, `27-71`, `92-121`, `123-178`, `178-214`, `219-228`, and `230-234,266-271` in the local Wang/Yang 1999 source.
- Verification status: `python3 -m py_compile src/dpf/validation/kr_targets.py src/dpf/validation/kr_corpus.py src/dpf/validation/__init__.py tests/test_kr_targets.py tests/test_kr_corpus.py` passed; `python3 -m pytest tests/test_kr_targets.py tests/test_kr_corpus.py -q` passed (`68 passed in 0.56s`); broad post-ratchet sweep passed with `119 passed in 0.60s` for quality/KR tests, `87 passed, 3 skipped in 1.55s` for MHD/physics/UQ tests, and `git diff --check` clean.
- Remaining scientific limit: this source lacks digitized `dI/dt`, current, voltage, interferogram phase shift, density inversion, vapor-species spectroscopy, X-ray time history/spectrum, electron beam energy/current, neutron diagnostics, detector response, and uncertainty.

Ratchet update 2026-05-05, Altarabulsi 2024 deuteron-beam fluence target:

- Modules touched: `src/dpf/validation/kr_targets.py`, `src/dpf/validation/__init__.py`, `tests/test_kr_targets.py`, `CortexFindings.md`, and `CodexFindings.md`.
- Reviewed `KnowledgeReference/original-deuteron-beam-fluence-emitted-from-dense-plasma-focus.md` and added `altarabulsi_deuteron_beam_fluence_targets()`.
- Classified the source as a Lee-code deuteron-beam fluence validation target for material-processing/ion-beam scope, not as neutron validation.
- Encoded three fitted devices: PF-1000 (`863.1 kJ`), MPEF-12 kJ (`9.7 kJ`), and PF-2.7 kJ (`2.7 kJ`) operated in deuterium using `RADPFV6.16FIB`.
- Encoded Table 1 device parameters, current-waveform fitting requirements, and the MPEF-12 fit scope to the end of pinch at about `2.08 us`.
- Encoded Table 3 fluence comparisons: PF-1000 at `14 cm`, `0.5 Torr`, simulated `7.3e19 ions/m^2` versus measured about `7.5e19`; MPEF-12 kJ at `14 cm`, pressures `0.76-7.5 Torr`, simulated `5.5e18-7.5e18` versus measured values with errors; and PF-2.7 kJ at `40 cm`, pressures `0.075-0.6 Torr`, simulated `1.77e15-4.94e15` versus measured values with errors.
- Encoded distance scaling: pinch-exit fluence order `1e20 ions/m^2`, `14 cm` fluence order `1e19 ions/m^2`, PF-24 at `11 Torr` with pinch-exit fluence `3.87e20 ions/m^2`, flux dropping from `8.7e27` to `2.61e26 ions/m^2/s` at `26 cm`, and energy flux dropping from `1.37e14` to `4.09e12 W/m^2`.
- Current corpus closure after this ratchet: 35 coded KR target records from 31 unique local KR source files; 30 of 54 DPF-named markdown files are represented by coded targets; 10 additional DPF-named markdown files are review-closed by explicit decisions; 14 DPF-named markdown files remain unreviewed.
- Updated triage counts after removing this source from the review queue: 1 circuit waveform candidate, 4 phase timing candidates, 1 spatial density candidate, 3 spatial magnetic/EM candidates, 9 spatial temperature candidates, 8 neutron validation candidates, and 0 uncertainty candidates.
- KnowledgeReference basis: lines `30-42`, `52-132`, `181-288`, `300-389,1461-1480`, `390-489`, `523-628`, `701-780,1515-1545`, `631-688,880-907`, and `909-943` in the local Altarabulsi 2024 source.
- Verification status: `python3 -m py_compile src/dpf/validation/kr_targets.py src/dpf/validation/kr_corpus.py src/dpf/validation/__init__.py tests/test_kr_targets.py tests/test_kr_corpus.py` passed; `python3 -m pytest tests/test_kr_targets.py tests/test_kr_corpus.py -q` passed (`69 passed in 0.57s`); broad post-ratchet sweep passed with `120 passed in 0.61s` for quality/KR tests, `87 passed, 3 skipped in 1.58s` for MHD/physics/UQ tests, and `git diff --check` clean.
- Remaining scientific limit: this target depends on fitted current waveforms and published fluence tables. It lacks raw digitized current/voltage waveforms, raw fluence detector response, raw detector calibration, same-shot density/temperature/beam divergence diagnostics, complete uncertainty propagation, and neutron timing/spectrum/anisotropy validation.

Ratchet update 2026-05-05, Narkis/Hahn 2021 Kr-doped Gemini-like DPF MHD target:

- Modules touched: `src/dpf/validation/kr_targets.py`, `src/dpf/validation/__init__.py`, `tests/test_kr_targets.py`, `CortexFindings.md`, and `CodexFindings.md`.
- Reviewed `KnowledgeReference/seyler-2021-kr-doped-dpf-mhd.md` and added `narkis_kr_doped_dpf_mhd_targets()`.
- Classified the source as a 2D radiation-MHD scope target for Kr-doped, Gemini-like DPF simulations, not as predictive total-neutron-yield validation.
- Encoded the source's key scientific warning: fully kinetic simulations are required for pinch stagnation and total neutron yield, and MHD cannot capture kinetic effects or beam-target neutron production.
- Encoded setup: HYDRA quasi-2D `R-Z` geometry, `2-3 MA`, Kr fractions `0`, `0.1%`, and `1%`, charging voltages `35`, `40`, `45`, and `50 kV`, experimental current data only for `35` and `40 kV`, anode radius `7.62 cm`, cathode radius `10.16 cm`, anode length `43.18 cm`, cathode length `59.18 cm`, and near-cap mesh `200 x 200 um`.
- Encoded circuit/current limits: RLC circuit `R=1.4 mOhm`, `L=40 nH`, `C=432 uF`; resistance is a free parameter; fill pressure uses scale factor `0.75`; matching implosion time and peak current is a sanity check, not strict quantitative comparison; breakdown is neglected.
- Encoded Table I timing, temperature, and density values. Example: `1%` Kr, `50 kV` has `t=6.525 us`, `Ti=156 eV`, `Te=98.5 eV`, and `ni=15.87e18 cm^-3`.
- Encoded radiation and neutron results: approximate peak temperatures `6.7`, `8.3`, and `12.6 keV` for `0%`, `0.1%`, and `1%` Kr; thermonuclear yield order `1e9-1e10`; all-point scaling exponents `5.726`, `4.643`, and `4.859`; and `35 kV` maximum `dN/dt` values `1.1e9`, `2.4e9`, and `1.8e9 neutrons/ns`.
- Current corpus closure after this ratchet: 36 coded KR target records from 32 unique local KR source files; 31 of 54 DPF-named markdown files are represented by coded targets; 10 additional DPF-named markdown files are review-closed by explicit decisions; 13 DPF-named markdown files remain unreviewed.
- Updated triage counts after removing this source from the review queue: 0 circuit waveform candidates, 3 phase timing candidates, 1 spatial density candidate, 3 spatial magnetic/EM candidates, 8 spatial temperature candidates, 7 neutron validation candidates, and 0 uncertainty candidates.
- KnowledgeReference basis: lines `59-69`, `72-127`, `129-143,194-196`, `144-181`, `182-189`, `191-231,251-261,319-329`, `272-299,330-401`, `406-460`, and `488-518` in the local Narkis/Hahn 2021 source.
- Verification status: `python3 -m py_compile src/dpf/validation/kr_targets.py src/dpf/validation/kr_corpus.py src/dpf/validation/__init__.py tests/test_kr_targets.py tests/test_kr_corpus.py` passed; `python3 -m pytest tests/test_kr_targets.py tests/test_kr_corpus.py -q` passed (`70 passed in 0.83s`); broad post-ratchet sweep passed with `121 passed in 0.63s` for quality/KR tests, `87 passed, 3 skipped in 1.76s` for MHD/physics/UQ tests, and `git diff --check` clean.
- Remaining scientific limit: this target lacks strict digitized current/voltage trace fitting, measured phase endpoints for every voltage/dopant case, breakdown physics, 3D instability growth, species separation, fully kinetic stagnation, beam-target neutron production, detector response, and neutron spectrum/anisotropy validation.

Ratchet update 2026-05-05, Auluck 2022 DPF theory part-1 extraction decision:

- Modules touched: `src/dpf/validation/kr_corpus.py`, `tests/test_kr_corpus.py`, `CortexFindings.md`, and `CodexFindings.md`.
- Reviewed `KnowledgeReference/auluck-2022-dpf-theory-part1.md`.
- Classified the local file as `insufficient_extractable_validation_data`: the metadata says the PDF is a 74-page theory source with tables and figures, but the markdown extraction contains only the final references page.
- No target was added. The KR-only rule does not allow inferring equations, model claims, or validation numbers from a title and bibliography.
- Current corpus closure after this ratchet: 36 coded KR target records from 32 unique local KR source files; 31 of 54 DPF-named markdown files are represented by coded targets; 11 additional DPF-named markdown files are review-closed by explicit decisions; 12 DPF-named markdown files remain unreviewed.
- Updated triage counts after removing this source from the review queue: 0 circuit waveform candidates, 3 phase timing candidates, 0 spatial density candidates, 2 spatial magnetic/EM candidates, 8 spatial temperature candidates, 6 neutron validation candidates, and 0 uncertainty candidates.
- KnowledgeReference basis: lines `1-64` in the local Auluck 2022 part-1 extraction, which expose only metadata and references.
- Verification status: `python3 -m py_compile src/dpf/validation/kr_corpus.py tests/test_kr_corpus.py` passed; `python3 -m pytest tests/test_kr_targets.py tests/test_kr_corpus.py -q` passed (`70 passed in 0.49s`); broad post-ratchet sweep passed with `121 passed in 0.57s` for quality/KR tests, `87 passed, 3 skipped in 1.55s` for MHD/physics/UQ tests, and `git diff --check` clean.
- Remaining scientific limit: this source must be re-ingested from the original PDF before it can support any KR-only theory target.

Ratchet update 2026-05-05, Auluck 2023 neutron-yield scaling failure target:

- Modules touched: `src/dpf/validation/kr_targets.py`, `src/dpf/validation/__init__.py`, `tests/test_kr_targets.py`, `CortexFindings.md`, and `CodexFindings.md`.
- Reviewed `KnowledgeReference/on-the-failure-of-neutron-yield-scaling-in-the-dense-plasma-focus-s-k-h-auluck-international.md` and added `auluck_neutron_yield_scaling_failure_targets()`.
- Classified the source as a narrow theory/test target, not as a neutron validation dataset. Only the exposed conclusion and references were used.
- Encoded the core claim: large DPF devices can abruptly stop following expected neutron-yield scaling above some voltage because the device must satisfy drive-parameter limits and generalized optimization criteria.
- Encoded the insulator-radius scaling claim: reaction yield should vary as the inverse fifth power of the outer-insulator-radius to anode-radius ratio; reducing the ratio from typical `~1` to `~0.4` by placing the insulator in the shadow of the anode is claimed to raise yield by two orders of magnitude only if all optimization conditions are met simultaneously.
- Encoded the proposed tests: lift-off time versus drive parameter and insulator radius; pressure-range changes under add-on insulators; and tests with insulator outer radius below anode radius. The source says small-device studies should use lift-off timing, not neutron measurements, as the primary scaling-failure test.
- Current corpus closure after this ratchet: 37 coded KR target records from 33 unique local KR source files; 32 of 54 DPF-named markdown files are represented by coded targets; 11 additional DPF-named markdown files are review-closed by explicit decisions; 11 DPF-named markdown files remain unreviewed.
- Updated triage counts after removing this source from the review queue: 0 circuit waveform candidates, 3 phase timing candidates, 0 spatial density candidates, 1 spatial magnetic/EM candidate, 7 spatial temperature candidates, 5 neutron validation candidates, and 0 uncertainty candidates.
- KnowledgeReference basis: lines `13-19`, `21-35`, `36-43`, and `46-144` in the local Auluck scaling-failure source.
- Verification status: `python3 -m py_compile src/dpf/validation/kr_targets.py src/dpf/validation/__init__.py tests/test_kr_targets.py` passed; `python3 -m pytest tests/test_kr_targets.py tests/test_kr_corpus.py -q` passed (`71 passed in 0.58s`); broad post-ratchet sweep passed with `122 passed in 0.59s` for quality/KR tests, `87 passed, 3 skipped in 1.57s` for MHD/physics/UQ tests, and `git diff --check` clean.
- Remaining scientific limit: equations `12` and `17`, the derivation, and validation data are missing from the markdown. Full use requires PDF re-ingestion or another KR source exposing lift-off-time, pressure-range, drive-parameter, and neutron-yield sweeps.

Ratchet update 2026-05-05, Ou/FOI 2D dense plasma focus simulation target:

- Modules touched: `src/dpf/validation/kr_targets.py`, `src/dpf/validation/kr_corpus.py`, `src/dpf/validation/__init__.py`, `tests/test_kr_targets.py`, `tests/test_kr_corpus.py`, `CortexFindings.md`, and `CodexFindings.md`.
- Reviewed `KnowledgeReference/two-dimensional-simulation-of-dense-plasma-focus.md`, added `ou_foi_2d_dpf_simulation_targets()`, and closed `KnowledgeReference/two-dimensional-simulation-of-dense-plasma-focus-5.md` as a duplicate header/PDF-name variant.
- Encoded FOI 2D MHD scope: electron inertia ignored, simplified Ohm law, electromagnetic solver `TVD-CP`, fluid solver `RTVD`, adiabatic single-phase ideal gas, high-resistivity swept/vacuum region, fixed electrodes, Courant number `0.5`, and sine-current boundary `Imax * sin(2*pi*f*t)`.
- Encoded LLNL reference case: anode diameter `15.2 cm`, cathode-anode gap `4.3 cm`, peak current `2.5 MA`, fill pressure `2926 Pa`, sheath images at `3.9 us`, `6.2 us`, `7.4 us`, and breakup at `7.4 us`. The source says morphology agrees with LLNL optical framing images but timing differs greatly.
- Encoded parameter sweeps: current amplitudes `1.5-3.5 MA`, pinch times `188.99`, `155.08`, `135.65`, `123.40`, `114.29 ns`, quarter period `135 ns`, pinch currents `1.213-3.399 MA`, pressure sweep `133-2660 Pa`, anode radii `30-50 mm`, and cathode-anode gaps `15-35 mm`.
- Current corpus closure after this ratchet: 38 coded KR target records from 34 unique local KR source files; 33 of 54 DPF-named markdown files are represented by coded targets; 12 additional DPF-named markdown files are review-closed by explicit decisions; 9 DPF-named markdown files remain unreviewed.
- Updated triage counts after removing these sources from the review queue: 0 circuit waveform candidates, 1 phase timing candidate, 0 spatial density candidates, 1 spatial magnetic/EM candidate, 5 spatial temperature candidates, 3 neutron validation candidates, and 0 uncertainty candidates.
- KnowledgeReference basis: lines `14-34`, `35-100`, `101-113,134-237`, `247-252,288-301`, `254-282,303-326`, `283-287,330-389`, `335-349,391-471`, and `482-521` in the local FOI 2D DPF source.
- Verification status: `python3 -m py_compile src/dpf/validation/kr_targets.py src/dpf/validation/kr_corpus.py src/dpf/validation/__init__.py tests/test_kr_targets.py tests/test_kr_corpus.py` passed; `python3 -m pytest tests/test_kr_targets.py tests/test_kr_corpus.py -q` passed (`72 passed in 0.49s`); broad post-ratchet sweep passed with `123 passed in 0.64s` for quality/KR tests, `87 passed, 3 skipped in 1.65s` for MHD/physics/UQ tests, and `git diff --check` clean.
- Remaining scientific limit: this source lacks measured current/voltage traces, timing uncertainty, quantitative LLNL frame alignment, density/temperature/magnetic-field diagnostics, and neutron outputs.

Ratchet update 2026-05-05, Sun 2025 two-temperature MHD DPF motion target:

- Modules touched: `src/dpf/validation/kr_targets.py`, `src/dpf/validation/__init__.py`, `tests/test_kr_targets.py`, `CortexFindings.md`, and `CodexFindings.md`.
- Reviewed `KnowledgeReference/2025-theoretical-and-numerical-studies-on-motion-process-of-dense-plasma-focus.md` and added `sun_two_temperature_mhd_motion_targets()`.
- Encoded the KR-backed scope: two-temperature nonideal MHD coupled to an external RLC circuit, electron-ion thermal nonequilibrium, Braginskii transport coefficients, resistive effects, and qualitative/plot-based benchmark comparisons against UNU current/voltage and UDMPF1 radial trajectory.
- Encoded UNU circuit and geometry: `15 kV`, `30 uF`, `110 nH`, `12 mOhm`, anode radius `0.95 cm`, cathode radius `3.2 cm`, gap `2.25 cm`, anode length `16 cm`, and cathode length `25 cm`.
- Encoded motion targets: axial phase `0-2.5 us`, radial implosion `2.78-2.90 us`, pinch around `2.8 us`, background density `2.4e23 m^-3`, background pressure about `3.5 Torr`, axial sheath speed up to `90 km/s`, axial ion temperature rise from `1` to `100 eV`, radial density about `1e24 m^-3`, and radial ion temperature about `1 keV`.
- Encoded design-scaling claims: large-DPF current saturates when increasing capacitance or decreasing inductance; increasing voltage is more effective; and the anode-to-cathode radius ratio should be small, with PF-1000 `c` cases `1.4`, `1.8`, `2.2`, and `2.6`.
- Current corpus closure after this ratchet: 39 coded KR target records from 35 unique local KR source files; 34 of 54 DPF-named markdown files are represented by coded targets; 12 additional DPF-named markdown files are review-closed by explicit decisions; 8 DPF-named markdown files remain unreviewed.
- Updated triage counts after removing this source from the review queue: 0 circuit waveform candidates, 1 phase timing candidate, 0 spatial density candidates, 1 spatial magnetic/EM candidate, 4 spatial temperature candidates, 2 neutron validation candidates, and 0 uncertainty candidates.
- KnowledgeReference basis: lines `28-34`, `83-172`, `423-481`, `484-548`, `552-623`, `626-755`, `755-832`, `914-996`, and `1000-1017` in the local Sun 2025 source.
- Verification status: `python3 -m py_compile src/dpf/validation/kr_targets.py src/dpf/validation/kr_corpus.py src/dpf/validation/__init__.py tests/test_kr_targets.py tests/test_kr_corpus.py` passed; `python3 -m pytest tests/test_kr_targets.py tests/test_kr_corpus.py -q` passed (`73 passed in 0.55s`); semantic/source audits passed; broad post-ratchet sweep passed with `124 passed in 0.66s` for quality/KR tests, `87 passed, 3 skipped in 1.83s` for MHD/physics/UQ tests, and `git diff --check` clean.
- Remaining scientific limit: this source strengthens macroscopic MHD motion, phase, temperature, and design-scaling targets, but it explicitly says MHD cannot self-consistently resolve high-energy particle beams or neutron production. It also lacks digitized current/voltage traces, quantified error bars, density/temperature profile uncertainty, and neutron validation outputs.

Ratchet update 2026-05-05, Demina/Gribkov DPF material-damage irradiation target:

- Modules touched: `src/dpf/validation/kr_targets.py`, `src/dpf/validation/__init__.py`, `tests/test_kr_targets.py`, `CortexFindings.md`, and `CodexFindings.md`.
- Reviewed `KnowledgeReference/application-of-a-plasma-accelerator-of-the-dense-plasma-focus-type-in-simulation-of-radiation.md` and added `demina_dpf_material_damage_targets()`.
- Classified this source as an application-response/material-damage target, not as core DPF machine validation.
- Encoded the devices and irradiation conditions: PF-5M `5 kJ`, PF-6 `7 kJ`, PF-1000 `1.2 MJ`, PF-1000 experiment energy about `600 kJ`, deuterium fill at `470 Pa`, power flux `1e7-1e10 W/cm2`, pulse duration `0.2-1 us`, `10` W/W-CFC pulses, and `5` CFC/SiC pulses.
- Encoded tungsten damage targets: melting, evaporation, wavelike relief, nanoscale cellular structure near `1e10 W/cm2`, microcracks above `1e8 W/cm2`, bubble size around `1 um`, microcrack penetration around `10 um`, and erosion table entries including about `2.05 um` per pulse for the highest ion/plasma-stream condition.
- Encoded CFC/CFC-SiC response: W droplets/ridges on CFC, stronger evaporation for fibers normal to the irradiated surface, lower erosion for fibers parallel to the surface, CFC-8SiC evaporated layer `2.6 um` per shot at `1e9 W/cm2`, and CFC-40SiC `1.9 um` per shot.
- Encoded redeposition limits: Cu/O/Fe/Cr on W, Fe/Cr/Si/Cu on CFC-SiC, steel holder and copper anode sources, and possible surface compounds `Fe2C`, `Fe5C2`, `Cu4Si`, and `(Cr,Fe)7C3`.
- Current corpus closure after this ratchet: 40 coded KR target records from 36 unique local KR source files; 35 of 54 DPF-named markdown files are represented by coded targets; 12 additional DPF-named markdown files are review-closed by explicit decisions; 7 DPF-named markdown files remain unreviewed.
- Updated triage counts after removing this source from the review queue: 0 circuit waveform candidates, 1 phase timing candidate, 0 spatial density candidates, 1 spatial magnetic/EM candidate, 3 spatial temperature candidates, 1 neutron validation candidate, and 0 uncertainty candidates.
- KnowledgeReference basis: lines `24-43`, `47-61`, `67-84`, `90-150`, `180-223`, `381-386`, `152-166`, `225-262`, `388-393`, `289-320`, and `322-342` in the local material-damage source.
- Verification status: `python3 -m py_compile src/dpf/validation/kr_targets.py src/dpf/validation/kr_corpus.py src/dpf/validation/__init__.py tests/test_kr_targets.py tests/test_kr_corpus.py` passed; `python3 -m pytest tests/test_kr_targets.py tests/test_kr_corpus.py -q` passed (`74 passed in 0.69s`); semantic/source audits passed; broad post-ratchet sweep passed with `125 passed in 0.61s` for quality/KR tests, `87 passed, 3 skipped in 1.63s` for MHD/physics/UQ tests, and `git diff --check` clean.
- Remaining scientific limit: this source can bound material erosion/redeposition under DPF-driven loads, but it does not provide current/voltage waveforms, incident particle spectra, sample-distance tables by condition, same-shot plasma profiles, neutron observables, or uncertainty budgets.

Ratchet update 2026-05-05, Unity front-end guide review decision:

- Modules touched: `src/dpf/validation/kr_corpus.py`, `tests/test_kr_corpus.py`, `CortexFindings.md`, and `CodexFindings.md`.
- Reviewed `KnowledgeReference/building-a-sci-fi-themed-dense-plasma-focus-simulation-front-end-in-unity.md`.
- Classified it as `non_scientific_frontend_guide`: the file is a Unity/URP/VFX Graph/raymarching/UI/WebSocket tutorial for displaying simulation data, not a verified DPF physics source.
- No validation target was added. The document may inform visualization UX, but it cannot be used as KR-only scientific evidence for equations, experimental targets, diagnostics, or model validation.
- Current corpus closure after this ratchet: 40 coded KR target records from 36 unique local KR source files; 35 of 54 DPF-named markdown files are represented by coded targets; 13 additional DPF-named markdown files are review-closed by explicit decisions; 6 DPF-named markdown files remain unreviewed.
- Updated triage counts after removing this source from the review queue: 0 circuit waveform candidates, 1 phase timing candidate, 0 spatial density candidates, 0 spatial magnetic/EM candidates, 2 spatial temperature candidates, 1 neutron validation candidate, and 0 uncertainty candidates.
- KnowledgeReference basis: lines `24-34`, `79-90`, `286-443`, `504-540`, `1011-1055`, and `1150-1209` show front-end setup, rendering, data ingestion, and visualization mechanics rather than DPF validation content.
- Verification status: `python3 -m py_compile src/dpf/validation/kr_targets.py src/dpf/validation/kr_corpus.py src/dpf/validation/__init__.py tests/test_kr_targets.py tests/test_kr_corpus.py` passed; `python3 -m pytest tests/test_kr_targets.py tests/test_kr_corpus.py -q` passed (`74 passed in 0.48s`); semantic/source audits passed; broad post-ratchet sweep passed with `125 passed in 0.61s` for quality/KR tests, `87 passed, 3 skipped in 1.57s` for MHD/physics/UQ tests, and `git diff --check` clean.
- Remaining scientific limit: this source provides no DPF physics validation data. It is review-closed only to keep the KR scientific queue accurate.

Ratchet update 2026-05-05, Lee 2014 radiative Lee-model review target:

- Modules touched: `src/dpf/validation/kr_targets.py`, `src/dpf/validation/__init__.py`, `tests/test_kr_targets.py`, `CortexFindings.md`, and `CodexFindings.md`.
- Reviewed `KnowledgeReference/lee-2014-plasma-focus-radiative-model.md` and added `lee_2014_radiative_model_review_targets()`.
- Encoded the peer-reviewed 5-phase Lee-model scope: axial snowplow, radial inward shock slug model, radial reflected shock, slow compression/pinch, expanded column, and optional Type-2 `Phase 4a` anomalous-resistance extension.
- Encoded equation-level timing/model targets: radial inward phase equation set `14,15,17,19`; reflected-shock equation set `34,35,36,37`; reflected-shock speed fraction `0.3`; axial phase ends when the current sheath reaches the anode end; radial inward phase ends when the shock reaches axis; pinch phase ends after one small-disturbance transit time.
- Encoded radiative-pinch physics: Joule heating, Spitzer resistance, Bennett temperature, Bremsstrahlung, line radiation, total `dQ/dt`, self-absorption, surface-emission transition, radiation-collapse behavior, deuterium collapse current `1.6 MA`, and Ne/Ar collapse current below `100 kA`.
- Current corpus closure after this ratchet: 41 coded KR target records from 37 unique local KR source files; 36 of 54 DPF-named markdown files are represented by coded targets; 13 additional DPF-named markdown files are review-closed by explicit decisions; 5 DPF-named markdown files remain unreviewed.
- Updated triage counts after removing this source from the review queue: 0 circuit waveform candidates, 0 phase timing candidates, 0 spatial density candidates, 0 spatial magnetic/EM candidates, 1 spatial temperature candidate, 1 neutron validation candidate, and 0 uncertainty candidates.
- KnowledgeReference basis: lines `6-14`, `18-24`, `30-88`, `88-117`, `119-190`, `196-200`, `204-208`, and `212-216` in the local Lee 2014 source.
- Verification status: `python3 -m py_compile src/dpf/validation/kr_targets.py src/dpf/validation/kr_corpus.py src/dpf/validation/__init__.py tests/test_kr_targets.py tests/test_kr_corpus.py` passed; `python3 -m pytest tests/test_kr_targets.py tests/test_kr_corpus.py -q` passed (`75 passed in 0.74s`); semantic/source audits passed; broad post-ratchet sweep passed with `126 passed in 0.61s` for quality/KR tests, `87 passed, 3 skipped in 1.66s` for MHD/physics/UQ tests, and `git diff --check` clean.
- Remaining scientific limit: this is an equation/scope source, not an experimental validation packet. It does not provide measured waveforms, shock/piston trajectories, radiated-power traces, profile diagnostics, neutron observables, or uncertainty budgets, and the local extract explicitly omits equations `51`, `52`, and `53`.

Ratchet update 2026-05-05, Focus Fusion p-B11 correction-only decision:

- Modules touched: `src/dpf/validation/kr_corpus.py`, `tests/test_kr_corpus.py`, `CortexFindings.md`, and `CodexFindings.md`.
- Reviewed `KnowledgeReference/2023-correction-to-focus-fusion-overview-of-progress-towards-p-b11-fusion-with-the.md`.
- Classified it as `correction_only`: the one-page notice corrects the original Focus Fusion abstract to `nτT = 3.4e20 keV-s/m3`.
- No new target was added. The corrected `3.4e20 keV-s/m3` value is already encoded in `ff1_focus_fusion_plasmoid_targets()` from the canonical original Focus Fusion source.
- Current corpus closure after this ratchet: 41 coded KR target records from 37 unique local KR source files; 36 of 54 DPF-named markdown files are represented by coded targets; 14 additional DPF-named markdown files are review-closed by explicit decisions; 4 DPF-named markdown files remain unreviewed.
- Updated triage counts after removing this source from the review queue: 0 circuit waveform candidates, 0 phase timing candidates, 0 spatial density candidates, 0 spatial magnetic/EM candidates, 0 spatial temperature candidates, 1 neutron validation candidate, and 0 uncertainty candidates.
- KnowledgeReference basis: lines `21-33` in the local correction source.
- Verification status: `python3 -m py_compile src/dpf/validation/kr_targets.py src/dpf/validation/kr_corpus.py src/dpf/validation/__init__.py tests/test_kr_targets.py tests/test_kr_corpus.py` passed; `python3 -m pytest tests/test_kr_targets.py tests/test_kr_corpus.py -q` passed (`75 passed in 0.49s`); semantic/source audits passed; broad post-ratchet sweep passed with `126 passed in 0.61s` for quality/KR tests, `87 passed, 3 skipped in 1.60s` for MHD/physics/UQ tests, and `git diff --check` clean.
- Remaining scientific limit: the correction notice adds no independent DPF validation data. It only fixes a scalar abstract value in a target already represented by the canonical source.

Ratchet update 2026-05-05, McAlpine 2014 DPF/NRTA MCNP application target:

- Modules touched: `src/dpf/validation/kr_targets.py`, `src/dpf/validation/__init__.py`, `tests/test_kr_targets.py`, `tests/test_kr_corpus.py`, `CortexFindings.md`, and `CodexFindings.md`.
- Reviewed `KnowledgeReference/monte-carlo-simulations-of-neutron-resonance-transmission-analysis-with-the-dense-plasma-focus.md` and added `mcalpine_dpf_nrta_mcnp_targets()`.
- Classified it as a downstream neutron-resonance transmission analysis application target, not a DPF plasma validation packet.
- Encoded DPF source context: LLNL DPF D-D `2.45 MeV` neutrons, yield about `1e7`, simulated pulse duration `20-60 ns`, generic DPF yield `1e4-1e13` neutrons in `10-100 ns`, deuterium working gas with optional DT context, and kinetic simulations used to inform desired yield/pinch length.
- Encoded MCNP/NRTA setup: monoenergetic isotropic point source, `3 cm` polyethylene moderator, detector volume `2 m` away, assumed `3He` detector with `1/v` absorption postprocessing, inspection object about `180 cm3`, Gaussian DPF pulse FWHM `20 ns`, conventional ENG trapezoid `4 us`, and `1e10` source particles per simulation.
- Encoded application results: TOF slightly broadens resonances but preserves locations; DPF resolves resonances not detectable with ENG; comparable ENG measurement would take about a day while DPF can do it in a single pulse; depleted uranium, highly enriched uranium, plutonium, and lead were compared and distinguished.
- Updated the corpus triage test because, after this source was closed, the remaining unreviewed DPF-named files have no category-marker hits and should be treated as unclassified manual-review items rather than forced scientific candidates.
- Current corpus closure after this ratchet: 42 coded KR target records from 38 unique local KR source files; 37 of 54 DPF-named markdown files are represented by coded targets; 14 additional DPF-named markdown files are review-closed by explicit decisions; 3 DPF-named markdown files remain unreviewed.
- Updated triage counts after removing this source from the review queue: 0 circuit waveform candidates, 0 phase timing candidates, 0 spatial density candidates, 0 spatial magnetic/EM candidates, 0 spatial temperature candidates, 0 neutron validation candidates, and 0 uncertainty candidates.
- KnowledgeReference basis: lines `32-42`, `45-117`, `119-142`, `144-165`, `168-204`, `250-302`, `303-339`, `342-384`, and `387-406` in the local McAlpine 2014 source.
- Verification status: `python3 -m py_compile src/dpf/validation/kr_targets.py src/dpf/validation/kr_corpus.py src/dpf/validation/__init__.py tests/test_kr_targets.py tests/test_kr_corpus.py` passed; `python3 -m pytest tests/test_kr_targets.py tests/test_kr_corpus.py -q` passed (`76 passed in 0.49s`); semantic/source audits passed; broad post-ratchet sweep passed with `127 passed in 0.62s` for quality/KR tests, `87 passed, 3 skipped in 1.65s` for MHD/physics/UQ tests, and `git diff --check` clean.
- Remaining scientific limit: this report models DPF-enabled NRTA, not the DPF plasma. It assumes a monoenergetic isotropic point source, postprocesses detector response, ignores room scatter/passive background, and explicitly calls for experiments, minimum-yield analysis, room geometry, and direct detector-response modeling.

Ratchet update 2026-05-05, DimLifePF96 empty extraction decision:

- Modules touched: `src/dpf/validation/kr_corpus.py`, `tests/test_kr_corpus.py`, `CortexFindings.md`, and `CodexFindings.md`.
- Reviewed `KnowledgeReference/dimensions-and-lifetime-of-the-plasma-focus-pinch-plasma-science-ieee-transactions-on-2.md`.
- Classified it as `insufficient_extractable_validation_data`: the local markdown contains only a title/source header and page stub.
- No target was added. Under the KR-only rule, pinch dimensions and lifetime cannot be inferred from the filename or missing PDF body.
- Current corpus closure after this ratchet: 42 coded KR target records from 38 unique local KR source files; 37 of 54 DPF-named markdown files are represented by coded targets; 15 additional DPF-named markdown files are review-closed by explicit decisions; 2 DPF-named markdown files remain unreviewed.
- Updated triage counts after removing this source from the review queue: all tracked scientific category counts are `0`.
- KnowledgeReference basis: lines `1-7` in the local DimLifePF96 extraction.
- Verification status: `python3 -m py_compile src/dpf/validation/kr_targets.py src/dpf/validation/kr_corpus.py src/dpf/validation/__init__.py tests/test_kr_targets.py tests/test_kr_corpus.py` passed; `python3 -m pytest tests/test_kr_targets.py tests/test_kr_corpus.py -q` passed (`76 passed in 0.59s`); broad post-ratchet sweep passed with `127 passed in 0.73s` for quality/KR tests, `87 passed, 3 skipped in 1.96s` for MHD/physics/UQ tests, and `git diff --check` clean.
- Remaining scientific limit: this source needs re-ingestion from the original PDF before any KR-only pinch dimension, lifetime, or diagnostic target can be extracted.

Ratchet update 2026-05-05, DPF-Bi-RRT acronym-collision decision:

- Modules touched: `src/dpf/validation/kr_corpus.py`, `tests/test_kr_corpus.py`, `CortexFindings.md`, and `CodexFindings.md`.
- Reviewed `KnowledgeReference/dpf-bi-rrt-an-improved-path-planning-algorithm-for-complex-3d-environments-with-adaptive-sampling.md`.
- Classified it as `non_dpf_acronym_collision`: in this IEEE Access path-planning paper, DPF means Dual Potential Field in `DPF-Bi-RRT*`, not Dense Plasma Focus.
- No target was added. The file concerns AAV path planning, RRT variants, biased random sampling, and obstacle avoidance; it provides no DPF physics or validation data.
- Current corpus closure after this ratchet: 42 coded KR target records from 38 unique local KR source files; 37 of 54 DPF-named markdown files are represented by coded targets; 16 additional DPF-named markdown files are review-closed by explicit decisions; 1 DPF-named markdown file remains unreviewed.
- Updated triage counts after removing this source from the review queue: all tracked scientific category counts are `0`.
- KnowledgeReference basis: lines `10-38`, `193-235`, `619-626`, `957-981`, and `1175-1193` show the path-planning meaning of DPF.
- Verification status: `python3 -m py_compile src/dpf/validation/kr_targets.py src/dpf/validation/kr_corpus.py src/dpf/validation/__init__.py tests/test_kr_targets.py tests/test_kr_corpus.py` passed; `python3 -m pytest tests/test_kr_targets.py tests/test_kr_corpus.py -q` passed (`76 passed in 0.54s`); broad post-ratchet sweep passed with `127 passed in 0.66s` for quality/KR tests, `87 passed, 3 skipped in 1.73s` for MHD/physics/UQ tests, and `git diff --check` clean.
- Remaining scientific limit: none for Dense Plasma Focus; this file is unrelated to the project’s scientific domain.

Ratchet update 2026-05-05, DPF simulator software-performance summary decision and final corpus-review closure:

- Modules touched: `src/dpf/validation/kr_corpus.py`, `tests/test_kr_corpus.py`, `CortexFindings.md`, and `CodexFindings.md`.
- Reviewed `KnowledgeReference/optimization-and-development-of-a-dense-plasma-focus-simulator.md`.
- Classified it as `non_scientific_software_performance_summary`: the local source is a two-page DPF simulator architecture/performance summary covering GUI, solvers, ML control, visualization, Metal GPU acceleration, CPU utilization, memory, and FPS.
- No target was added. The source provides no verified DPF physics equations, experimental diagnostics, calibration data, validation targets, or uncertainty data.
- Updated corpus tests for the completed review state: the DPF-named markdown queue is now fully reviewed, but `kr_corpus_review_status()["passed"]` remains false because validation coverage and same-scope predictive evidence are still incomplete.
- Current corpus closure after this ratchet: 42 coded KR target records from 38 unique local KR source files; 37 of 54 DPF-named markdown files are represented by coded targets; 17 additional DPF-named markdown files are review-closed by explicit decisions; 0 DPF-named markdown files remain unreviewed.
- Updated triage status: `kr_unreviewed_dpf_source_triage()` now passes with `0` unreviewed DPF-named markdown files and all tracked scientific category counts at `0`.
- Remaining KR target coverage blockers after full DPF-named corpus review: `circuit_waveform`, `phase_timing`, and `spatial_temperature` remain missing or partial; same-scope predictive readiness remains false.
- KnowledgeReference basis: lines `10-39` in the local software-performance source.
- Verification status: `python3 -m py_compile src/dpf/validation/kr_targets.py src/dpf/validation/kr_corpus.py src/dpf/validation/__init__.py tests/test_kr_targets.py tests/test_kr_corpus.py` passed; `python3 -m pytest tests/test_kr_targets.py tests/test_kr_corpus.py -q` passed (`76 passed in 0.54s`); semantic/source audits passed; broad post-ratchet sweep passed with `127 passed in 0.65s` for quality/KR tests, `87 passed, 3 skipped in 1.80s` for MHD/physics/UQ tests, and `git diff --check` clean.
- Remaining scientific limit: all DPF-named local markdown files have now been reviewed or target-extracted, but the repository is still not a validated end-to-end predictive DPF simulation tool. The remaining blockers are validation depth and implementation fidelity, not unreviewed DPF-named KR files.

Ratchet update 2026-05-05, corpus-review completion plan update:

- Modules touched: `src/dpf/validation/kr_corpus.py`, `tests/test_kr_corpus.py`, `CortexFindings.md`, and `CodexFindings.md`.
- Reviewed the post-corpus status after all DPF-named markdown files reached closure.
- Updated `kr_corpus_review_status()["next_ratcheting_steps"]` so it no longer tells us to review unreviewed DPF-named files after the queue is empty.
- New local plan reported by the code:
  - DPF-named KnowledgeReference markdown review is complete.
  - Close remaining target coverage blockers: `circuit_waveform`, `phase_timing`, and `spatial_temperature`.
  - Promote one same-scope validation packet by adding KR-backed circuit, phase, spatial, neutron, and uncertainty evidence for a single device/shot/scope, or keep readiness blocked when KR lacks those observables.
- Verification status: `python3 -m py_compile src/dpf/validation/kr_corpus.py tests/test_kr_corpus.py` passed; `python3 -m pytest tests/test_kr_targets.py tests/test_kr_corpus.py -q` passed (`76 passed in 0.50s`); `python3 -m pytest tests/test_quality_assessment.py tests/test_kr_targets.py tests/test_kr_corpus.py -q` passed (`127 passed in 0.61s`); `git diff --check` clean.
- Remaining scientific limit: source review is no longer the ratchet. The next ratchet must improve validation evidence or explicitly preserve readiness blockers where KR data is absent.

Ratchet update 2026-05-05, same-scope closure-path report:

- Modules touched: `src/dpf/validation/kr_targets.py`, `tests/test_kr_targets.py`, `CortexFindings.md`, and `CodexFindings.md`.
- Reviewed same-scope target status after full DPF-named corpus closure.
- Added `widest_available_scope` and `next_same_scope_steps` to `kr_validation_same_scope_target_report()`.
- The report now distinguishes two useful views:
  - `best_available_scope`: currently MJOLNIR has fewer total blockers but is missing several required groups.
  - `widest_available_scope`: PF-1000 full-energy `pf1000_full_energy_2007_gribkov_scholz` has all required groups present but remains incomplete because most groups are partial.
- The PF-1000 full-energy partial blockers are `circuit_waveform`, `neutron_anisotropy`, `neutron_detector_response`, `neutron_spectrum`, `neutron_timing`, `phase_timing`, `spatial_magnetic_or_em`, `spatial_temperature`, and `uncertainty`.
- The code-level next step now says to use the widest same-scope packet as the closure path and keep predictive readiness blocked until those partial groups have digitized traces, uncertainty, and same-shot diagnostic support.
- Verification status: `python3 -m py_compile src/dpf/validation/kr_targets.py tests/test_kr_targets.py` passed; `python3 -m pytest tests/test_kr_targets.py tests/test_kr_corpus.py -q` passed (`76 passed in 0.62s`); `python3 -m pytest tests/test_quality_assessment.py tests/test_kr_targets.py tests/test_kr_corpus.py -q` passed (`127 passed in 0.61s`); `git diff --check` clean.
- Remaining scientific limit: the KR corpus, as currently extracted, gives a broad PF-1000 packet but not a complete predictive validation packet. Closing it requires digitized waveform, phase, spatial, neutron, and uncertainty evidence for the same PF-1000 scope, or explicit permanent blockers where KR lacks those observables.

Ratchet update 2026-05-05, PF-1000 closure-blocker checklist:

- Modules touched: `src/dpf/validation/kr_targets.py`, `tests/test_kr_targets.py`, `CortexFindings.md`, and `CodexFindings.md`.
- `kr_validation_same_scope_target_report()` now emits `closure_blockers` and `closure_blocker_groups` for each same-scope packet.
- For the PF-1000 full-energy closure path, the report now records target ids, KR source files, and missing items for each partial group.
- The encoded PF-1000 blockers include `digitized_current_trace_points`, `radial_transit_start_and_end_times`, `direct_experimental_temperature_diagnostic`, `neutron_field_transport_or_room_scatter_response_model`, and `fast_ion_distribution_uncertainty`.
- Verification status: `python3 -m py_compile src/dpf/validation/kr_targets.py tests/test_kr_targets.py` passed; `python3 -m pytest tests/test_kr_targets.py::test_kr_validation_same_scope_target_report_requires_one_scope -q` passed (`1 passed in 0.53s`); `python3 -m pytest tests/test_kr_targets.py tests/test_kr_corpus.py -q` passed (`76 passed in 0.49s`); `python3 -m pytest tests/test_quality_assessment.py tests/test_kr_targets.py tests/test_kr_corpus.py -q` passed (`127 passed in 0.64s`); `python3 -m pytest tests/test_mhd_physics_integration.py tests/test_mhd_numerical_fidelity.py tests/test_circuit_field_coupling.py tests/test_physics_fidelity.py tests/test_uncertainty_budget.py -q` passed (`87 passed, 3 skipped in 1.64s`); `git diff --check` clean.
- Remaining scientific limit: this ratchet does not create the missing data. It prevents the closure plan from being hand-wavy by making the exact KR-backed blockers inspectable by code and tests.

Ratchet update 2026-05-05, broad DPF-content corpus queue:

- Modules touched: `src/dpf/validation/kr_corpus.py`, `tests/test_kr_corpus.py`, `CortexFindings.md`, and `CodexFindings.md`.
- Replaced filename-only source-closure accounting with filename-or-strong-content DPF relevance. Strong content markers include `dense plasma focus`, `plasma focus`, `PF-1000`, `PF1000`, `PF 1000`, `MJOLNIR`, `Mather-type`, and `Filippov`.
- Current corpus status now reports 827 total files, 398 markdown files, 396 JSON files, 54 DPF-named markdown files, 94 DPF-content markdown files, and 96 DPF-relevant markdown files.
- Review status is deliberately open again: 55 of 96 DPF-relevant markdown files are review-closed by coded target or explicit decision, leaving 41 DPF-relevant markdown files to review.
- Verification status: `python3 -m py_compile src/dpf/validation/kr_corpus.py tests/test_kr_corpus.py` passed; `python3 -m pytest tests/test_kr_corpus.py -q` passed (`4 passed in 1.37s`); `python3 -m pytest tests/test_kr_targets.py tests/test_kr_corpus.py -q` passed (`76 passed in 1.24s`).
- Remaining scientific limit: the previous DPF-named review was real but not sufficient for the user's broader "all source-of-truth documents" question. The next ratchet is to review the 41 DPF-content files and either extract targets or record explicit non-target decisions.

Ratchet update 2026-05-05, broad DPF-content review wave 1:

- Module touched: `src/dpf/validation/kr_corpus.py`.
- Reviewed and closed 20 of the 41 newly exposed broad DPF-content markdown files by explicit decision.
- Decisions added for duplicate FAETON-I and hybrid X-pinch extractions; general/non-DPF Z-pinch model papers; reference-only DPF hits; educational/software/image-index sources; and application/materials papers that do not provide DPF machine validation observables.
- Current broad status: 75 of 96 DPF-relevant markdown files are review-closed, leaving 21 DPF-relevant candidates open.
- Verification status: `python3 -m py_compile src/dpf/validation/kr_corpus.py` passed; `python3 -m pytest tests/test_kr_corpus.py -q` passed (`4 passed in 1.09s`).
- Remaining scientific limit: several remaining files look more scientifically relevant, including MJOLNIR first-experiment/diagnostic sources, PF-1000 pinch-column/optical/plasma-emission sources, DPF diagnostic papers, Lee-code studies, and Auluck theory/survey papers. These still require target extraction or explicit closure decisions.

Ratchet update 2026-05-05, broad DPF-content review closure:

- Modules touched: `src/dpf/validation/kr_targets.py`, `src/dpf/validation/kr_corpus.py`, `src/dpf/validation/__init__.py`, `tests/test_kr_targets.py`, `tests/test_kr_corpus.py`, `CortexFindings.md`, and `CodexFindings.md`.
- Added `mjolnir_first_experiments_2021_offermann` as a coded partial MJOLNIR 1 MJ campaign target from the IEEE first-experiments/radiographs paper. It captures device geometry, 204 uF/67.4 nH/12.5 mohm circuit constants, Rogowski/light-gate/nToF/activation diagnostics, 3.8e11 max yield, and detector-response limits.
- Added `uofsi_argon_temperature_thesis_2020` as a coded partial UofS-I 1 kJ argon temperature target. It captures the 5 uF/20 kV/100-200 mTorr device, Lee current-fit factors, 1.15 us axial timing, 1.3 us pinch timing, and 5.7 +/- 0.7 keV soft-x-ray-filter electron-temperature result.
- Added explicit decisions for the remaining broad DPF-content files, including Auluck filamentation/poloidal-flux context, Orellana UHF diagnostics, Lee/Saw scaling reviews, PF-1000 late-pinch qualitative context, detector-calibration papers, DPF1000U stream spectroscopy, legacy thesis OCR, p-B11 alpha application context, Lee-code model-only studies, and MJOLNIR presentation context.
- Current source-review status: 96 of 96 DPF-relevant markdown files are now review-closed by coded target or explicit decision; the unreviewed DPF-relevant queue is empty.
- Remaining validation blockers now reported by code: `circuit_waveform`, `phase_timing`, `spatial_temperature`, and `uncertainty` remain missing or partial; no same-scope validation packet passes.
- Verification status: `python3 -m py_compile src/dpf/validation/kr_corpus.py src/dpf/validation/kr_targets.py src/dpf/validation/__init__.py tests/test_kr_corpus.py tests/test_kr_targets.py` passed; `python3 -m pytest tests/test_kr_corpus.py -q` passed (`4 passed in 1.13s`); `python3 -m pytest tests/test_kr_targets.py -q` passed (`74 passed in 0.46s`); `python3 -m pytest tests/test_quality_assessment.py tests/test_kr_targets.py tests/test_kr_corpus.py -q` passed (`129 passed in 1.29s`); `python3 -m pytest tests/test_mhd_physics_integration.py tests/test_mhd_numerical_fidelity.py tests/test_circuit_field_coupling.py tests/test_physics_fidelity.py tests/test_uncertainty_budget.py -q` passed (`87 passed, 3 skipped in 5.88s`); `git diff --check` clean.
- Remaining scientific limit: source review is now closed for DPF-relevant markdown, but the product is still not a validated end-to-end simulator. The blocker has shifted fully to evidence quality: digitized same-scope traces, complete phase endpoints, direct same-scope spatial temperature/density/B validation, detector response, and propagated uncertainty.

Ratchet update 2026-05-05, source-review gap closure in readiness reports:

- Modules touched: `src/dpf/validation/quality_assessment.py`, `tests/test_quality_assessment.py`, `tests/test_mhd_physics_integration.py`, `CortexFindings.md`, and `CodexFindings.md`.
- `scientific_accuracy_gap_report()` now includes a separate `kr_source_review` gap.
- That gap is `supported` when the DPF-relevant KnowledgeReference markdown queue is empty; current runs report the source review as closed rather than mixing it with validation-evidence blockers.
- The `kr_target_coverage` blocker now names the widest same-scope closure path when target coverage is partial. Current widest path remains PF-1000 full-energy, with closure blockers carried by the same-scope target report.
- Verification status: `python3 -m py_compile src/dpf/validation/quality_assessment.py tests/test_quality_assessment.py tests/test_mhd_physics_integration.py` passed; `python3 -m pytest tests/test_quality_assessment.py::TestQualityAssessment::test_scientific_accuracy_gap_report_lists_remaining_work tests/test_mhd_physics_integration.py::test_predictive_readiness_exported_and_blocks_unvalidated_claims -q` passed (`2 passed in 1.16s`); `python3 -m pytest tests/test_quality_assessment.py -q` passed (`51 passed in 2.38s`); `python3 -m pytest tests/test_quality_assessment.py tests/test_kr_targets.py tests/test_kr_corpus.py -q` passed (`129 passed in 3.13s`); `python3 -m pytest tests/test_mhd_physics_integration.py tests/test_mhd_numerical_fidelity.py tests/test_circuit_field_coupling.py tests/test_physics_fidelity.py tests/test_uncertainty_budget.py -q` passed (`87 passed, 3 skipped in 7.50s`); `git diff --check` clean.
- Remaining scientific limit: this improves product honesty. It does not close the experimental evidence blockers that keep high-fidelity readiness false.

Ratchet update 2026-05-05, same-scope uncertainty packet gate:

- Modules touched: `src/dpf/validation/uncertainty_budget.py`, `tests/test_uncertainty_budget.py`, `CortexFindings.md`, and `CodexFindings.md`.
- Tightened `uncertainty_evidence_from_result()` so a complete uncertainty component set must share one `validation_scope`.
- Cross-scope UQ components now fail with `same_scope_uncertainty_packet` in `missing_or_unvalidated_components`.
- The complete synthetic high-fidelity test still passes because its UQ components use one scope.
- Verification status: `python3 -m py_compile src/dpf/validation/uncertainty_budget.py tests/test_uncertainty_budget.py` passed; `python3 -m pytest tests/test_uncertainty_budget.py -q` passed (`10 passed in 0.91s`); `python3 -m pytest tests/test_quality_assessment.py::TestQualityAssessment::test_high_fidelity_readiness_can_pass_with_complete_evidence_packet tests/test_uncertainty_budget.py::test_complete_uncertainty_components_must_share_validation_scope -q` passed (`2 passed in 0.65s`).
- Remaining scientific limit: this closes a validation-gate loophole, not the underlying UQ dataset. Real DPF runs still need same-scope experimental, input, numerical, model-form, shot-to-shot, propagated, and KR-target uncertainty components.

Ratchet update 2026-05-05, same-scope physics-fidelity packet gate:

- Modules touched: `src/dpf/validation/physics_fidelity.py`, `tests/test_physics_fidelity.py`, `CortexFindings.md`, and `CodexFindings.md`.
- Tightened `physics_fidelity_evidence_from_result()` so a complete high-fidelity physics-effect packet must share one `validation_scope`.
- Cross-scope physics-effect components now fail with `same_scope_physics_packet` in `missing_or_unvalidated_effects`.
- The complete synthetic high-fidelity test still passes because its physics-effect evidence uses one scope.
- Verification status: `python3 -m py_compile src/dpf/validation/physics_fidelity.py tests/test_physics_fidelity.py` passed; `python3 -m pytest tests/test_physics_fidelity.py -q` passed (`7 passed in 1.32s`); `python3 -m pytest tests/test_quality_assessment.py::TestQualityAssessment::test_high_fidelity_readiness_can_pass_with_complete_evidence_packet tests/test_physics_fidelity.py::test_complete_physics_effects_must_share_validation_scope tests/test_uncertainty_budget.py::test_complete_uncertainty_components_must_share_validation_scope -q` passed (`3 passed in 0.85s`); `python3 -m pytest tests/test_mhd_physics_integration.py tests/test_mhd_numerical_fidelity.py tests/test_circuit_field_coupling.py tests/test_physics_fidelity.py tests/test_uncertainty_budget.py -q` passed (`89 passed, 3 skipped in 7.71s`).
- Remaining scientific limit: this closes another validation-gate loophole, not the underlying physics dataset. Real runs still need one KR-backed validation scope whose required EOS/conductivity, ionization, two-temperature, radiation, impurity/ablation, kinetic/Hall/FLR, 3D instability, flashover, restrike, and beam-target effects are implemented and validated or explicitly bounded out.

Ratchet update 2026-05-05, same-scope circuit/field-coupling packet gate:

- Modules touched: `src/dpf/validation/circuit_field_coupling.py`, `src/dpf/validation/quality_assessment.py`, `tests/test_circuit_field_coupling.py`, `CortexFindings.md`, and `CodexFindings.md`.
- Tightened `field_coupling_evidence_from_result()` so a complete field-coupling component packet must share one `validation_scope`.
- Cross-scope field-coupling components now fail with `same_scope_field_coupling_packet` in `missing_or_unvalidated_evidence`.
- `scientific_accuracy_gap_report()` now treats a complete-but-cross-scope field-coupling packet as `blocked`, not merely `partial`.
- The complete synthetic high-fidelity test still passes because its field-coupling evidence uses one scope.
- Verification status: `python3 -m py_compile src/dpf/validation/circuit_field_coupling.py src/dpf/validation/quality_assessment.py tests/test_circuit_field_coupling.py` passed; `python3 -m pytest tests/test_circuit_field_coupling.py -q` passed (`12 passed in 0.85s`); `python3 -m pytest tests/test_quality_assessment.py::TestQualityAssessment::test_high_fidelity_readiness_can_pass_with_complete_evidence_packet tests/test_circuit_field_coupling.py::test_complete_field_coupling_components_must_share_validation_scope tests/test_physics_fidelity.py::test_complete_physics_effects_must_share_validation_scope tests/test_uncertainty_budget.py::test_complete_uncertainty_components_must_share_validation_scope -q` passed (`4 passed in 1.05s`); `python3 -m pytest tests/test_quality_assessment.py tests/test_mhd_physics_integration.py tests/test_mhd_numerical_fidelity.py tests/test_circuit_field_coupling.py tests/test_physics_fidelity.py tests/test_uncertainty_budget.py -q` passed (`141 passed, 3 skipped in 9.54s`).
- Remaining scientific limit: this closes a field-coupling validation-gate loophole, not the underlying current-coupling dataset. Real MHD-mode current prediction still needs same-scope validated inductance, dL/dt/back-EMF, Poynting power, circuit energy, transition timing, and KR experimental comparison evidence.

Ratchet update 2026-05-05, global high-fidelity scope-alignment gate:

- Modules touched: `src/dpf/validation/quality_assessment.py`, `tests/test_quality_assessment.py`, `CortexFindings.md`, and `CodexFindings.md`.
- Added `same_scope_high_fidelity_claim` to `scientific_accuracy_gap_report()`.
- The new gap requires KR target coverage, field-coupling, physics-fidelity, and uncertainty packets to share at least one `validation_scope`.
- Complete-but-cross-scope support packets are now blocked at the high-fidelity claim level even when each packet passes internally.
- The complete synthetic high-fidelity test still passes because target coverage, field coupling, physics fidelity, and uncertainty all share `synthetic_complete_high_fidelity_scope`.
- Verification status: `python3 -m py_compile src/dpf/validation/quality_assessment.py tests/test_quality_assessment.py` passed; `python3 -m pytest tests/test_quality_assessment.py::TestQualityAssessment::test_scientific_accuracy_gap_report_lists_remaining_work tests/test_quality_assessment.py::TestQualityAssessment::test_high_fidelity_readiness_can_pass_with_complete_evidence_packet tests/test_quality_assessment.py::TestQualityAssessment::test_high_fidelity_scope_alignment_blocks_cross_scope_packets -q` passed (`3 passed in 1.30s`); `python3 -m pytest tests/test_quality_assessment.py tests/test_mhd_physics_integration.py tests/test_mhd_numerical_fidelity.py tests/test_circuit_field_coupling.py tests/test_physics_fidelity.py tests/test_uncertainty_budget.py -q` passed (`142 passed, 3 skipped in 9.86s`).
- Remaining scientific limit: this enforces claim consistency across validation packets. It does not create the missing same-scope experimental waveform, phase, spatial, neutron, detector-response, physics-closure, coupling, or uncertainty data needed for a real predictive DPF validation packet.

Ratchet update 2026-05-05, global scope gate extended to tier evidence:

- Modules touched: `src/dpf/validation/quality_assessment.py`, `tests/test_quality_assessment.py`, `CortexFindings.md`, and `CodexFindings.md`.
- Extended `same_scope_high_fidelity_claim` so source authority, circuit validation, snowplow validation, spatial validation, neutron validation, and neutron detector response must also share the high-fidelity `validation_scope`.
- The global gate now aligns the actual tier evidence with the KR target packet, field-coupling packet, physics-fidelity packet, and uncertainty packet.
- Verification status: `python3 -m py_compile src/dpf/validation/quality_assessment.py tests/test_quality_assessment.py` passed; `python3 -m pytest tests/test_quality_assessment.py::TestQualityAssessment::test_scientific_accuracy_gap_report_lists_remaining_work tests/test_quality_assessment.py::TestQualityAssessment::test_high_fidelity_readiness_can_pass_with_complete_evidence_packet tests/test_quality_assessment.py::TestQualityAssessment::test_high_fidelity_scope_alignment_blocks_cross_scope_packets -q` passed (`3 passed in 1.31s`); `python3 -m pytest tests/test_quality_assessment.py tests/test_mhd_physics_integration.py tests/test_mhd_numerical_fidelity.py tests/test_circuit_field_coupling.py tests/test_physics_fidelity.py tests/test_uncertainty_budget.py -q` passed (`142 passed, 3 skipped in 9.78s`).
- Remaining scientific limit: no real KR-backed DPF run currently supplies a complete same-scope packet across source authority, circuit, phase, spatial, neutron, detector response, coupling, physics closures, and uncertainty.

Ratchet update 2026-05-05, same-scope MHD numerical-fidelity packet gate:

- Modules touched: `src/dpf/validation/mhd_numerical_fidelity.py`, `src/dpf/validation/circuit_field_coupling.py`, `src/dpf/validation/quality_assessment.py`, `tests/test_mhd_numerical_fidelity.py`, `CortexFindings.md`, and `CodexFindings.md`.
- Added `verification_scope` metadata to cylindrical convergence, resistive diffusion, backend parity, MHD phase scope-limit, and circuit-coupled energy verification evidence.
- Tightened `mhd_numerical_fidelity_evidence_from_result()` so a complete Tier-3 numerical-fidelity packet must share one verification scope.
- Cross-scope numerical verification bundles now fail with `same_scope_mhd_numerical_packet` in `missing_or_unvalidated_evidence`.
- `scientific_accuracy_gap_report()` now treats a complete-but-cross-scope MHD numerical packet as `blocked`, not merely `partial`.
- Verification status: `python3 -m py_compile src/dpf/validation/mhd_numerical_fidelity.py src/dpf/validation/circuit_field_coupling.py src/dpf/validation/quality_assessment.py tests/test_mhd_numerical_fidelity.py` passed; `python3 -m pytest tests/test_mhd_numerical_fidelity.py -q` passed (`21 passed in 1.09s`); `python3 -m pytest tests/test_quality_assessment.py tests/test_mhd_physics_integration.py tests/test_mhd_numerical_fidelity.py tests/test_circuit_field_coupling.py tests/test_physics_fidelity.py tests/test_uncertainty_budget.py -q` passed (`143 passed, 3 skipped in 10.25s`).
- Remaining scientific limit: this closes a Tier-3 packet-consistency loophole, not the DPF validation problem. Real high-fidelity readiness still needs DPF-specific numerical verification tied to the same claim scope as the experimental target packet.

Ratchet update 2026-05-05, same-scope predictive-readiness tier gate:

- Modules touched: `src/dpf/validation/quality_assessment.py`, `tests/test_quality_assessment.py`, `CortexFindings.md`, and `CodexFindings.md`.
- Tightened `predictive_readiness_report()` so circuit waveform validation, snowplow phase/timing validation, spatial DPF validation, and neutron timing/spectrum/anisotropy validation must share one `validation_scope`.
- Cross-scope tier evidence now fails the lower predictive-readiness gate with `Predictive validation scope alignment` in `missing_evidence`.
- Verification status: `python3 -m py_compile src/dpf/validation/quality_assessment.py tests/test_quality_assessment.py` passed; `python3 -m pytest tests/test_quality_assessment.py::TestQualityAssessment::test_predictive_readiness_passes_only_with_all_required_tiers tests/test_quality_assessment.py::TestQualityAssessment::test_predictive_readiness_requires_one_validation_scope tests/test_quality_assessment.py::TestQualityAssessment::test_high_fidelity_readiness_requires_gap_closure tests/test_quality_assessment.py::TestQualityAssessment::test_high_fidelity_readiness_can_pass_with_complete_evidence_packet -q` passed (`4 passed in 1.11s`); `python3 -m pytest tests/test_quality_assessment.py tests/test_mhd_physics_integration.py tests/test_mhd_numerical_fidelity.py tests/test_circuit_field_coupling.py tests/test_physics_fidelity.py tests/test_uncertainty_budget.py -q` passed (`144 passed, 3 skipped in 10.45s`).
- Remaining scientific limit: this prevents a misleading lower `predictive_ready` label from independently sourced tier evidence. It does not create the same-scope circuit, snowplow, spatial, and neutron data needed for a real predictive run.

Ratchet update 2026-05-05, machine-readable KR data-availability blockers:

- Modules touched: `src/dpf/validation/kr_targets.py`, `tests/test_kr_targets.py`, `CortexFindings.md`, and `CodexFindings.md`.
- Added `data_availability` and `required_data_to_complete` to each `closure_blockers` record from `kr_validation_same_scope_target_report()`.
- Missing same-scope groups are now labeled `absent_from_same_scope_targets`; partial groups are labeled `partial_only_in_same_scope_targets`.
- For the PF-1000 closure path, the exact missing data list is now duplicated into `required_data_to_complete`, so downstream product planning can read the blocker list without parsing prose.
- Verification status: `python3 -m py_compile src/dpf/validation/kr_targets.py tests/test_kr_targets.py` passed; `python3 -m pytest tests/test_kr_targets.py::test_kr_validation_same_scope_target_report_requires_one_scope -q` passed (`1 passed in 0.55s`); `python3 -m pytest tests/test_kr_targets.py tests/test_kr_corpus.py -q` passed (`78 passed in 1.13s`).
- Remaining scientific limit: the report now states the data availability status more clearly, but the current KR corpus still lacks a complete same-scope PF-1000 packet with digitized waveforms, complete phase timing, direct temperature diagnostics, detector response, and propagated uncertainty.

Ratchet update 2026-05-05, verification sweep checkpoint:

- Validation/KR/readiness regression sweep passed: `python3 -m pytest tests/test_quality_assessment.py tests/test_mhd_physics_integration.py tests/test_mhd_numerical_fidelity.py tests/test_circuit_field_coupling.py tests/test_physics_fidelity.py tests/test_uncertainty_budget.py tests/test_kr_targets.py tests/test_kr_corpus.py -q` returned `222 passed, 3 skipped in 11.85s`.
- `git diff --check` is clean.
- Full `python3 -m pytest -q` was attempted and aborted during collection while importing `dpf.metal.mlx_device` from `tests/test_amr_mlx.py`; this failed before assertions ran in the MLX import path.
- Current live gap report: KR source review is supported at 96/96 DPF-relevant markdown files reviewed, PF-1000 full-energy remains the widest but incomplete same-scope closure path, and predictive/high-fidelity readiness remains blocked by missing same-scope validation evidence, physics fidelity, field coupling, UQ, and Tier-3/Tier-4/Tier-5 evidence.

Ratchet update 2026-05-06, user decisions and next scientific-closure plan:

- User decisions captured:
  - New source-of-truth material is allowed only after an AI researches and provides a link/source document and the user acquires the correct document.
  - Manual digitization is allowed only with a reproducible one-for-one verification method.
  - Device choice is secondary; physics closure is the objective.
  - Product target is a full high-fidelity neutron-predictive DPF simulator.
  - Scientific closure is priority 1; hardening is priority 2.
- Next plan:
  1. Build a digitization provenance and verification workflow: source file hash, figure/page/axis metadata, calibration points, extracted arrays, reviewer check, and residual/error report against the source image or table.
  2. Convert current closure blockers into a source-acquisition queue by physics need: circuit waveform, phase timing, spatial density/B/T, neutron timing/spectrum/anisotropy, detector response, and uncertainty.
  3. Research candidate source documents and provide links for user acquisition before anything is added to `KnowledgeReference`.
  4. After user acquisition, ingest locally, review under the KR-only rule, extract typed targets, and rerun same-scope closure reports.
  5. Once evidence exists, implement or validate the required physics closures: EOS/conductivity, ionization, two-temperature energy partition, radiation transport/opacities, impurity/ablation, Hall/FLR/PIC or bounded kinetic treatment, 3D instability scope, flashover/startup, restrike/anomalous resistance, and beam-target neutron coupling.
- Working assumption: until acquired documents or verified digitized data close these gaps, the repository must keep predictive and high-fidelity readiness blocked.

Ratchet update 2026-05-06, digitization gate and source-acquisition queue:

- Modules touched: `src/dpf/validation/digitization.py`, `src/dpf/validation/source_acquisition.py`, `src/dpf/validation/__init__.py`, `tests/test_digitization.py`, `docs/SCIENTIFIC_CLOSURE_SOURCE_QUEUE.md`, `CortexFindings.md`, and `CodexFindings.md`.
- Added `digitization_verification_evidence()` and `sha256_file()` as the one-for-one verification method for manually digitized KR figures/tables. The audit requires a local `KnowledgeReference/` source hash, figure image hash for figure data, source item/page metadata, axis calibration with residual limits, extracted arrays with units, overlay residual evidence, and at least one accepted independent review. It fails closed on `KnowledgeReference` path traversal and malformed review-count metadata.
- Added `scientific_closure_source_acquisition_queue()` so the current same-scope KR blockers are exposed as machine-readable acquisition items by physics need, required data, current KR sources, candidate DOI links, status, and done condition.
- Added `docs/SCIENTIFIC_CLOSURE_SOURCE_QUEUE.md` as the human-facing queue for user acquisition. It records the current widest scope (`pf1000_full_energy_2007_gribkov_scholz`), the strict KR-only rule, the acquisition process, the digitization gate, and candidate sources for circuit waveform, phase timing, spatial temperature, uncertainty, neutron anisotropy, neutron detector response, neutron spectrum, neutron timing, and spatial magnetic/EM closure.
- Corrected the Zr/Be activation detector candidate DOI to `10.1016/j.nima.2020.164830` and added tests requiring every live blocker to carry at least one candidate DOI/URL.
- Candidate links are explicitly not treated as validation evidence. They are only acquisition leads until the user obtains the paper, it is placed under `KnowledgeReference/`, Codex reviews it locally, and any digitized data passes the provenance audit.
- Verification status: `python3 -m py_compile src/dpf/validation/digitization.py src/dpf/validation/source_acquisition.py src/dpf/validation/__init__.py tests/test_digitization.py` passed; `python3 -m pytest tests/test_digitization.py -q` passed (`5 passed in 0.75s`); `python3 -m pytest tests/test_digitization.py tests/test_kr_targets.py tests/test_kr_corpus.py tests/test_quality_assessment.py::TestQualityAssessment::test_scientific_accuracy_gap_report_lists_remaining_work -q` passed (`84 passed in 1.62s`); `python3 -m pytest tests/test_quality_assessment.py tests/test_mhd_physics_integration.py tests/test_mhd_numerical_fidelity.py tests/test_circuit_field_coupling.py tests/test_physics_fidelity.py tests/test_uncertainty_budget.py tests/test_kr_targets.py tests/test_kr_corpus.py tests/test_digitization.py -q` passed (`227 passed, 3 skipped in 12.22s`); `git diff --check` is clean.
- Remaining scientific limit: this ratchet creates the acquisition and verification machinery needed for scientific closure, but does not close the simulator. The live blockers are still missing verified local data for current traces with uncertainties, absolute phase timing, direct temperature diagnostics, same-shot magnetic/EM validation, neutron pulse/spectrum digitization, detector/room-scatter response, fast-ion distribution uncertainty, and propagated same-scope UQ.

Ratchet update 2026-05-06, local PDF source audit:

- Modules/docs touched: `docs/SCIENTIFIC_CLOSURE_SOURCE_QUEUE.md`, `docs/LOCAL_PDF_SOURCE_AUDIT_2026_05_06.md`, `CortexFindings.md`, and `CodexFindings.md`.
- Checked local PDFs under the DPF-Unified tree by filename search, PDF metadata, DOI/title text extraction, and SHA-256 duplicate checks.
- Found exact local PDF matches for Akel 2021 PF-1000 neutron-yield/current paper, Gribkov Part I, Gribkov Part II, Schmidt/Goyon MJOLNIR high-low paper, Malir 2024 interferometry paper, and Goyon 2025 MA-class neutron-generation paper.
- Identified filename problems:
  - `archive_reference_OLD/references/papers/core-dpf/gribkov-2007-pf1000-jphysd-part2.pdf` and `KnowledgeReference/gribkov-2007-pf1000-jphysd-part2.md` are actually Gribkov et al. 2007 Part I, DOI `10.1088/0022-3727/40/7/021`.
  - `archive_reference_OLD/references/papers/core-dpf/scholz-2007-pf1000-part2-jphysd.pdf` and `KnowledgeReference/scholz-2007-pf1000-part2-jphysd.md` are actually Gribkov et al. 2007 Part II, DOI `10.1088/0022-3727/40/12/008`.
  - `goyon-2022-mjolnir-high-low` names the Schmidt et al. 2022 MJOLNIR article by a non-first author.
  - `petrov-2022-mjolnir-high-low-discharges` appears to be an LLNL accepted-manuscript/preprint copy of the Schmidt/Goyon MJOLNIR article, not a separate Petrov-authored target paper.
- Found exact duplicate PDF hashes for Akel 2021 between `core-dpf/` and `archive/`; found exact duplicate PDF hashes for Goyon 2025 across the short and long filename copies.
- Did not find exact local PDFs for Cikhardtova 2015 linear-density timing, Sadowski/Scholz 2004 PF-1000 fast ions/neutrons, Catenacci 2020 neutron time-energy tomography, Springham 2021 Zr/Be activation detectors, Klir 2011 TOF detector calibration, or Jednorog 2017 PF-1000 activation monitor.
- Correction from the subsequent parity pass: Akel 2021 was already represented in `KnowledgeReference/radiation-physics-and-chemistry-188-2021-109633.md`. The next action is verified digitization of its waveform/yield figures and tables, not paper ingestion.
- Verification status: `git diff --check` is clean after the audit documentation update.

Ratchet update 2026-05-06, KR PDF parity verification:

- Modules/docs touched: `scripts/verify_kr_pdf_parity.py`, `docs/LOCAL_PDF_SOURCE_AUDIT_2026_05_06.md`, `docs/SCIENTIFIC_CLOSURE_SOURCE_QUEUE.md`, `CortexFindings.md`, and `CodexFindings.md`.
- Added a reusable parity verifier. It requires PDF page count to match KR JSON page count, every PDF page's extracted text to match KR JSON `pages[].text`, and every PDF page's extracted text to be present in the KR markdown after normalization.
- Correction to prior audit: Akel et al. 2021 was already in `KnowledgeReference` as `KnowledgeReference/radiation-physics-and-chemistry-188-2021-109633.md`; my earlier filename-only check missed it because the filename is generic.
- No new `KnowledgeReference` markdown file was created. All exact local PDF matches already had KR markdown/JSON pairs and passed text parity:
  - Akel et al. 2021: 6/6 pages, SHA-256 `9a762bc36bc1f5c175a0ec8dc07b69c48ad956d0c6a382882daf4e24677dcb3b`.
  - Gribkov et al. 2007 Part I: 13/13 pages, SHA-256 `7acfb46d1db6ee5894978f70e1372edda7efaa5171d8e7c3bdf0baf7025eff43`.
  - Gribkov et al. 2007 Part II: 16/16 pages, SHA-256 `c4d62f5015bc6040aa85070e43f3cb6e7e4a8329e5d2baf33fa4d38f828caa4f`.
  - Schmidt et al. 2022 MJOLNIR article: 29/29 pages, SHA-256 `89877f5c880dcd9c4454925984398cf51984f95d2ff78ac4437f5f755e98fe6a`.
  - Schmidt/Goyon accepted-manuscript copy: 16/16 pages, SHA-256 `d9674bd39b12c3a87e7549c540384f56722d739f5b85a693fab73c24b2d32623`.
  - Malir et al. 2024: 14/14 pages, SHA-256 `fafc32261c9172702b1c8dfdc92bcc33b1a32aeeb4cb9680d535478191db46c9`.
  - Goyon et al. 2025 canonical KR record and short-name KR duplicate: 10/10 pages, SHA-256 `9c0bc58d72ced9c914914aabdab63937a2b9c7820950eb0fa2412be9fd9d0f8c`.
- Important boundary: this is text parity only. Figure pixels and plotted curves have not been converted to numeric evidence. They still require `digitization_verification_evidence()` before use in validation.
- Immediate next action: verified digitization of Akel 2021 waveform/yield figures and tables, not paper ingestion.
- Verification status: `python3 -m py_compile scripts/verify_kr_pdf_parity.py` passed; `python3 -m pytest tests/test_digitization.py -q` passed (`5 passed in 0.85s`); `git diff --check` is clean.

Ratchet update 2026-05-06, Akel 2021 typed table target:

- Modules/docs touched: `src/dpf/validation/kr_targets.py`, `src/dpf/validation/__init__.py`, `tests/test_kr_targets.py`, `docs/SCIENTIFIC_CLOSURE_SOURCE_QUEUE.md`, `docs/LOCAL_PDF_SOURCE_AUDIT_2026_05_06.md`, `CodexFindings.md`, and `CortexFindings.md`.
- Promoted Akel et al. 2021 Tables 1 and 2 from KR text into `pf1000_16kv_shot_table_2021_akel`, a typed PF-1000 16 kV validation target containing 24 merged shot rows.
- Each row now carries pressure, L0, r0, peak current, pinch current, Lee fitted factors, axial/shock/piston speeds, pinch density, pinch radius/length, computed neutron yield, measured neutron yield, and the row's printed measured-yield uncertainty.
- Added table extraction provenance: Table 1 row count, Table 2 row count, merged row count, shot-ID match, source line windows, KR markdown/PDF parity flag, parity verifier name, and the Akel PDF SHA-256.
- Corrected the existing shot-12581 phase target `fmr` from `0.25` to `0.26` for the table-backed scalar row. The prose on lines 270-273 says `0.25`; Table 1 lines 344-353 gives `0.26`. The table target keeps the table value because this ratchet is table-row extraction.
- Unified the PF-1000 16 kV Akel waveform, phase, and table targets under validation scope `pf1000_16kv_2021_akel`.
- Scientific closure gained: scalar current, fitted-parameter, pinch-geometry, and neutron-yield ensemble targets can now be compared shot-by-shot without re-reading the paper text.
- Scientific closure still missing: digitized current traces with per-point uncertainty, phase-transition timing traces, neutron timing, neutron spectrum, neutron anisotropy, detector response, and a blind-prediction acceptance rule. This target strengthens scalar/yield validation but does not close high-fidelity neutron-predictive simulation.
- Verification status: `python3 -m py_compile src/dpf/validation/kr_targets.py src/dpf/validation/__init__.py tests/test_kr_targets.py` passed; focused Akel/KR target tests passed (`6 passed in 0.92s`); `python3 -m pytest tests/test_kr_targets.py tests/test_kr_corpus.py -q` passed (`79 passed in 1.29s`); `python3 -m pytest tests/test_quality_assessment.py::TestQualityAssessment::test_scientific_accuracy_gap_report_lists_remaining_work tests/test_kr_targets.py::test_kr_validation_same_scope_target_report_requires_one_scope -q` passed (`2 passed in 0.68s`); `git diff --check` is clean.

Ratchet update 2026-05-06, Akel 2021 scalar-table evidence comparator:

- Modules touched: `src/dpf/validation/kr_targets.py`, `src/dpf/validation/__init__.py`, `tests/test_kr_targets.py`, `CodexFindings.md`, and `CortexFindings.md`.
- Added `pf1000_16kv_akel_table_candidate_evidence()` so a simulation can be checked against the 24 Akel scalar/yield rows instead of treating the new table target as passive metadata.
- The comparator accepts either shot-keyed mappings or row lists. It checks all 24 required shots by default across peak current, pinch current, axial speed, shock speed, piston speed, pinch density, pinch radius, pinch length, and neutron yield.
- Neutron-yield comparison defaults to measured neutron yield as the validation target; the paper's computed Lee yield remains in the target rows as source context.
- The evidence records missing shots, extra shots, missing fields, per-field pass/fail, maximum relative errors, and per-shot/per-field relative errors.
- Scientific boundary: a passing comparator result is scalar table agreement only. It is not waveform validation, phase-timing validation, neutron timing, neutron spectrum, anisotropy, detector response, or a KR-sourced blind-prediction acceptance criterion.
- Verification status: `python3 -m py_compile src/dpf/validation/kr_targets.py src/dpf/validation/__init__.py tests/test_kr_targets.py` passed; focused comparator tests passed (`3 passed in 0.70s`); `python3 -m pytest tests/test_kr_targets.py tests/test_kr_corpus.py -q` passed (`81 passed in 1.24s`); `python3 -m pytest tests/test_kr_targets.py tests/test_kr_corpus.py tests/test_quality_assessment.py::TestQualityAssessment::test_scientific_accuracy_gap_report_lists_remaining_work -q` passed (`82 passed in 1.47s`); broader validation slice passed (`230 passed, 3 skipped in 12.36s`); `git diff --check` is clean.

Ratchet update 2026-05-06, Tier-5 scalar-yield closure gate:

- Modules/docs touched: `src/dpf/validation/quality_assessment.py`, `src/dpf/validation/kr_targets.py`, `tests/test_quality_assessment.py`, `tests/test_kr_targets.py`, `tests/test_mhd_physics_integration.py`, `docs/joss-paper-draft.md`, `docs/AI_DISCLOSURE.md`, `CodexFindings.md`, and `CortexFindings.md`.
- Tightened `neutron_validation_scope_closure_report()` so Tier 5 now requires same-scope neutron scalar-yield validation in addition to mechanism/timing, spectrum, and anisotropy.
- Added `neutron_yield_validation` to the result-level source-authority audit keys.
- Updated `validation_tier_report()` and `predictive_readiness_report()` wording from neutron timing/spectrum/anisotropy closure to neutron yield/mechanism/timing/spectrum/anisotropy closure.
- Updated `pf1000_16kv_akel_table_candidate_evidence()` so a passing scalar-yield comparison advertises `validated_features={"yield": True}` and can serve as the yield component of a same-scope Tier-5 packet.
- Scientific closure gained: a result can no longer become Tier-5 supported with neutron timing/spectrum/anisotropy alone. A KR-backed scalar yield comparison must be in the same validation scope.
- Scientific closure still missing: no production app path currently emits a complete same-scope neutron-yield validation packet together with timing, spectrum, anisotropy, and detector response. MJOLNIR helper output correctly drops back to `decomposed_estimate` until scalar yield evidence is attached.
- Verification status: `python3 -m py_compile src/dpf/validation/quality_assessment.py src/dpf/validation/kr_targets.py tests/test_quality_assessment.py tests/test_kr_targets.py tests/test_mhd_physics_integration.py` passed; focused Tier-5/yield tests passed (`6 passed in 0.90s`); `python3 -m pytest tests/test_quality_assessment.py tests/test_kr_targets.py tests/test_mhd_physics_integration.py -q` passed (`171 passed, 3 skipped in 8.84s`); broader validation slice passed (`230 passed, 3 skipped in 12.49s`); `git diff --check` is clean.

Ratchet update 2026-05-06, neutron-yield KR target group:

- Modules/docs touched: `src/dpf/validation/kr_targets.py`, `src/dpf/validation/source_acquisition.py`, `tests/test_kr_targets.py`, `tests/test_digitization.py`, `docs/SCIENTIFIC_CLOSURE_SOURCE_QUEUE.md`, `CodexFindings.md`, and `CortexFindings.md`.
- Added `neutron_yield` to `_END_TO_END_TARGET_GROUPS`, `_typed_observable_groups()`, same-scope closure blocker records, and the source-acquisition queue.
- Made PF-1000 full-energy yield context explicit in `pf1000_full_energy_neutron_spatial_2007_scholz` with scalar yield range, maximum yield, activation/angle context, and detector-response dependency.
- Marked neutron-yield groups partial when predictive-yield blockers are present. Current PF-1000 full-energy blockers include `yield_calibration_uncertainty`, `neutron_field_transport_or_room_scatter_response_model`, and `fast_ion_distribution_uncertainty`.
- Added priority-1 `neutron_yield` source-acquisition items. Akel 2021 is now listed as the local shot-resolved scalar-yield source, while Klir 2011 remains an acquisition lead for detector timing/sensitivity calibration needed to close predictive yield.
- Scientific closure gained: the KR target coverage plan, same-scope closure report, source-acquisition queue, and Tier-5 readiness gate now agree that scalar neutron yield is a first-class closure requirement.
- Scientific closure still missing: scalar yield remains partial for the widest PF-1000 full-energy scope because calibration/response and fast-ion uncertainty are not closed in the same scope.
- Verification status: `python3 -m py_compile src/dpf/validation/kr_targets.py src/dpf/validation/source_acquisition.py tests/test_kr_targets.py tests/test_digitization.py` passed; `python3 -m pytest tests/test_kr_targets.py tests/test_kr_corpus.py tests/test_digitization.py tests/test_quality_assessment.py::TestQualityAssessment::test_scientific_accuracy_gap_report_lists_remaining_work -q` passed (`87 passed in 1.90s`); broader validation slice passed (`230 passed, 3 skipped in 12.30s`); `git diff --check` is clean.

Ratchet update 2026-05-06, app-level Akel scalar-yield validation hook:

- Modules touched: `app_mhd.py`, `tests/test_mhd_physics_integration.py`, `CodexFindings.md`, and `CortexFindings.md`.
- Added an explicit PF-1000 16 kV Akel table hook in `_apply_post_processing()`. If a result supplies `pf1000_16kv_akel_table_predictions`, `akel_2021_table_predictions`, or `neutron_yield_validation_rows`, and the run is PF-1000 at 16 kV, the app compares the full table with `pf1000_16kv_akel_table_candidate_evidence()`.
- Passing 24-shot scalar/yield rows are promoted to `neutron_yield_validation`; failing or incomplete rows remain `neutron_yield_validation_candidate`.
- Updated app-level neutron closure so `neutron_yield_validation` alone triggers `neutron_validation_scope_closure`, making missing timing/spectrum/anisotropy explicit instead of invisible.
- Scientific closure gained: production result dictionaries now have a path to carry KR-backed scalar-yield validation evidence, not just passive target metadata.
- Scientific closure still missing: this hook requires callers to supply the full Akel 24-shot prediction table. A single run's scalar yield is still not enough to validate predictive neutron performance.
- Verification status: `python3 -m py_compile app_mhd.py tests/test_mhd_physics_integration.py` passed; focused app Akel/Tier-5 tests passed (`3 passed in 1.72s`); broader validation slice passed (`232 passed, 3 skipped in 13.01s`); `git diff --check` is clean.

Ratchet update 2026-05-06, Akel 2021 figure digitization queue:

- Modules/docs touched: `src/dpf/validation/digitization.py`, `src/dpf/validation/__init__.py`, `tests/test_digitization.py`, `docs/SCIENTIFIC_CLOSURE_SOURCE_QUEUE.md`, `CodexFindings.md`, and `CortexFindings.md`.
- Added `scientific_closure_digitization_queue()` as a machine-readable local queue for the remaining Akel 2021 figures.
- The queue tracks six figure tasks tied to KR line windows and the parity-verified Akel PDF SHA-256:
  - Fig. 1 current waveform, shot 12581, 1.2 Torr, source lines 294-295.
  - Fig. 2 current waveform, shot 12584, 1.2 Torr, source lines 296-297.
  - Fig. 3 current waveform, shot 12592, 1.05 Torr, source lines 298-299.
  - Fig. 4 current waveform, shot 12604, 1.05 Torr, source lines 300-301.
  - Fig. 5 neutron-yield plot at 1.2 Torr, source line 916.
  - Fig. 6 neutron-yield plot at 1.05 Torr, source line 917.
- Each task records required series, page hints, local PDF candidates, the KR markdown SHA-256, the Akel PDF SHA-256, the text-parity flag, the required `digitization_verification_evidence()` gate, and the fact that no figure image has been extracted yet.
- Scientific closure gained: the remaining Akel waveform/yield plot work is now a tested local queue rather than an open prose note.
- Scientific boundary: this queue is not evidence. The current-waveform figures still need page rendering, crop/hash capture, axis calibration, per-series extraction, overlay residuals, and independent review before they can validate a run.
- Verification status: `python3 -m py_compile src/dpf/validation/digitization.py src/dpf/validation/__init__.py tests/test_digitization.py` passed; focused digitization/KR/quality tests passed (`89 passed in 1.88s`); broader validation slice passed (`234 passed, 3 skipped in 13.47s`); `git diff --check` is clean.

Ratchet update 2026-05-06, digitization queue acceptance status:

- Modules/docs touched: `src/dpf/validation/digitization.py`, `src/dpf/validation/__init__.py`, `tests/test_digitization.py`, `docs/SCIENTIFIC_CLOSURE_SOURCE_QUEUE.md`, `CodexFindings.md`, and `CortexFindings.md`.
- Added `scientific_closure_digitization_status()` so future Akel digitization packets can be evaluated against the local queue, not just against the generic packet gate.
- A task is accepted only if its packet passes `digitization_verification_evidence()` and also matches the queue task ID, KR source path, KR source hash, local PDF hash, source line window, figure ID, page, and required series names.
- The status report now separates accepted, failed, open, invalid, and extra packets and lists missing or failed task IDs.
- Scientific closure gained: the workflow now has a tested one-for-one acceptance method for the user's requested figure digitization process.
- Scientific boundary: no Akel figure packet is accepted yet. The current status with no packets is six open tasks.
- Verification status: `python3 -m py_compile src/dpf/validation/digitization.py src/dpf/validation/__init__.py tests/test_digitization.py` passed; `tests/test_digitization.py` passed (`10 passed in 0.44s`); focused digitization/KR/quality tests passed (`92 passed in 1.48s`); broader validation slice passed (`237 passed, 3 skipped in 12.80s`); `git diff --check` is clean.

Ratchet update 2026-05-06, app-level digitization closure export:

- Modules touched: `app_mhd.py`, `tests/test_mhd_physics_integration.py`, `CodexFindings.md`, and `CortexFindings.md`.
- App post-processing now exports `scientific_closure_digitization_queue` and `scientific_closure_digitization_status` on every result.
- If a caller supplies `scientific_closure_digitization_packets` or `digitization_packets`, the app evaluates them with `scientific_closure_digitization_status()`.
- Default production runs now explicitly report the Akel figure queue as open rather than hiding figure digitization outside the result metadata.
- Scientific closure gained: app outputs now carry the local figure-digitization blockers next to KR target coverage, corpus review status, predictive readiness, and high-fidelity readiness.
- Scientific boundary: this does not create or accept any Akel figure data. It only exposes the current open queue and future packet acceptance path.
- Verification status: `python3 -m py_compile app_mhd.py tests/test_mhd_physics_integration.py` passed; focused app/digitization tests passed (`11 passed in 1.02s`); broader validation slice passed (`237 passed, 3 skipped in 12.67s`); `git diff --check` is clean.

Ratchet update 2026-05-06, figure-digitization scientific-accuracy gap:

- Modules touched: `src/dpf/validation/quality_assessment.py`, `tests/test_quality_assessment.py`, `tests/test_mhd_physics_integration.py`, `CodexFindings.md`, and `CortexFindings.md`.
- Added `figure_digitization` to `scientific_accuracy_gap_report()`.
- The gap uses `scientific_closure_digitization_status` when present, or computes the default open queue status when absent.
- Status rules:
  - `supported` only when the local digitization queue is complete.
  - `partial` when at least one task is accepted and the rest remain failed or open.
  - `blocked` when no task is accepted or status is unavailable.
- App results now show `figure_digitization` as blocked by `0/6` accepted local scientific-closure figure tasks.
- High-fidelity readiness now treats accepted local figure digitization as part of scientific closure, while the synthetic all-supported readiness test can still pass when it supplies a complete digitization status.
- Scientific closure gained: open Akel figure digitization is now a first-class high-fidelity blocker, not just a sidecar queue.
- Scientific boundary: this still does not produce digitized data; it makes the missing data impossible to miss in readiness outputs.
- Verification status: `python3 -m py_compile src/dpf/validation/quality_assessment.py tests/test_quality_assessment.py tests/test_mhd_physics_integration.py` passed; targeted readiness/gap tests passed (`3 passed in 1.52s`); broader validation slice passed (`237 passed, 3 skipped in 13.60s`); `git diff --check` is clean.

Ratchet update 2026-05-06, app-level source-acquisition queue export:

- Modules touched: `app_mhd.py`, `tests/test_mhd_physics_integration.py`, `CodexFindings.md`, and `CortexFindings.md`.
- App post-processing now exports `scientific_closure_source_acquisition_queue`.
- Result payloads now carry candidate DOI links and required local-ingestion steps beside KR target coverage, corpus review status, digitization status, predictive readiness, and high-fidelity readiness.
- Scientific closure gained: the app now exposes the acquisition workflow the user requested: AI supplies source leads; the user acquires the correct document; the document becomes usable only after local `KnowledgeReference` ingestion, review, and any required digitization verification.
- Scientific boundary: source-acquisition candidates are still not evidence and do not satisfy any validation tier until the local KR process is complete.
- Verification status: `python3 -m py_compile app_mhd.py tests/test_mhd_physics_integration.py` passed; focused app/source-queue tests passed (`3 passed in 1.44s`); broader validation slice passed (`237 passed, 3 skipped in 13.10s`); `git diff --check` is clean.

Ratchet update 2026-05-06, local-vs-acquisition source queue split:

- Modules/docs touched: `src/dpf/validation/source_acquisition.py`, `tests/test_digitization.py`, `docs/SCIENTIFIC_CLOSURE_SOURCE_QUEUE.md`, `CodexFindings.md`, and `CortexFindings.md`.
- Source-acquisition queue entries now annotate each DOI with local status from the PDF parity audit.
- Queue items now separate:
  - `local_sources_available`
  - `candidate_sources_for_acquisition`
  - the compatibility field `candidate_sources`, with local status metadata on each source lead.
- Akel 2021, Gribkov 2007 Parts I/II, Schmidt 2022, Malir 2024, and Goyon 2025 are tagged as `parity_verified_knowledge_reference` with local KR paths and PDF hashes.
- Cikhardtova 2015, Sadowski/Scholz 2004, Catenacci 2020, Springham 2021, Klir 2011, and Jednorog 2017 are tagged as `not_found_as_exact_local_pdf`.
- Scientific closure gained: the queue no longer tells the user to acquire sources already verified locally. It points local sources toward target extraction or digitization and preserves true external acquisition leads separately.
- Scientific boundary: a parity-verified local source still does not close a validation group unless typed targets or verified digitized data exist for the missing observable.
- Verification status: `python3 -m py_compile src/dpf/validation/source_acquisition.py tests/test_digitization.py` passed; focused source-queue/app tests passed (`3 passed in 1.38s`); broader validation slice passed (`237 passed, 3 skipped in 12.87s`); `git diff --check` is clean.

Ratchet update 2026-05-06, Akel figure render page correction:

- Modules/docs touched: `src/dpf/validation/digitization.py`, `tests/test_digitization.py`, `docs/SCIENTIFIC_CLOSURE_SOURCE_QUEUE.md`, `CodexFindings.md`, and `CortexFindings.md`.
- Rendered the parity-verified Akel local PDF pages into `/private/tmp/dpf_akel_digitization` using `pdftoppm` as a non-evidence workbench.
- Corrected the digitization queue page hints:
  - Figs. 1-4 render on PDF page 3, not page 4.
  - Figs. 5-6 render on PDF page 5, not page 6.
- Page 4 is the typed table page; page 6 is references. The previous page hints would have sent future digitization to the wrong rendered pages.
- Scientific closure gained: the queue now points to the rendered pages that actually contain the cited Akel plots.
- Scientific boundary: the temporary page renders are not stored as KR evidence, are not accepted digitization packets, and do not close the `figure_digitization` gap.
- Verification status: `python3 -m py_compile src/dpf/validation/digitization.py tests/test_digitization.py` passed; `tests/test_digitization.py` passed (`10 passed in 0.76s`); broader validation slice passed (`237 passed, 3 skipped in 13.36s`); `git diff --check` is clean.

Ratchet update 2026-05-06, Akel scalar-yield uncertainty diagnostics:

- Modules touched: `src/dpf/validation/kr_targets.py`, `tests/test_kr_targets.py`, `CodexFindings.md`, and `CortexFindings.md`.
- `pf1000_16kv_akel_table_candidate_evidence()` now reports neutron-yield absolute error, the source-reported measured-yield uncertainty for each row, and the measurement-uncertainty-normalized error.
- The evidence summary now includes `max_measurement_uncertainty_normalized_error` across all compared neutron-yield rows.
- Scientific closure gained: scalar-yield comparison now exposes the uncertainty scale printed in Akel Table 2, instead of reporting only software relative error.
- Scientific boundary: this is still a scalar table comparator. The pass/fail threshold remains the explicit software tolerance; the source does not provide a blind-prediction acceptance criterion, and this does not close detector response, neutron timing, spectrum, or anisotropy.
- Verification status: `python3 -m py_compile src/dpf/validation/kr_targets.py tests/test_kr_targets.py` passed; focused comparator/app tests passed (`3 passed in 0.94s`); broader validation slice passed (`237 passed, 3 skipped in 13.30s`); `git diff --check` is clean.

Ratchet update 2026-05-06, PF-1000 16 kV candidate scope consistency:

- Modules touched: `src/dpf/validation/kr_targets.py`, `tests/test_kr_targets.py`, `tests/test_mhd_physics_integration.py`, `CodexFindings.md`, and `CortexFindings.md`.
- Fixed `pf1000_16kv_phase_candidate_evidence_from_history()` and `pf1000_16kv_derived_output_candidate_evidence()` to report `validation_scope="pf1000_16kv_2021_akel"` instead of the individual target ID.
- App-level PF-1000 16 kV phase and derived-output candidates now carry the same Akel validation scope as the waveform and table targets.
- Scientific closure gained: scope identity is now consistent across Akel phase, derived-output, waveform, and scalar-yield candidate evidence, which is required for same-scope closure accounting.
- Scientific boundary: these phase and derived-output packets are still candidate/partial evidence because the KR record lacks complete measured axial, radial, and pinch phase endpoints.
- Verification status: `python3 -m py_compile src/dpf/validation/kr_targets.py tests/test_kr_targets.py tests/test_mhd_physics_integration.py` passed; focused scope tests passed (`3 passed in 0.84s`); broader validation slice passed (`237 passed, 3 skipped in 12.92s`); `git diff --check` is clean.

Ratchet update 2026-05-06, Akel phase-semantics target:

- Modules touched: `src/dpf/validation/kr_targets.py`, `tests/test_kr_targets.py`, `CodexFindings.md`, and `CortexFindings.md`.
- Added `phase_semantics` to `pf1000_16kv_shot12581_phase_2021_akel`.
- The target now records that Akel's fitted Lee factors map to axial phase mass/current (`fm`, `fc`) and radial phase mass/current (`fmr`, `fcr`) semantics, with current waveform fitting driving the phase-dynamics outputs.
- Same-scope target reporting now marks `phase_semantics` present for `pf1000_16kv_2021_akel` instead of missing.
- Scientific closure gained: the Akel 16 kV scope no longer has a false missing phase-semantics blocker.
- Scientific boundary: phase timing remains partial because the source does not provide complete measured axial, radial, and pinch endpoint timings with uncertainty.
- Verification status: `python3 -m py_compile src/dpf/validation/kr_targets.py tests/test_kr_targets.py` passed; focused target/same-scope tests passed (`2 passed in 0.60s`); broader validation slice passed (`237 passed, 3 skipped in 12.65s`); `git diff --check` is clean.

Ratchet update 2026-05-06, Akel table uncertainty target:

- Modules touched: `src/dpf/validation/kr_targets.py`, `tests/test_kr_targets.py`, `CodexFindings.md`, and `CortexFindings.md`.
- Added an explicit `uncertainty` block to `pf1000_16kv_shot_table_2021_akel`.
- The target now records that measured neutron-yield uncertainty is available per row, with a row uncertainty range of `2.0e7` to `2.0e8` neutrons/shot.
- Added missing uncertainty components for current trace uncertainty, detector-response uncertainty, model-form uncertainty, input-parameter covariance, and blind-prediction acceptance.
- Same-scope reporting now marks `uncertainty` present but partial for `pf1000_16kv_2021_akel`.
- Scientific closure gained: Akel's scalar yield uncertainty is now represented as typed KR target data, not only comparator output.
- Scientific boundary: this is not a full uncertainty budget. It lacks waveform uncertainty, detector response/systematics, model-form uncertainty, input covariance, and a source-backed acceptance rule.
- Verification status: `python3 -m py_compile src/dpf/validation/kr_targets.py tests/test_kr_targets.py` passed; focused table/same-scope tests passed (`2 passed in 0.43s`); broader validation slice passed (`237 passed, 3 skipped in 12.67s`); `git diff --check` is clean.

## Concise argument

The KnowledgeReference corpus supports a narrow claim: DPF rundown and some early pinch behavior can be modeled with Lee/snowplow or MHD-like reduced models, and circuit waveforms can be fit or predicted at that level for some devices. The corpus does not support a broad claim that this repository currently predicts final pinch, beam formation, high-Z radiation, or total neutron yield from a validated MHD model. Multiple KnowledgeReference documents state that late pinch and neutron production require kinetic, beam-target, instability, 3D, EOS, and radiation effects that the project either lacks, treats empirically, or explicitly bypasses.

Therefore my argument is:

For: use `dpf-unified` as a circuit/Lee/snowplow and diagnostics platform with promising MHD infrastructure.

Against: use `dpf-unified` today as a validated Dense Plasma Focus machine simulation tool for predictive design claims, especially for neutron yield, high-Z dopants, p-B11, final pinch structure, or MHD-driven current prediction.

## Definition-of-done status for this review

Completed:

- Reviewed README claims, core engine coupling, pure-MLX engine, snowplow implementations, diagnostics, radiation modules, and validation tests.
- Cross-checked scientific claims against the local KnowledgeReference corpus only.
- Created this running findings file for future updates.
- Began ratchet-loop implementation, updating this file after each completed module.
- Completed bounded physics ratchets for MLX circuit coupling, MLX snowplow scope, fluid/Lee reflected-shock density, D2 cold-fill pressure, post-pinch empirical-resistance reporting, neutron-yield decomposition, scaling-law metadata, radiation provenance, and p-B11 provenance.
- Completed source-authority ratchets for circuit waveform validation, raw waveform metrics, raw neutron-yield metrics, validation-ready device registry queries, V&V reports, app-level predictive-readiness export, validation-error reporting, and readiness blocking on pipeline errors.
- Completed tier-gate hardening for circuit, snowplow phase/timing, MHD verification, spatial DPF validation, and neutron timing/spectrum/anisotropy evidence. Placeholder or unsourced dictionaries no longer support predictive readiness.
- Completed a structured scientific-accuracy gap report and app export so every result can carry the remaining blockers, next ratchet, and done condition for high-fidelity predictive readiness.
- Began tier-2 target curation by adding Lee/RADPF phase semantics and a PF-1000 16 kV shot-12581 partial phase target that remains candidate-only until complete same-shot phase timing is available; app post-processing now exports that candidate for matching PF-1000 16 kV phase histories.
- Added a strict high-fidelity readiness gate that requires the five validation tiers and the scientific-accuracy gap list to be closed before a result can be marked high-fidelity ready.
- Added high-fidelity audit records so every app result can report missing or unvalidated EOS, ionization, two-temperature, radiation transport, impurity, kinetic, 3D, startup, restrike, beam-target, circuit-field coupling, uncertainty-budget, and MHD numerical-fidelity evidence.
- Completed claim-surface cleanup for `README.md`, `docs/SCOPE.md`, `docs/V_AND_V_SUMMARY.md`, `docs/joss-paper-draft.md`, and `docs/AI_DISCLOSURE.md`, with regression tests for the withdrawn validation claims.
- Added corpus-review status and triage controls: the local corpus currently has 827 files, 398 markdown files, 396 JSON files, 54 DPF-named markdown files, 42 coded target records, 38 unique coded KR source files, 37 DPF-named files represented by coded targets, 17 DPF-named files review-closed by explicit decisions, and 0 DPF-named files still unreviewed.
- Added the PF-1000 full-energy 2007 target bundle, giving the project a broad but still partial same-scope PF-1000 packet with every required target group present but not yet complete.
- Current registry status under the KR-only rule: `get_validation_ready_devices()` returns only `PF-1000`; POSEIDON-60kV and UNU-ICTP remain blocked because their waveform arrays are external archive traces, not KR-verified waveforms.
- Current verification snapshot: the prior focused ratchet regression slice passed (`93 passed in 0.82s`); the validation/app/KR-target slice passed (`66 passed in 0.78s`); the spatial/neutron-scope/audit/readiness slice passed (`91 passed in 1.18s`); the detector-response/high-fidelity gap slice passed (`77 passed in 1.24s`); the PF-1000 derived-output candidate slice passed (`36 passed in 0.62s`); the uncertainty-coverage slice passed (`8 passed in 0.69s`); the Lee dynamic-inductance power-accounting slice passed (`7 passed in 0.67s`); the MHD numerical-method metadata slice passed (`6 passed in 0.68s`); the cylindrical convergence evidence slice passed (`8 passed in 0.70s`); the resistive diffusion evidence slice passed (`11 passed in 0.45s`); the circuit-coupled energy slice passed (`21 passed in 0.45s`); the backend parity slice passed (`15 passed in 0.44s`); the finite-volume MHD channel slice passed (`16 passed in 0.45s`); the MHD phase/scope-limit slice passed (`19 passed in 0.75s`); the MHD numerical-fidelity closure-path slice passed (`21 passed in 0.55s`); the high-fidelity physics-effect evidence slice passed (`6 passed in 0.69s`); the uncertainty-budget component evidence slice passed (`9 passed in 0.46s`); the field-coupling component evidence slice passed (`11 passed in 0.43s`); the source-authority evidence slice passed (`2 passed in 0.58s`); the high-fidelity readiness closure-path slice passed (`1 passed in 0.50s`); the app-level MHD scope limiter slice passed (`1 passed in 0.73s`); the result-derived source-authority slice passed (`3 passed in 0.84s`); the failed-source-authority blocker slice passed (`3 passed in 1.08s`); the manual-packet cross-check slice passed (`3 passed in 0.72s`); the local-KR-file source-authority slice passed (`5 passed in 0.77s`); the source-line-range validation slice passed (`5 passed in 0.51s`); the focused KR target/corpus slice passed (`76 passed in 0.54s`); semantic/source audits passed; the current quality/KR target/corpus slice passed (`127 passed in 0.65s`); the current app/MHD/coupling/physics/UQ slice passed (`87 passed, 3 skipped in 1.80s`); `git diff --check` is clean; Python syntax compilation passed for the touched validation/KR-target/KR-corpus/test modules.

Still not done:

- External literature/web lookup has been used only to identify candidate acquisition links. Those links are not source-of-truth evidence until the user acquires the document, it is placed under `KnowledgeReference/`, and it passes local KR review.
- No new experimental calibration, new measured waveform ingestion, or same-shot spatial DPF validation was attempted.
- Tier 2 still needs KR-extracted device/shot phase targets for ordinary production runs.
- Tier 4 still needs same-scope density, magnetic-field, and temperature comparisons from KnowledgeReference-backed diagnostics.
- Tier 5 still needs calibrated production outputs for neutron timing, spectrum, anisotropy, and detector/activation response, not only helper comparisons on supplied arrays.
- High-fidelity predictive status still requires implementing or explicitly bounding EOS, ionization, two-temperature, radiation transport/opacities, high-Z/impurity/ablation, Hall/FLR/kinetic/PIC, 3D instability, flashover, restrike, and anomalous-resistance physics for the claimed scope.
- Full uncertainty budgets still need to propagate through phase, spatial, neutron, and numerical evidence, not only circuit waveform comparisons.
- The project is still not a validated end-to-end predictive DPF simulation tool; the completed ratchets narrow, harden, and label gaps rather than close the kinetic/3D/EOS/radiation-transport validation problem.

Ratchet update 2026-05-06, MLX collection abort hardening:

- Modules touched: `src/dpf/metal/mlx_device.py`, `src/dpf/metal/device.py`, `src/dpf/metal/mlx_amr.py`, `src/dpf/metal/mlx_kernels.py`, `tests/conftest.py`, `tests/test_amr_mlx.py`, `tests/test_mlx_device.py`, `tests/test_mlx_species.py`, `tests/test_mlx_sts.py`, `tests/test_saha_eos.py`, `tests/test_validation_ci.py`, `tests/test_pic_validation.py`, `CodexFindings.md`, and `CortexFindings.md`.
- Issue identified: this local environment does not merely lack MLX; importing `mlx.core` aborts the Python interpreter. That means `pytest.importorskip("mlx.core")` cannot protect collection if the import is attempted in-process.
- Local package/runtime metadata: `mlx` is installed as version `0.31.0` under Python `3.11.9` on `macOS-26.3.1-arm64-arm-64bit`, but the safe project detector reports `HAS_MLX=False` and `mlx_device_info()` reports unavailable because the child-process import probe fails.
- Original abort path: collecting `tests/test_amr_mlx.py` imported `dpf.metal.mlx_amr`; package import entered `dpf.metal.__init__`, then `dpf.metal.mlx_device`, where the module-level `import mlx.core` aborted Python before pytest could skip. Full collection later exposed the same eager-import problem in `dpf.metal.mlx_kernels`.
- Fix implemented: MLX availability now uses a child-process probe in `mlx_device.py`, so a broken native MLX import can kill only the probe process, not pytest or the application import path. `require_mlx()` refuses to import in-process unless that probe succeeds.
- Follow-on hardening: `DeviceManager.detect_mlx()` and `has_mlx()` now delegate to the safe detector; AMR and kernel modules no longer raw-import MLX on import when the safe probe fails; MLX tests and mixed CPU/MLX tests now use safe `HAS_MLX` gates.
- Test-collection hardening: `tests/conftest.py` wraps `pytest.importorskip("mlx.core")` so legacy MLX test files skip through the safe detector instead of attempting the native import directly.
- Scientific status: this is infrastructure hardening, not scientific closure. It lets non-MLX KR validation and CPU fallback tests run in an environment with broken MLX, but it does not validate the MLX/Metal physics backend. MLX-specific predictive claims remain skipped here until MLX imports cleanly and the backend tests execute.
- Verification status: targeted MLX/mixed tests passed or skipped cleanly (`15 passed, 28 skipped in 0.30s`); full pytest collection completed without abort (`3657/3775 tests collected, 118 deselected in 3.11s`); broader KR/validation slice passed (`237 passed, 3 skipped in 12.97s`); `git diff --check` is clean.

Ratchet update 2026-05-06, MLX runtime triage and scientific gate correction:

- Modules touched: `tests/test_mlx_circuit_coupling.py`, `tests/test_mlx_pf1000.py`, `CodexFindings.md`, and `CortexFindings.md`.
- MLX collection issue explained: the installed MLX package is not the root problem. The local package is `mlx==0.31.0` / `mlx-metal==0.31.0`, and outside the sandbox `import mlx.core as mx; print(mx.default_device())` reports `Device(gpu, 0)`. Inside the sandbox, Metal enumeration returns zero devices (`MTLCopyAllDevices()` count `0`, `MTLCreateSystemDefaultDevice()` `nil`), and `mlx.core` aborts natively during Metal device construction before Python can catch or skip it.
- MLX troubleshooting result: reversible environment mitigations did not fix the sandbox import abort (`MLX_DEFAULT_DEVICE=cpu`, `MLX_DEVICE=cpu`, `MLX_DISABLE_METAL=1`, and `MLX_DISABLE_COMPILE=1` still abort before a CPU fallback can be selected). The correct local split is therefore:
  - sandbox collection and non-MLX validation must use the safe child-process detector and skip MLX imports;
  - real MLX/Metal tests must run in a Metal-visible process outside the sandbox.
- Scientific correction: `tests/test_mlx_circuit_coupling.py` now asserts that `B_theta` is stronger at smaller radius. This follows the on-file source-of-truth relation `B_theta = mu I / (2*pi*r)` in `KnowledgeReference/plasma-formulary.md:2470-2473` and the DPF boundary relation `Bphi = mu I / (2*pi*r)` in `KnowledgeReference/two-dimensional-simulation-of-dense-plasma-focus-5.md:78-84`.
- PF-1000 full-discharge gate correction: the `TestMLXPF1000MustHave` and `TestMLXPF1000ShouldHave` classes are now marked `xfail(run=False)` because the class fixture is the documented full-discharge gate and the project docs mark M6 as blocked by CFL/full-duration stability (`docs/SPRINT4_VALIDATION_REVIEW.md:105-113`; `docs/METAL_V2_DOD.md:330-337`). Fast config-level checks in the same file still run.
- Scientific status:
  - Closed: MLX no longer prevents collection in the sandbox, real MLX import and tests execute when Metal is visible, and a scientifically reversed `B_theta` test has been corrected.
  - Still open: PF-1000 full-discharge MLX acceptance is not scientifically closed. A partial run that stops before the M6 duration cannot be counted as five-phase predictive validation. The M6/CFL closure plan remains: make the solver reach the required post-peak/full-discharge interval, then re-enable M1-M8/S1-S3 as real running gates, with no `xfail(run=False)`.
- Verification status: Python syntax compilation passed for the touched MLX hardening/test files; `git diff --check` is clean; sandbox full collection now completes (`4228/4346 tests collected, 118 deselected in 9.91s`); the standing KR validation slice passed (`237 passed, 3 skipped in 13.40s`); outside-sandbox targeted MLX tests passed (`139 passed in 0.97s`); outside-sandbox full MLX glob passed with the scientifically blocked PF-1000 gate recorded as xfail (`553 passed, 19 xfailed in 50.43s`); `tests/test_mlx_pf1000.py` alone reports `4 passed, 14 xfailed in 0.97s`.

Current next plan after this fix:

- Close PF-1000 M6/CFL duration stability before claiming full-discharge MLX validation.
- Re-enable the blocked PF-1000 full-discharge classes only after the fixture reaches the required post-peak/full-discharge interval without the fixed 20000-step cap.
- Add or extract same-scope KR-backed PF-1000 spatial state and neutron diagnostics before any high-fidelity neutron-predictive claim.
- Keep the sandbox safe detector in place even after MLX is healthy, because native optional dependencies must not be able to abort collection.

Ratchet update 2026-05-06, PF-1000 MLX probe stability through 10000 steps:

- Modules touched: `src/dpf/metal/mlx_primitives.py`, `src/dpf/metal/mlx_state.py`, `src/dpf/metal/mlx_solver.py`, `tests/test_mlx_primitives.py`, `tests/test_mlx_state.py`, `CodexFindings.md`, and `CortexFindings.md`.
- Issue isolated: the rerun 1900-step probe passed, but the 2200-step probe found a deterministic `pressure` NaN at engine step `1985` (`t=0.690156 us`). After fixing pressure unpacking, the next run exposed the underlying conservative-state overflow as non-finite `B` at step `1986`.
- Fix implemented: dual-energy pressure recovery now sanitizes total-energy and entropy pressure candidates before blending, so an unused infinite candidate cannot produce `NaN` through `0*inf`. `MLXState.to_state_dict()` uses the same finite blend for cylindrical and Cartesian unpacking.
- Fix implemented: the MLX solver's CPU-side post-hyperbolic floor now rebuilds momentum from bounded velocity after density flooring instead of multiplying momentum by `_rho_floor/rho` in vacuum cells. CPU-side energy and vacuum `B_theta` prescription bookkeeping now use finite float64 intermediates before returning to MLX float32.
- Operational MLX float32 rule retained for future fixes: when another MLX float32 nonfinite or overflow appears, first test the same repair pattern before adding narrow clamps: perform CPU-side repair bookkeeping in float64, recover finite primitive-like quantities, rebuild conserved fields from bounded finite values, clip only for representability, and then return to MLX float32. Do not multiply conserved components by huge density-floor ratios in vacuum cells; rebuild from finite velocity, pressure/energy, and magnetic components instead.
- Verification status: Python syntax compilation passed for touched MLX files and tests; focused MLX regressions passed (`2 passed in 0.73s`); short PF-1000 probe passed at 600 steps (`1 passed in 16.79s`); former failure-window probe passed at 2200 steps (`1 passed in 48.60s`); original long probe passed at 3000 steps, reaching `t=0.868989 us`, `I=0.671730 MA`, and `max_B=1.164430` (`1 passed in 65.62s`); extended probe passed at 5000 steps, reaching `t=1.141062 us`, `I=0.872277 MA`, and `max_B=1.512073` (`1 passed in 110.73s`); longer probe passed at 10000 steps, reaching `t=1.584247 us`, `I=1.187937 MA`, and `max_B=2.059263` (`1 passed in 218.22s`).
- Late-window native-abort triage update: a first 20000-step probe attempt exited at native/process level with code `-1` before the first 2000-step checkpoint, with no Python faulthandler traceback and no probe assertion. A dense 2000-step rerun passed with `DPF_MLX_PROBE_PRINT_INTERVAL=100`, reaching `t=0.747684 us`, `I=0.580770 MA`, and `max_B=1.006752`. An exact 20000-step rerun then reached step 18000 (`t=2.075847 us`, `I=1.521106 MA`, `max_B=2.636804`) before another native/process-level `-1` exit before step 20000. The Python nonfinite state checks did not fire, so the current hypothesis is late-window MLX/Metal runtime stability or cache/resource pressure rather than a caught Python-level NaN.
- Probe instrumentation update: `tests/test_mlx_pf1000_probe.py` now supports `DPF_MLX_PROBE_MEMORY=1` for `mlx_active_MB`, `mlx_cache_MB`, and `mlx_peak_MB` telemetry, plus `DPF_MLX_PROBE_CLEAR_CACHE_INTERVAL=N` for periodic `mlx.clear_cache()`. A cache-clearing 20000-step run with `DPF_MLX_PROBE_CLEAR_CACHE_INTERVAL=1000` and `DPF_MLX_PROBE_MEMORY=1` reached step 12000 (`t=1.709192 us`, `I=1.274353 MA`, `max_B=2.209062`, `mlx_active_MB=0.288`, `mlx_cache_MB=10.770`, `mlx_peak_MB=9.801`) and then exited with native code `-1` before step 14000. Periodic `mlx.clear_cache()` did not remove the late-window native abort.
- Dense-window and crash-report evidence: a bounded run with `DPF_MLX_PROBE_PRINT_START=12000` and `DPF_MLX_PROBE_PRINT_START_INTERVAL=25` exited natively after the first printed step, so the abort is not fixed to one deterministic late field value. Fresh macOS crash reports under `~/Library/Logs/DiagnosticReports/Python-2026-05-06-*.ips` show native infrastructure failures: the latest MLX-related report has `SIGABRT`, `NSRangeException`, `-[__NSArray0 objectAtIndex:]: index 0 beyond bounds for empty array`, with the backtrace in `mlx::core::metal::Device::Device()`, and a separate report shows `crashed on child side of fork pre-exec`. A fresh direct MLX initialization check after those reports still passed and reported `Device(gpu, 0)`.
- Standalone probe isolation: added `scripts/run_mlx_pf1000_probe.py`, which bypasses pytest/conftest and sets `DPF_MLX_ASSUME_AVAILABLE=1` only after directly importing `mlx.core` in the Metal-visible process. `mlx_device.py` now supports this explicit opt-in for already-validated Metal-visible processes; do not use it for sandboxed collection because it bypasses the protective child import probe. The standalone probe passed 2000 steps with memory telemetry, reaching `t=0.747684 us`, `I=0.580770 MA`, `max_B=1.006752`, `mlx_active_MB=0.288`, `mlx_cache_MB=10.525`, and `mlx_peak_MB=9.801`.
- Standalone 20000-step cap result: the standalone runner passed the 20000-step cap, reaching `t=2.200558 us`, `I=1.602652 MA`, `max_B=2.778162`, `mlx_active_MB=0.288`, `mlx_cache_MB=10.238`, and `mlx_peak_MB=9.801`. This indicates the prior `-1` exits were tied to pytest/conftest/subprocess behavior or local process-spawn/device-discovery instability rather than a deterministic MLX solver-state failure.
- Verification after standalone isolation: `python3 -m py_compile src/dpf/metal/mlx_device.py tests/test_mlx_pf1000_probe.py scripts/run_mlx_pf1000_probe.py` passed; `python3 -m pytest tests/test_mlx_device.py -q` passed (`21 passed`); `python3 -m pytest tests/test_mlx_primitives.py tests/test_mlx_state.py -q` passed (`61 passed`); `git diff --check` passed.
- Target-time gate update: `tests/test_mlx_pf1000.py` now uses named PF-1000 cap/target controls instead of a hidden `range(20000)`: `DPF_MLX_PF1000_STEP_CAP` and `DPF_MLX_PF1000_TARGET_US`. The target is increase-only and cannot be set below the M6 `6 us` requirement. The fixture records target, cap, and cap-exhaustion metadata on the engine so M6 reports `step cap reached before target` explicitly. Both PF-1000 probe paths now accept `DPF_MLX_PROBE_TARGET_US`; the pytest probe asserts if the target is not reached within `DPF_MLX_PROBE_STEPS`, while the standalone runner prints `CAP_EXHAUSTED` and returns exit code `2`.
- Verification after target-time update: `python3 -m pytest tests/test_mlx_pf1000.py -q` passed with blocked gates preserved (`4 passed, 14 xfailed`); standalone target success smoke passed with `DPF_MLX_PROBE_TARGET_US=0.00005`; standalone cap-exhaustion smoke returned code `2` with `CAP_EXHAUSTED steps=5 target_us=1.000000 final_t_us=0.243416`; focused MLX detector/pressure/state regressions passed (`82 passed`); `git diff --check` passed.
- M6 target-time probe: standalone `DPF_MLX_PROBE_TARGET_US=6` with `DPF_MLX_PROBE_STEPS=80000` reached the M6 target and exited `PASSED`. Checkpoints: step 10000 `t=1.584247 us`, step 20000 `t=2.200558 us`, step 30000 `t=2.812534 us`, step 40000 `t=3.427694 us`, step 50000 `t=4.066377 us`, step 60000 `t=4.711821 us`, and step 70000 `t=5.354566 us`. The pre-fix runner did not print the final target-hit step/time; the probe now prints final `PASSED steps=... final_t_us=...` and includes target-hit in the telemetry print condition.
- Remaining scientific limit: this closes the observed early PF-1000 MLX probe instability through the M6 `6 us` target on the standalone path, provided the cap is raised above the old 20000-step fixture limit. The PF-1000 full-discharge acceptance classes should remain blocked because the current waveform is not accepted: by step 70000 the current was still rising at `3.215728 MA`, far above the M2 nominal upper band, and S2 current-dip behavior is not demonstrated.
- Additional limit: the existing 20000-step fixture cap is not enough by itself. The standalone cap run reached only `2.200558 us`, still below the M6 `6 us` requirement, and current had already exceeded the M2 nominal upper band.

Ratchet update 2026-05-06, DoD source-of-truth audit:

- Modules/docs touched: `docs/METAL_V2_DOD.md`, `docs/METAL_V2_SPEC.md`, `docs/SPRINT4_VALIDATION_REVIEW.md`, `tests/test_mlx_pf1000.py`, `CodexFindings.md`, and `CortexFindings.md`.
- Source-of-truth correction: the Metal v2 DoD and architecture spec now explicitly separate engineering/numerical-method requirements from scientific validation gates. Scientific gates must name a local `KnowledgeReference/` target or typed KR manifest scope.
- PF-1000 scope correction: Akel 2021 16 kV (`pf1000_akel`) is now kept separate from the Scholz/Gribkov full-energy PF-1000 scope. Do not mix Akel 16 kV M2/S1/S2 acceptance with 27 kV/full-energy PF-1000 targets.
- M2 correction: for Akel shot 12581, the gate is `Ipeak = 1.165 MA +/- 10%`, i.e. `1.0485-1.2815 MA`, sourced to `KnowledgeReference/radiation-physics-and-chemistry-188-2021-109633.md` and the typed KR target. The older unspecified `1.2 MA`/`1.87 MA` references were marked or replaced as mixed-scope/stale.
- S1/S2 correction: Akel establishes measured current waveform figures and current derivative/dip timing context, but NRMSE and dip-depth acceptance remain blocked until accepted same-scope digitized current traces and uncertainty are present. Scalar tables/captions alone do not close waveform validation.
- M3 correction: the DoD no longer treats open PF-1000 discharge mass as exactly conserved. Closed-domain tests may still use `<5%` drift; open-discharge runs must report finite positive mass plus outflow/floor accounting.
- M6 correction: `12 us` is documented as a conservative Akel 16 kV engineering endurance gate, not a direct measured KR value. Full-discharge acceptance remains blocked until same-scope duration, M2, S1, and S2 are credible together.
- Latest Akel MLX probe evidence recorded: standalone `pf1000_akel` 40000-step run passed to `t = 3.238777 us` with `peak_I = 1.685154 MA` and stable memory telemetry, but this is above the Akel shot-12581 M2 upper bound and still rising.

Ratchet update 2026-05-06, Akel preset and axial pressure-coupling correction:

- Modules/tests touched: `src/dpf/presets.py`, `src/dpf/engine/circuit_coupling.py`, `src/dpf/metal/mlx_snowplow.py`, `src/dpf/metal/mlx_engine.py`, `tests/test_pf1000_akel_preset.py`, `tests/test_snowplow_consolidated.py`, `tests/test_mlx_snowplow.py`, `tests/test_mlx_pf1000_probe.py`, `scripts/run_mlx_pf1000_probe.py`, `CodexFindings.md`, and `CortexFindings.md`.
- Source-scope fix: `pf1000_akel` is now a shot-12581 preset, not an average/nominal Akel 24-shot preset. It uses Akel shot 12581 values from the typed KR target and local source lines: `p0=1.2 Torr`, `rho0=2.583e-4 kg/m^3`, `C0=1332 uF`, `V0=16 kV`, `L0=25 nH`, `r0=6.1 mOhm`, `fm=0.17`, `fc=0.70`, `fmr=0.26`, and `fcr=0.75`.
- Added `tests/test_pf1000_akel_preset.py` to compare `get_preset("pf1000_akel")` directly against `pf1000_16kv_shot12581_phase_targets()`, so future nominal/per-shot mixing fails fast.
- MLX reduced snowplow parity fix: `MLXSnowplow` now accepts and uses `radial_current_fraction` (`fcr`) separately from axial `current_fraction` (`fc`), and `run_mlx_discharge()` forwards the preset's radial current fraction. This keeps reduced MLX Lee/RADPF runs aligned with the CPU `SnowplowModel` parameter surface.
- Axial pressure-coupling fix: `_dynamic_sheath_pressure()` now returns configured cold fill pressure during axial rundown. The old path averaged MHD total ion+electron pressure ahead of the sheath and fed that into the Lee/RADPF snowplow as back-pressure. For Akel shot 12581 that inflated the intended `160 Pa` cold fill to about `640 Pa` at step 1, delaying rundown, reducing early `Lp`, and letting current overshoot.
- Evidence before pressure fix but after preset fix: standalone 40000-step `pf1000_akel` probe passed natively to `t=3.316852 us`, but current was still rising at `peak_I=1.367902 MA`, above the `1.2815 MA` M2 upper bound.
- Evidence after pressure fix: standalone 32000-step `pf1000_akel` probe passed to `t=2.971234 us` with `peak_I=0.977154 MA`. Checkpoints now match the reference snowplow trajectory: step 10000 `t=1.389409 us`, `I=0.678638 MA`, `phase=rundown`, `Lp=2.762352 nH`, `sheath_p=160 Pa`; step 20000 `t=2.028429 us`, `I=0.844696 MA`, `Lp=5.431736 nH`; step 30000 `t=2.806466 us`, `I=0.961079 MA`, `Lp=9.382222 nH`.
- Reduced reference check: `run_mlx_discharge(preset_name="pf1000_akel", mode="lee", max_steps=80000)` now peaks at `1.150685 MA` at `5.250577 us`, inside the Akel M2 band.
- Verification status: `python3 -m py_compile` passed for the touched preset/coupling/probe/test files; `tests/test_mlx_snowplow.py tests/test_pf1000_akel_preset.py` passed (`6 passed`); `tests/test_snowplow_consolidated.py::TestDynamicPressureFallback` passed (`9 passed`); Akel KR target checks passed (`2 passed`); `tests/test_mlx_pf1000.py -q` remains `4 passed, 14 xfailed`; `git diff --check` was clean before the final findings append.
- Post-fix M6 probe: standalone `pf1000_akel` with `DPF_MLX_PROBE_TARGET_US=6` and `DPF_MLX_PROBE_STEPS=90000` exited `PASSED`, reaching `t=6.000007 us` in `76948` steps. Final reported `peak_I=1.047183 MA` at `t=4.990339 us`. Checkpoints included step 40000 `t=3.576539 us`, `I=1.018345 MA`, step 50000 `t=4.283569 us`, `I=1.041204 MA`, step 60000 `t=4.927035 us`, `I=1.047142 MA`, and step 70000 `t=5.567840 us`, `I=1.044211 MA`.
- Post-fix 8 us radial/pinch probe: standalone `pf1000_akel` with `DPF_MLX_PROBE_TARGET_US=8` and `DPF_MLX_PROBE_STEPS=130000` exited `PASSED`, reaching `t=8.000045 us` in `107566` steps. Final reported `peak_I=1.047183 MA` at `t=4.990339 us`. New phase evidence: step 90000 `t=6.809701 us`, `phase=radial`, `I=1.015650 MA`, `r=12.387635 cm`, `Lp=34.725876 nH`; step 100000 `t=7.435049 us`, `phase=radial`, `I=0.923846 MA`, `r=6.068274 cm`, `Lp=44.316989 nH`, `dLdt=26.719455 nH/us`; final step 107566 `phase=pinch`, `I=0.739814 MA`, `r=2.863039 cm`, `Lp=54.412990 nH`, `dLdt=-15.836659 nH/us`. Memory telemetry stayed flat (`mlx_active_MB=0.288`, cache about `10.47 MB`, peak about `9.80 MB`).
- Post-fix 12 us endurance probe: standalone `pf1000_akel` with `DPF_MLX_PROBE_TARGET_US=12` and `DPF_MLX_PROBE_STEPS=220000` exited `PASSED`, reaching `t=12.000000 us` in `160418` steps. Final reported `peak_I=1.047183 MA` at `t=4.990339 us`. Post-8 us checkpoints stayed finite and memory-flat: step 110000 `t=8.171744 us`, `phase=pinch`, `I=0.716376 MA`, `r=3.442282 cm`, `Lp=51.936660 nH`, `dLdt=-13.171781 nH/us`; step 120000 `t=9.000231 us`, `I=0.643373 MA`, `r=6.237254 cm`, `Lp=43.947850 nH`; step 140000 `t=10.458063 us`, `I=0.641996 MA`, `r=11.155373 cm`, `Lp=36.134086 nH`; step 160000 `t=11.965820 us`, `I=0.520224 MA`, `r=15.200000 cm`, `Lp=31.976097 nH`; final step 160418 `I=0.517539 MA`, `r=15.200000 cm`, `Lp=31.976097 nH`. Memory telemetry stayed flat (`mlx_active_MB=0.288`, cache about `10.489 MB`, peak about `9.801 MB`).
- Late-voltage telemetry explanation before source-scope cleanup: `V_kV=0.000000` after about `11.19 us` was explained by the inherited fixed-time crowbar previously present in `pf1000_akel` (`crowbar_enabled=True`, `crowbar_mode="fixed_time"`, `crowbar_time=10.5e-6`). The local Akel source search did not find shot-scope crowbar timing support, so that post-10.5 us voltage/current behavior was engineering crowbar behavior, not same-scope Akel waveform evidence.
- Probe telemetry update: `scripts/run_mlx_pf1000_probe.py` and `tests/test_mlx_pf1000_probe.py` now print `crowbar` and `crowbar_t_us` fields so future `V_kV=0` checkpoints show whether the zero came from the configured crowbar. Verification: `python3 -m py_compile scripts/run_mlx_pf1000_probe.py tests/test_mlx_pf1000_probe.py` passed; 5-step standalone Akel smoke passed and printed `crowbar=0 crowbar_t_us=-1.000000`.
- Remaining limit before source-scope cleanup: this fixed the identified M2 overshoot mechanism and closed the standalone `6 us`, `8 us`, and conservative `12 us` engineering endurance targets through post-pinch expansion. Full-discharge acceptance remained blocked because strict M2 was still a low-side near miss (`1.047183 MA` vs lower bound `1.0485 MA`), S1/S2 still needed accepted same-scope digitized waveform evidence, and the fixed-time crowbar was not Akel shot-scope sourced.
- Next checks before source-scope cleanup: inspect the strict M2 low-side near miss without arbitrary tuning, decide whether `pf1000_akel` should keep the unsourced fixed-time crowbar or move it behind an engineering preset/override, and continue same-scope Akel current trace digitization for S1/S2 waveform and dip acceptance.
- Final verification snapshot after 12 us/crowbar telemetry update: `git diff --check` clean; trailing-whitespace scan clean for touched notes/docs/probe files; targeted preset/PF-1000 gate slice passed (`5 passed, 14 xfailed in 1.72s`).

Ratchet update 2026-05-06, Akel source-scoped crowbar cleanup:

- Source audit result: the typed Akel shot-12581 target records circuit, geometry, Lee factors, waveform availability, and phase/dip context, but no crowbar enablement, crowbar time, crowbar resistance, or crowbar inductance. Local search in `KnowledgeReference/radiation-physics-and-chemistry-188-2021-109633.md` found no shot-scope support for the inherited `10.5 us` fixed-time crowbar.
- Fix implemented: removed the unsourced inherited fixed-time crowbar from the source-scoped `pf1000_akel` preset. The preset now keeps `crowbar_enabled=False` and no longer carries `crowbar_time`, `crowbar_resistance`, or `crowbar_inductance`. `tests/test_pf1000_akel_preset.py` ratchets this source boundary.
- Probe telemetry retained: `scripts/run_mlx_pf1000_probe.py` and `tests/test_mlx_pf1000_probe.py` still print `crowbar` and `crowbar_t_us`, so engineering crowbar overrides will remain visible in future logs.
- Verification after cleanup: `python3 -m py_compile src/dpf/presets.py tests/test_pf1000_akel_preset.py scripts/run_mlx_pf1000_probe.py tests/test_mlx_pf1000_probe.py` passed; focused Akel/PF-1000 gate slice passed (`5 passed, 14 xfailed in 1.83s`); 5-step standalone Akel smoke passed with `crowbar=0 crowbar_t_us=-1.000000`.
- Source-scoped no-crowbar 12 us probe: standalone `pf1000_akel` with `DPF_MLX_PROBE_TARGET_US=12`, `DPF_MLX_PROBE_STEPS=220000`, and `DPF_MLX_PROBE_PRINT_INTERVAL=20000` exited `PASSED`, reaching `t=12.000000 us` in `161659` steps. Final reported `peak_I=1.047183 MA` at `t=4.990339 us`.
- No-crowbar checkpoint evidence: step 120000 `t=9.000231 us`, `phase=pinch`, `I=0.643373 MA`, `V=10.293450 kV`, `r=6.237254 cm`, `Lp=43.947850 nH`; step 140000 `t=10.458063 us`, `I=0.641996 MA`, `V=9.600860 kV`, `r=11.155373 cm`; step 160000 `t=11.879750 us`, `I=0.704460 MA`, `V=8.881655 kV`, `r=15.200000 cm`; final step 161659 `I=0.707858 MA`, `V=8.817907 kV`, `crowbar=0`, `r=15.200000 cm`, `Lp=31.976097 nH`. Memory telemetry stayed flat (`mlx_active_MB=0.288`, cache about `10.529 MB`, peak about `9.793 MB`).
- Updated remaining limit: standalone `6 us`, `8 us`, and conservative `12 us` source-scoped endurance are now closed without a crowbar. Scientific acceptance remains blocked because strict M2 is still a low-side near miss (`1.047183 MA` vs lower bound `1.0485 MA`) and S1/S2 still require accepted same-scope digitized Akel waveform evidence and uncertainty.
- Next checks: troubleshoot the M2 low-side near miss by comparing full-engine `Lp/dLdt/phase/current` against the reduced Lee path that peaks inside band; continue Akel current trace digitization to turn S1/S2 into source-backed gates.

Ratchet update 2026-05-07, CPU snowplow Lee current-factor circuit loading:

- Modules/tests touched: `src/dpf/fluid/snowplow.py`, `tests/test_snowplow_consolidated.py`, `CodexFindings.md`, and `CortexFindings.md`.
- Source-of-truth basis: the Lee course describes axial phase `fm`/`fc` as coupled to the circuit equation and defines `fc` as the fraction of current effectively flowing in/driving the axial moving structure. It also says radial phase `fmr`/`fcr` are incorporated in all three radial phases, and gives an explicit axial dynamic-resistance example where `0.5*dL/dt` is reduced from about `5 mOhm` to `3.5 mOhm` when the current factor is considered. This supports current-factor scaling of circuit-facing `Lp`/`dLdt`, not just magnetic force.
- Issue isolated: the full `SimulationEngine` path used CPU `SnowplowModel`, whose magnetic force already used `(fc*I)^2`, but whose axial `plasma_inductance`, axial `dL_dt`, and frozen axial inductance used the unscaled coaxial `L_coeff`. The reduced `MLXSnowplow` path already scaled axial circuit inductance by `fc` and radial circuit inductance/back-EMF by `fcr`; it peaked at `1.150685 MA`, inside the Akel M2 band. The CPU/full path therefore over-loaded the circuit by about `1/fc` during axial current rise, explaining the prior low-side strict M2 miss.
- Fix implemented: `SnowplowModel` now keeps `L_coeff` as the unscaled coaxial geometry coefficient, but exposes circuit-facing helpers for axial `fc * L_coeff * z`, radial `fcr_eff * (mu0/2pi) * z_f * ln(b/r)`, and corresponding `dL/dt`. Axial rundown, radial compression, reflected shock, and post-pinch expansion now return current-factor-scaled `L_plasma`/`dL_dt` to the circuit. Tests were updated to assert this circuit-facing convention while preserving the unscaled geometry-coefficient tests.
- Verification status: `python3 -m py_compile src/dpf/fluid/snowplow.py tests/test_snowplow_consolidated.py` passed; focused snowplow formula slice passed (`35 passed in 1.30s`); full consolidated snowplow suite passed (`417 passed, 1 xfailed, 5 xpassed in 11.93s`); focused Akel/PF-1000 gate slice still passed (`5 passed, 14 xfailed in 1.40s`).
- Standalone 6 us M2 evidence after the circuit-loading fix: `pf1000_akel` no-crowbar probe with `DPF_MLX_PROBE_TARGET_US=6` exited `PASSED`, reaching `t=6.000050 us` in `75181` steps with `peak_I_MA=1.150507` at `peak_t_us=5.250198`. This is inside the Akel shot-12581 M2 band `1.0485-1.2815 MA`. Checkpoints: step 40000 `t=3.597952 us`, `I=1.103840 MA`, `Lp=10.142424 nH`; step 50000 `t=4.301412 us`, `I=1.137455 MA`, `Lp=13.357013 nH`; step 60000 `t=5.023197 us`, `I=1.149869 MA`, `Lp=16.792884 nH`; final step 75181 `I=1.144742 MA`, `V=11.965346 kV`, `crowbar=0`, `z=47.273209 cm`, `Lp=21.569092 nH`.
- Standalone 8 us radial/pinch evidence after the circuit-loading fix: `pf1000_akel` no-crowbar probe with `DPF_MLX_PROBE_TARGET_US=8` exited `PASSED`, reaching `t=8.000071 us` in `105978` steps with the same peak `1.150507 MA` at `5.250198 us`. Radial/pinch checkpoints stayed finite: step 80000 `t=6.295662 us`, `phase=radial`, `I=1.136085 MA`, `r=14.036811 cm`, `Lp=23.220228 nH`, `dLdt=6.427329 nH/us`; step 90000 `t=6.949331 us`, `phase=radial`, `I=1.053240 MA`, `r=7.214705 cm`, `Lp=29.929093 nH`, `dLdt=17.453964 nH/us`; step 100000 `t=7.599487 us`, `phase=pinch`, `I=0.825187 MA`, `r=3.078976 cm`, `Lp=38.512458 nH`, `dLdt=-12.436875 nH/us`; final step 105978 `phase=pinch`, `I=0.767598 MA`, `V=10.479495 kV`, `r=4.600753 cm`, `Lp=34.464097 nH`, `dLdt=-8.323167 nH/us`.
- Updated status: standalone source-scoped no-crowbar M2 is now closed for the `6 us` and `8 us` probes. The prior no-crowbar `12 us` endurance evidence was produced before the radial/reflected `fcr_eff` circuit-loading correction, so rerun the conservative `12 us` probe before claiming post-8us endurance is current. S1/S2 remain blocked until accepted same-scope digitized Akel waveform/dip evidence and uncertainty exist.
- Next checks: rerun the source-scoped no-crowbar `12 us` probe with the current circuit-loading fix; update `docs/METAL_V2_DOD.md`, `docs/METAL_V2_SPEC.md`, and `docs/SPRINT4_VALIDATION_REVIEW.md` so they no longer say M2 is a low-side near miss after the rerun evidence is complete; then continue Akel current-trace digitization for S1/S2.

Ratchet update 2026-05-07, current-factor-corrected 12 us Akel probe and doc cleanup:

- Modules/docs/tests touched: `docs/METAL_V2_DOD.md`, `docs/METAL_V2_SPEC.md`, `docs/SPRINT4_VALIDATION_REVIEW.md`, `tests/test_mlx_pf1000.py`, `CodexFindings.md`, and `CortexFindings.md`.
- Rerun completed: standalone `pf1000_akel` no-crowbar probe with `DPF_MLX_PROBE_TARGET_US=12`, `DPF_MLX_PROBE_STEPS=220000`, `DPF_MLX_PROBE_PRINT_INTERVAL=20000`, and `DPF_MLX_PROBE_MEMORY=1` exited `PASSED`.
- Current-factor-corrected 12 us evidence: the run reached `t=12.000000 us` in `159912` steps. Final reported `peak_I_MA=1.150507` at `peak_t_us=5.250198`, inside the Akel shot-12581 M2 band `1.0485-1.2815 MA`.
- Checkpoints stayed finite and no-crowbar: step 20000 `t=2.012285 us`, `I=0.880408 MA`, `phase=rundown`, `Lp=3.827323 nH`; step 40000 `t=3.597952 us`, `I=1.103840 MA`, `Lp=10.142424 nH`; step 60000 `t=5.023197 us`, `I=1.149869 MA`, `Lp=16.792884 nH`; step 80000 `t=6.295662 us`, `phase=radial`, `I=1.136085 MA`, `r=14.036811 cm`; step 100000 `t=7.599487 us`, `phase=pinch`, `I=0.825187 MA`, `r=3.078976 cm`; step 120000 `t=9.147131 us`, `I=0.709868 MA`, `r=8.958310 cm`; step 140000 `t=10.597788 us`, `I=0.763376 MA`, `r=14.469201 cm`.
- Final printed state: `phase=pinch`, `I=0.811876 MA`, `V=8.228613 kV`, `crowbar=0`, `r=15.200000 cm`, `Lp=22.417737 nH`, `dLdt=0.000000 nH/us`, `mlx_active_MB=0.288`, `mlx_cache_MB=10.333`, and `mlx_peak_MB=9.794`.
- Status update: standalone source-scoped no-crowbar M2 and conservative M6 endurance are now current after the Lee current-factor circuit-loading fix. Full scientific waveform acceptance remains blocked because S1/S2 still need accepted same-scope digitized Akel current waveform and current-dip evidence with uncertainty.
- Cleanup implemented: `docs/METAL_V2_DOD.md`, `docs/METAL_V2_SPEC.md`, and `docs/SPRINT4_VALIDATION_REVIEW.md` no longer describe the current run as a low-side M2 near miss. `tests/test_mlx_pf1000.py` no longer says the long xfailed gate is blocked by M6/CFL duration stability; it now records the remaining source-closure blocker for S1/S2.

Ratchet update 2026-05-07, Akel Fig. 1 extraction status:

- Modules/docs/assets/tests touched: `src/dpf/validation/digitization.py`, `tests/test_digitization.py`, `docs/SCIENTIFIC_CLOSURE_SOURCE_QUEUE.md`, `KnowledgeReference/figures/akel-2021-fig1-current-waveform-shot-12581.png`, `CodexFindings.md`, and `CortexFindings.md`.
- Progress made: promoted the local Akel 2021 Fig. 1 page-3 crop into `KnowledgeReference/figures/akel-2021-fig1-current-waveform-shot-12581.png`.
- Figure provenance: the crop was made from the parity-verified local Akel PDF page-3 render at 300 dpi. Its SHA-256 is `4c574525f1de413e54cd02bd06aa35d549db700270281310a3809edc54ab255e`.
- OCR/axis check: the extracted panel preserves the `0-10 us` x-axis, `0-1400 kA` y-axis, and legend entries for measured `PF1000 D2 Meas. curr. kA 1.2 Torr shot 12581` and computed `PF1000 D2 comp. curr. kA 1.2 Torr`.
- Draft vector extraction route: the current `pdftocairo` page-3 SVG separates a measured-current candidate as filled black paths `1987-2280` (`294` compact path elements, approximately `0.02-9.98 us`) and a computed-current candidate as black stroke paths `1942-1975` (`34` path elements, approximately `0.01-10.01 us`). Filled black paths `2345-2411` are legend glyphs in the white legend box and must be excluded. This is extraction metadata only.
- Queue update: `scientific_closure_digitization_queue()` now reports `akel_2021_fig1_current_waveform_shot_12581` as `extracted_not_digitized`, with figure path/hash, candidate axis calibration points, and draft vector path-separation metadata. The other Akel figure tasks remain `not_extracted`.
- Verification status: `python3 -m py_compile src/dpf/validation/digitization.py tests/test_digitization.py` passed; `python3 -m pytest tests/test_digitization.py -q` passed (`11 passed`); `python3 -m pytest tests/test_digitization.py tests/test_quality_assessment.py::TestQualityAssessment::test_scientific_accuracy_gap_report_lists_remaining_work -q` passed (`12 passed`); `shasum -a 256 KnowledgeReference/figures/akel-2021-fig1-current-waveform-shot-12581.png` matched the queue hash; `git diff --check` passed; trailing-whitespace scan over touched text files found no matches.
- Scientific boundary: this does not close S1/S2. The extracted figure is a provenance artifact only; accepted waveform evidence still requires measured/computed series arrays, overlay residuals, and independent review through `digitization_verification_evidence()`.

Ratchet update 2026-05-07, Akel Fig. 1 draft arrays:

- Modules/docs/assets/tests touched: `src/dpf/validation/digitization.py`, `src/dpf/validation/__init__.py`, `src/dpf/validation/quality_assessment.py`, `tests/test_digitization.py`, `tests/test_quality_assessment.py`, `docs/SCIENTIFIC_CLOSURE_SOURCE_QUEUE.md`, `KnowledgeReference/digitization/akel-2021-fig1-current-waveform-shot-12581-draft-packet.json`, `CodexFindings.md`, and `CortexFindings.md`.
- Draft packet created: `KnowledgeReference/digitization/akel-2021-fig1-current-waveform-shot-12581-draft-packet.json`, SHA-256 `0b8fae6147480392fcbe77eabeebc915a6a9561ec994daec32dea22859878017`.
- Candidate arrays: measured current has `294` points from filled black paths `1987-2280`; computed current has `34` points from black stroke paths `1942-1975`; legend glyphs `2345-2411` remain excluded (`67` filled path elements).
- Gate result: `digitization_verification_evidence(akel_fig1_draft_digitization_packet())` fails on exactly `independent_review_missing`, `overlay_residual_too_large`, and `review_status_not_accepted`. It no longer fails on missing source, source hash, figure image, axis calibration, or required series.
- Status report behavior: `scientific_closure_digitization_status([akel_fig1_draft_digitization_packet()])` reports `failed_task_count=1`, `open_task_count=5`, and `accepted_task_count=0`. Fig. 1 now appears as `failed` draft evidence needing review/correction instead of `digitization_packet_missing`.
- Verification status: `python3 -m py_compile src/dpf/validation/digitization.py src/dpf/validation/quality_assessment.py src/dpf/validation/__init__.py tests/test_digitization.py tests/test_quality_assessment.py` passed; focused digitization/gap-report pytest slice passed (`16 passed`); full touched-file pytest slice passed (`68 passed`); `git diff --check` passed; trailing-whitespace scan over touched files found no matches; draft packet `shasum -a 256` matched `0b8fae6147480392fcbe77eabeebc915a6a9561ec994daec32dea22859878017`.
- Scientific boundary: this still does not close S1/S2. The packet is `draft_unreviewed`; accepted waveform evidence still requires measured overlay residuals, at least one independent review, and `review_status="accepted"`.

Ratchet update 2026-05-07, Akel Fig. 1 internal overlay residual:

- Modules/docs/assets/tests touched: `src/dpf/validation/digitization.py`, `tests/test_digitization.py`, `docs/SCIENTIFIC_CLOSURE_SOURCE_QUEUE.md`, `KnowledgeReference/digitization/akel-2021-fig1-current-waveform-shot-12581-draft-packet.json`, `KnowledgeReference/digitization/akel-2021-page3.svg`, `CodexFindings.md`, and `CortexFindings.md`.
- Source SVG archived: `KnowledgeReference/digitization/akel-2021-page3.svg`, SHA-256 `b045c3b7033e50bd355e025ecf7c40d96edc1ffc7fcb6ef26832fe065fe99d3f`.
- Draft packet hash updated after adding overlay metadata: `KnowledgeReference/digitization/akel-2021-fig1-current-waveform-shot-12581-draft-packet.json`, SHA-256 `abe4a283ee154f84f6061da8ea508d3871faf3b14dddb2d1cfc8a7a0a5f8e0e7`.
- Overlay residual method: reprojected the draft data arrays through the Fig. 1 axis calibration and compared them with transformed `pdftocairo` SVG path bounding-box centers from the archived page-3 SVG. This is an internal vector round-trip residual, not an independent review.
- Overlay residual result: combined `328` candidate points had RMS residual `0.213455189 px` and max residual `2.733560259 px`; computed-current RMS was `0.000027947 px` over `34` points, and measured-current RMS was `0.225460245 px` over `294` points.
- Gate result after overlay update: `digitization_verification_evidence(akel_fig1_draft_digitization_packet())` now fails only on `independent_review_missing` and `review_status_not_accepted`. `overlay_residual_too_large` is no longer present.
- Verification status: `python3 -m py_compile src/dpf/validation/digitization.py src/dpf/validation/quality_assessment.py src/dpf/validation/__init__.py tests/test_digitization.py tests/test_quality_assessment.py` passed; `python3 -m pytest tests/test_digitization.py tests/test_quality_assessment.py -q` passed (`68 passed`); `git diff --check` passed; trailing-whitespace scan over touched files found no matches; packet and SVG SHA-256 checks matched the values above.
- Scientific boundary: S1/S2 remain blocked. The residual is measured and below the verifier RMS threshold, but the packet remains `draft_unreviewed` until independent review accepts it.

Ratchet update 2026-05-07, Akel waveform digitization readiness status:

- Modules/docs/tests touched: `src/dpf/validation/kr_targets.py`, `src/dpf/validation/__init__.py`, `tests/test_kr_targets.py`, `docs/SCIENTIFIC_CLOSURE_SOURCE_QUEUE.md`, `CodexFindings.md`, and `CortexFindings.md`.
- Helper added: `pf1000_16kv_current_waveform_digitization_candidate_evidence()` reports the Akel Fig. 1 waveform digitization state without comparing a simulation waveform to the trace.
- Current helper result: with the local draft packet, it returns `passed=False`, `waveform_digitization_status="blocked_by_review"`, required series present, overlay RMS `0.213455189 px`, and missing checks `["independent_review_missing", "review_status_not_accepted"]`.
- Verification status: `python3 -m py_compile src/dpf/validation/digitization.py src/dpf/validation/quality_assessment.py src/dpf/validation/kr_targets.py src/dpf/validation/__init__.py tests/test_digitization.py tests/test_quality_assessment.py tests/test_kr_targets.py` passed; `python3 -m pytest tests/test_digitization.py tests/test_quality_assessment.py tests/test_kr_targets.py -q` passed (`146 passed`); `git diff --check` passed; trailing-whitespace scan over touched files found no matches.
- Boundary: this moves downstream status from missing waveform data to draft blocked by review, but it intentionally does not close S1/S2 or tier-1 waveform validation.
