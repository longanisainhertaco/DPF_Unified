# FP-15 Generalized DPF-Machine Source Search

Date: 2026-05-15  
Scope: local `KnowledgeReference/` source truth and first-principles planning artifacts only.

## Verdict

FP-15 is blocked. A general first-principles DPF tool cannot be claimed after a
single PF-1000/Akel engineering path. The source truth supports candidate second
scopes, but each one must repeat `FP-1` through `FP-14` with typed evidence,
review, UQ, and a separate certificate. No second-scope packet is accepted.

The package-native runner now emits a fail-closed `generalization` packet with
status `blocked_generalized_dpf_machine_path_not_available`. It can list
candidate second scopes, but it cannot claim generalized DPF-machine authority.

## Source Findings

### Plan Gate

- `docs/FIRST_PRINCIPLES_FINISH_LINE_PLAN.md:88-89` defines the general tool
  threshold as a second device or shot repeating the full evidence path without
  hidden PF-1000/Akel assumptions.
- `docs/FIRST_PRINCIPLES_FINISH_LINE_PLAN.md:133-135` keeps broader
  DPF-machine completion downstream of the first accepted demonstrator.
- `docs/FIRST_PRINCIPLES_FINISH_LINE_PLAN.md:172` defines FP-15 directly.
- `docs/FIRST_PRINCIPLES_FINISH_LINE_PLAN.md:713` maps FP-15 to `DPF-VV-018`.

### Current Missing Same-Scope Set

- `docs/FIRST_PRINCIPLES_SOURCE_TRUTH_INDEX.md:86-91` states that no complete
  same-scope 3D validation set is currently available for geometry, drive,
  density, fields, temperatures, neutron timing/spectrum/anisotropy, detector
  response, and UQ.
- `docs/FIRST_PRINCIPLES_SOURCE_TRUTH_INDEX.md:130-157` lists directly needed
  candidate sources, but this is source availability, not accepted validation.

### Candidate Second Scopes

1. `soto2010_cchen_pf400j_pf50j_speed_nanofocus_matrix`
   - Supported by Soto 2010 CCHEN device configurations, aggregate shot and
     diagnostic observations, and cross-device scaling matrix:
     `KnowledgeReference/studies-on-scalability-and-scaling-laws-for-the-plasma-focus-similarities-and-differences-5f680756.md:232-239,305-320,331-354,403-438,526-898,917-957,976-1006,1170-1202`.
   - Blocker: table targets are not accepted typed evidence yet, aggregate
     observations are not complete same-shot time histories, and each selected
     device still needs its own FP-1 through FP-14 packet chain and certificate.

2. `pf1000_full_energy_anisotropy_450_500kj_3p5torr`
   - Supported by PF-1000 full-energy anisotropy/TOF/direct-scattered context:
     `KnowledgeReference/anisotropy-of-the-emission-of-dd-fusion-neutrons-caused-by-the-plasma-focus-vessel-527cc533.md:121-137,175-204,269-284,432-438`.
   - Supported by PF-1000 interferometry density/pinch context:
     `KnowledgeReference/sixteenframe-interferometer-for-a-study-of-a-pinch-dynamics-in-pf1000-device-f8dc9d1b.md:130-174`.
   - Blocker: not Akel shot 12581; requires its own full packet chain and review.

3. `faeton_i_100kv_second_device_scope`
   - Supported by high-voltage current-sheath, voltage, neutron-yield,
     anisotropy, PMT-scintillator, and Faraday-cup context:
     `KnowledgeReference/faeton-i-investigation-of-plasma-dynamics-and-radiation-output-of-a-100-kv-plasma-focus-device.md:46-55,64-78`.
   - Blocker: current source leans on Lee-model comparison and must be
     re-extracted as first-principles evidence before use.

4. `llnl_180ka_kinetic_or_hybrid_reference`
   - Supported by LLNL-like fully kinetic geometry/current/beam/yield context:
     `KnowledgeReference/fully-kinetic-simulations-of-dense-plasma-focus-z-pinch.md:87-118,135-156`.
   - Supported by the newly ingested hybrid PIC-fluid source:
     `KnowledgeReference/fully-electromagnetic-hybrid-pic-fluid-dpf-neutron-yield-acb71fa9.md:58-63,685-711,952-990,1018-1040,1083-1089,1240-1263`.
   - Blocker: hybrid source is 2D axisymmetric and public same-scope
     experimental packet is incomplete.

5. `mjolnir_60kv_735kj_9torr_mechanism_scope`
   - Supported by MJOLNIR diagnostic layout, transmission-line circuit,
     MHD-to-kinetic modeling, neutron pulse/spectrum/anisotropy, and activation
     detector context:
     `KnowledgeReference/neutron-generation-dynamics-inside-a-ma-class-dense-plasma-focus-z-pinch-5.md:101-145,174-211,429-430,480-487,575-608,748-764`.
   - Blocker: strong mechanism material, but no accepted first-principles
     certificate until all FP gates are re-run for this scope.

6. `pf1000_akel_other_shot_or_pressure_series`
   - Supported by Akel/PF-1000 shot-series scalar context already captured in
     FP-9 through FP-14.
   - Blocker: a second shot on the same machine can test reproducibility, but it
     does not by itself prove cross-device generality.

## Required Generalization Channels

The runner must block until all channels are accepted:

- accepted primary-scope certificate;
- declared second scope;
- second-scope geometry, drive waveform, startup, power port, dimensionality,
  physics closure, density/field/temperature, neutron authority, detector/UQ,
  comparator/UQ, numerical fidelity, and certificate packets;
- proof of no hidden PF-1000/Akel assumptions;
- device parameterization schema;
- scale-transition or nondimensionalization review;
- regression against primary scope;
- source review certificate;
- cross-scope negative tests.

## Implementation Ratchet

- Added `src/dpf/first_principles/generalization.py`.
- Wired the packet into `src/dpf/first_principles/runner.py`, manifest candidate
  evidence, validation packet summaries, and deck manifest config.
- Exported `build_generalized_dpf_machine_packet` from
  `src/dpf/first_principles/__init__.py`.
- Added focused tests in `tests/test_first_principles_runner.py`.

The packet is deliberately non-promoting. It does not select a second
demonstrator; it preserves the candidate list and forces the next execution
step to assemble one complete second-scope evidence chain.

Additional ratchet added in this pass:

- Soto 2010 CCHEN PF-400J/PF-50J/SPEED/Nanofocus matrix is now listed as
  candidate second-scope requirement material, not acceptance evidence.
- `src/dpf/first_principles/generalization.py` now emits
  `generalization_channel_status`, `claim_policy`,
  `required_second_scope_gate_ids`, `candidate_second_scope_decisions`, and
  `upstream_acceptance_gate`.
- Candidate scopes are now explicitly marked
  `candidate_requirement_material_not_acceptance` and each requires an
  independent FP-1 through FP-14 packet chain plus its own certificate.
- `tests/test_first_principles_runner.py` asserts that a single PF-1000/Akel
  scope is not a generalized claim, the second scope must carry `FP-1` through
  `FP-14`, and blocked primary packets prevent starting a generalization claim.

## Validated Commands

- `python3 -m pytest tests/test_first_principles_runner.py` -> 11 passed after
  adding the Soto 2010 CCHEN second-scope candidate to the runtime packet.
- `python3 -c "import json; [json.load(open(path)) for path in ['docs/FIRST_PRINCIPLES_SOTO2010_SOURCE_TRIAGE_2026_05_15.json','docs/USER_PDF_KR_PROMOTION_2026_05_15.json','docs/FIRST_PRINCIPLES_GENERALIZATION_SOURCE_SEARCH_2026_05_15.json']]; print('json ok')"` -> valid JSON.
- `python3 -m pytest tests/test_first_principles_runner.py` -> 8 passed.
- `python3 -m json.tool docs/FIRST_PRINCIPLES_GENERALIZATION_SOURCE_SEARCH_2026_05_15.json` -> valid JSON.
- `python3 -m pytest tests/test_first_principles_input_deck.py tests/test_first_principles_runner.py tests/test_first_principles_manifest.py tests/test_cli_first_principles_3d.py tests/test_kinetic_yield_history.py tests/test_hybrid_3d_loop.py tests/test_hybrid_pic_3d_validation_packet.py` -> 38 passed.
