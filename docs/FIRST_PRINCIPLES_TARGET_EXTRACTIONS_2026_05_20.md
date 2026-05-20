# First-Principles Target Extractions - 2026-05-20

Scope: convert already-local `KnowledgeReference/` material from
`source_available_not_target_extracted` into typed target records without
promoting validation or whole-shot first-principles acceptance.

Implemented packet:

- `src/dpf/first_principles/source_targets.py::sprint4_source_available_target_extractions`
- `src/dpf/validation/kr_targets.py::pf1000_krasa_vessel_scatter_anisotropy_targets`
- `src/dpf/fields/source_geometry.py::PF1000GeometryPacket`

## Extraction Results

| Target | New status | Extracted support | Runtime impact |
| --- | --- | --- | --- |
| PF-1000 Krasa 2008 vessel geometry and scatter | `target_extracted_source_supported_geometry_context_wrong_scope_for_akel_validation` | `KnowledgeReference/anisotropy-of-the-emission-of-dd-fusion-neutrons-caused-by-the-plasma-focus-vessel-527cc533.md:113-118,121-130,132-140,269-301` | Chamber wall material/thickness are source-supported geometry fields; Krasa is also a detector/scatter requirement target. It remains full-energy PF-1000, not Akel 16 kV validation. |
| PF-1000 Stepniewski 2004 hollow-anode bore | `target_extracted_modeling_context_requires_review` | `KnowledgeReference/doi-10-1016-j-vacuum-2004-05-019-f931cb0b.md:306-314` | Hollow radius `0.015 m` is extracted, but the runtime geometry field stays blocked because the value is simulation-parameter context until hardware-scope review. |
| UCSD/Beg current-sheath initiation | `target_extracted_wrong_scope_startup_method_context` | `KnowledgeReference/effect-of-current-sheath-initiation-on-the-radial-collapse-and-energetic-particle-accelera-b2e95b88.md:160-205,458-500,616-670` | Startup terms, current start jitter, pressure-regime notes, and insulator/liftoff scaling are available as method context only; no PF-1000 startup BVP acceptance. |
| Neon gas-puff Hall/LHDI anomalous resistivity | `target_extracted_non_dpf_zpinch_closure_candidate` | `KnowledgeReference/the-hall-term-and-anomalous-resistivity-effects-in-neon-gas-puff-z-pinches.md:185-266` | Generalized Ohm/Hall/anomalous-resistivity formula is extracted for candidate closure review. It is not DPF same-scope transport authority. |
| NRL 2019 transport formulary core | `target_extracted_formulary_crosscheck_not_dpf_authority` | `KnowledgeReference/2019nrlplasma-formulary-037290d4.md:2698-2710,2996-3038,3186-3308,3371-3428` | Spitzer, anomalous ion-sound, two-temperature equilibration, multispecies transport, weakly ionized conductivity, and validity limits are extracted for unit/regime cross-checks. It is not DPF same-scope closure authority. |
| Talebitaher 2012 NX2 detector/anisotropy | `target_extracted_nx2_detector_anisotropy_context` | `KnowledgeReference/chunks/coded-aperture-imaging-of-nuclear-fusion-in-the-plasma-focus-device-9b79429f/pages-0101-0125.md:167-198,1188-1268`; `pages-0151-0175.md:170-245`; `pages-0201-0225.md:390-486` | Be detector, CAI geometry, deuteron cone model, and material-scatter anisotropy are extracted for detector/beam-target method design. It remains NX2, not PF-1000 neutron authority. |
| Klir 2011 ToF detector response | `already_target_extracted_in_kr_targets` | `KnowledgeReference/fusion-neutron-detector-for-time-of-flight-measurements-in-z-pinch-and-plasma-focus-214fbdae.md:78-207` | Existing KR target extraction is now represented in the Sprint 4 extraction packet; missing digitized response curves stay blocked. |

## Geometry State Change

Krasa 2008 lines 113-115 now support:

- `chamber_wall_material`: `1`, units `stainless_steel_material_flag`
- `chamber_wall_thickness_m`: `0.010 m`

The chamber-wall mask still reports `candidate_projection_not_source_mask` for
the default Krauz packet because the cathode-cage radial split remains a
conflict. A focused test proves it would promote only after that radial split is
source-supported.

Stepniewski 2004 lines 310-314 now support the extracted candidate:

- `hollow_radius_centre_of_electrode_m`: `0.015 m`

That value is intentionally not promoted into `PF1000GeometryPacket` because it
is a simulation-parameter extraction, not reviewed hardware-scope geometry.

## Still Blocked

- PF-1000 insulator wall thickness and outer-radius hardware source.
- PF-1000 backplate radial extent and axial thickness.
- D2 Townsend/Paschen data and a source-reviewed DPF restrike equation.
- Direct Braginskii 1965 transport-coefficient table extraction for any claim
  that requires those exact coefficients.
- Same-scope Akel 16 kV `V(t)`, `Te/Ti`, X-ray, neutron spectrum, and anisotropy.
- Any whole-shot first-principles validation certificate.

## Verification

- `.venv312/bin/python -m ruff check src/dpf/first_principles/source_targets.py src/dpf/fields/source_geometry.py src/dpf/validation/kr_targets.py src/dpf/first_principles/__init__.py src/dpf/validation/__init__.py tests/test_first_principles_source_targets.py tests/test_source_geometry_packet.py tests/test_kr_targets.py`
- `.venv312/bin/python -m pytest tests/test_first_principles_source_targets.py tests/test_source_geometry_packet.py tests/test_kr_targets.py -q -rx`

Result: `165 passed`; ruff clean.
