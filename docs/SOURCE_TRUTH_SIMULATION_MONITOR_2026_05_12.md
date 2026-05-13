# Source-Truth Simulation Monitor

- Generated: `2026-05-12T20:01:13+00:00`
- Scientific authority: local `KnowledgeReference/` and registry source metadata only.
- Boundary: this monitor does not accept draft digitizations or issue certificates.

## Summary

- `device_count`: `9`
- `validation_ready_device_count`: `1`
- `preset_count`: `16`
- `broken_preset_count`: `0`
- `accuracy_review_preset_count`: `0`
- `source_gap_review_preset_count`: `1`
- `model_coverage_review_preset_count`: `1`
- `warning_preset_count`: `0`
- `source_config_review_preset_count`: `1`
- `accuracy_review_device_count`: `0`
- `source_gap_review_device_count`: `7`
- `model_coverage_review_device_count`: `1`
- `pytest_lane_count`: `3`
- `pytest_failed_lane_count`: `0`

## Circuit/Waveform Devices

| Device | Source State | Workflow | Ipeak Err | Timing Err | NRMSE | Accuracy | Source Gap | Model Coverage |
| --- | --- | --- | ---: | ---: | ---: | --- | --- | --- |
| FAETON-I | nonaccepting: kr_status=unverified, waveform_provenance=reconstructed, waveform_kr_status=unverified | source_gap_review_needed | 1.833% | 3.875% | 0.260 | - | waveform_reconstructed_not_digitized, waveform_kr_status=unverified | - |
| MJOLNIR | nonaccepting: kr_status=unverified, waveform_provenance=reconstructed, waveform_kr_status=unverified | model_coverage_review_needed | 27.463% | 11.990% | 0.147 | peak_current_error>0.160_two_sigma | waveform_reconstructed_not_digitized, waveform_kr_status=unverified | mjolnir_restrike_current_trace_model_required_by_kr_but_no_accepted_timing_or_magnitude_parameters |
| NX2 | nonaccepting: kr_status=unverified, reliability=reference_only, waveform_missing, waveform_provenance=unset, waveform_kr_status=unverified | source_gap_review_needed | 23.542% | 46.044% | - | peak_current_error>0.160_two_sigma, timing_error>0.240_two_sigma | reference_only_device_not_scientific_validation_target, measured_current_waveform_missing, waveform_kr_status=unverified, nx2_course_example_not_same_shot_deuterium_target_missing=deuterium_device_match,measured_current_trace_source,experimental_phase_timing_uncertainty | - |
| PF-1000 | ready | within_current_pipeline | 2.326% | 9.328% | 0.168 | - | - | - |
| PF-1000-16kV | nonaccepting: kr_status=unverified, waveform_provenance=reconstructed, waveform_kr_status=unverified | source_gap_review_needed | 1.613% | 12.667% | 0.167 | - | waveform_reconstructed_not_digitized, waveform_kr_status=unverified | - |
| PF-1000-20kV | nonaccepting: kr_status=unverified, reliability=estimated, waveform_missing, waveform_provenance=unset, waveform_kr_status=unverified | source_gap_review_needed | 8.701% | 2.333% | - | - | measured_current_waveform_missing, waveform_kr_status=unverified | - |
| PF-1000-Gribkov | nonaccepting: kr_status=unverified, waveform_kr_status=unverified | source_gap_review_needed | 7.019% | 1.925% | 0.157 | - | waveform_kr_status=unverified | - |
| POSEIDON-60kV | nonaccepting: waveform_kr_status=unverified | source_gap_review_needed | 1.099% | 0.444% | 0.071 | - | waveform_kr_status=unverified | - |
| UNU-ICTP | nonaccepting: waveform_kr_status=unverified | source_gap_review_needed | 0.500% | 12.618% | 0.107 | - | waveform_kr_status=unverified | - |

## Preset Runs

| Preset | Reference | Source Scope Status | Workflow | Ipeak MA | Ipeak Err | tpeak us | Nonfinite | Warnings | Source Gap | Model Coverage | Source Config |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |
| tutorial | Tutorial Device (UNU-ICTP based) | not_validation_evidence | completed | 0.185 | - | 2.504 | 0 | 0 | - | - | - |
| pf1000 | PF-1000 | same_scope_source_reviewed_not_certificate | completed | 1.826 | 2.337% | 6.346 | 0 | 0 | - | - | - |
| pf1000_akel | PF-1000-16kV | same_scope_blocked_by_review | completed | 1.150 | 1.249% | 5.251 | 0 | 0 | waveform_reconstructed_not_digitized, waveform_kr_status=unverified | - | - |
| pf1000_20kv | PF-1000-20kV | derived_operating_point_not_validation_evidence | completed | 1.278 | 8.721% | 6.150 | 0 | 0 | measured_current_waveform_missing, waveform_kr_status=unverified | - | - |
| nx2 | NX2 | reference_only_not_validation_evidence | source_gap_review_needed | 0.306 | 23.545% | 0.971 | 0 | 0 | reference_only_device_not_scientific_validation_target, measured_current_waveform_missing, waveform_kr_status=unverified, nx2_course_example_not_same_shot_deuterium_target_missing=deuterium_device_match,measured_current_trace_source,experimental_phase_timing_uncertainty | - | - |
| unu_ictp | UNU-ICTP | same_scope_source_reviewed_waveform_unverified_not_certificate | completed | 0.181 | 0.502% | 2.479 | 0 | 0 | waveform_kr_status=unverified | - | - |
| llnl_dpf | LLNL-DPF | not_validation_evidence | completed | 0.282 | - | 1.796 | 0 | 0 | - | - | - |
| mjolnir | MJOLNIR | same_scope_partial_source_review_waveform_reconstructed_not_certificate | model_coverage_review_needed | 3.186 | 27.439% | 5.131 | 0 | 0 | waveform_reconstructed_not_digitized, waveform_kr_status=unverified | mjolnir_restrike_current_trace_model_required_by_kr_but_no_accepted_timing_or_magnitude_parameters | - |
| faeton | FAETON-I | same_scope_partial_source_review_waveform_reconstructed_not_certificate | source_config_review_needed | 0.982 | 1.833% | 3.556 | 0 | 0 | waveform_reconstructed_not_digitized, waveform_kr_status=unverified | - | snowplow.radial_transition_time_not_in_faeton_kr_extract_observed=7e-06 |
| poseidon | POSEIDON | not_validation_evidence | completed | 2.879 | - | 5.002 | 0 | 0 | - | - | - |
| poseidon_60kv | POSEIDON-60kV | same_scope_source_reviewed_waveform_unverified_not_certificate | completed | 3.155 | 1.102% | 1.990 | 0 | 0 | waveform_kr_status=unverified | - | - |
| aecs_pf2 | AECS-PF2 | not_validation_evidence | completed | 0.148 | - | 2.160 | 0 | 0 | - | - | - |
| pf400j | PF-400J | not_validation_evidence | completed | 0.126 | - | 0.295 | 0 | 0 | - | - | - |
| custom | Custom Device | not_validation_evidence | completed | 0.241 | - | 1.587 | 0 | 0 | - | - | - |
| cartesian_demo | Generic | not_validation_evidence | completed | 0.046 | - | 0.761 | 0 | 0 | - | - | - |
| phase_p_fidelity | Generic | not_validation_evidence | completed | 0.046 | - | 0.761 | 0 | 0 | - | - | - |

## Pytest Lanes

- `/Library/Frameworks/Python.framework/Versions/3.11/bin/python3 -m pytest tests/test_validation_ci.py -q -o addopts=`
  - status: `passed`, elapsed_s: `3.0`
  - output tail:

```text
...........................s                                             [100%]
27 passed, 1 skipped in 2.14s
```

- `/Library/Frameworks/Python.framework/Versions/3.11/bin/python3 -m pytest tests/test_mhd_acceptance.py -q -rsx -o addopts=`
  - status: `passed`, elapsed_s: `1.0`
  - output tail:

```text
sssss                                                                    [100%]
=========================== short test summary info ============================
SKIPPED [1] tests/test_mhd_acceptance.py:52: MLX not available
SKIPPED [1] tests/test_mhd_acceptance.py:70: MLX not available
SKIPPED [1] tests/test_mhd_acceptance.py:81: MLX not available
SKIPPED [1] tests/test_mhd_acceptance.py:106: MLX not available
SKIPPED [1] tests/test_mhd_acceptance.py:147: MLX not available
5 skipped in 0.33s
```

- `/Library/Frameworks/Python.framework/Versions/3.11/bin/python3 -m pytest tests/test_quality_assessment.py::TestQualityAssessment::test_scientific_accuracy_gap_report_lists_remaining_work tests/test_mhd_physics_integration.py::test_predictive_readiness_exported_and_blocks_unvalidated_claims tests/test_akel_digitization_source_integrity.py tests/test_preset_source_scope.py tests/test_unreviewed_physics_metadata.py -q -o addopts=`
  - status: `passed`, elapsed_s: `3.6`
  - output tail:

```text
.........................                                                [100%]
=============================== warnings summary ===============================
<frozen importlib._bootstrap>:241
<frozen importlib._bootstrap>:241
  <frozen importlib._bootstrap>:241: DeprecationWarning: builtin type SwigPyPacked has no __module__ attribute

<frozen importlib._bootstrap>:241
<frozen importlib._bootstrap>:241
  <frozen importlib._bootstrap>:241: DeprecationWarning: builtin type SwigPyObject has no __module__ attribute

<frozen importlib._bootstrap>:241
  <frozen importlib._bootstrap>:241: DeprecationWarning: builtin type swigvarlink has no __module__ attribute

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
25 passed, 5 warnings in 2.54s
sys:1: DeprecationWarning: builtin type swigvarlink has no __module__ attribute
```

## Findings

- All preset app-engine simulations completed without nonfinite arrays.
- No validation-ready preset crossed the current monitor accuracy flags.
- Preset source-gap review needed for: nx2 vs NX2 [reference_only_device_not_scientific_validation_target, measured_current_waveform_missing, waveform_kr_status=unverified, nx2_course_example_not_same_shot_deuterium_target_missing=deuterium_device_match,measured_current_trace_source,experimental_phase_timing_uncertainty]
- Preset model-coverage review needed for: mjolnir vs MJOLNIR [mjolnir_restrike_current_trace_model_required_by_kr_but_no_accepted_timing_or_magnitude_parameters]
- Preset source-config review needed for: faeton vs FAETON-I [snowplow.radial_transition_time_not_in_faeton_kr_extract_observed=7e-06]
- No validation-ready waveform device crossed the current monitor accuracy flags.
- Waveform/device source-gap review needed for: FAETON-I [waveform_reconstructed_not_digitized, waveform_kr_status=unverified]; NX2 [reference_only_device_not_scientific_validation_target, measured_current_waveform_missing, waveform_kr_status=unverified, nx2_course_example_not_same_shot_deuterium_target_missing=deuterium_device_match,measured_current_trace_source,experimental_phase_timing_uncertainty]; PF-1000-16kV [waveform_reconstructed_not_digitized, waveform_kr_status=unverified]; PF-1000-20kV [measured_current_waveform_missing, waveform_kr_status=unverified]; PF-1000-Gribkov [waveform_kr_status=unverified]; POSEIDON-60kV [waveform_kr_status=unverified]; UNU-ICTP [waveform_kr_status=unverified]
- Waveform/device model-coverage review needed for: MJOLNIR [mjolnir_restrike_current_trace_model_required_by_kr_but_no_accepted_timing_or_magnitude_parameters]
- Nonaccepting waveform/device evidence still simulated but not scored as accepted: FAETON-I, MJOLNIR, NX2, PF-1000-16kV, PF-1000-20kV, PF-1000-Gribkov, POSEIDON-60kV, UNU-ICTP
- All requested pytest monitor lanes returned success.

