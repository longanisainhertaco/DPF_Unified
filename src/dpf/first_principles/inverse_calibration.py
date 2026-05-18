"""Experimental inverse calibration for source-backed DPF machine decks.

The calibration path runs source-bounded parameter candidates through the
package-native first-principles simulator and scores them against typed source
observables.  It is intentionally non-promoting: a best fit is not treated as a
unique physical conclusion unless the candidate set passes a basic
identifiability check.
"""

from __future__ import annotations

import itertools
import math
from typing import Any

import numpy as np

CALIBRATION_STATUS = "experimental_inverse_calibration_not_validation"
SOURCE_REFERENCES = (
    "docs/FIRST_PRINCIPLES_ENGINEERING_FIRM_DOSSIER_2026_05_16.md",
    "docs/FIRST_PRINCIPLES_NUMERICAL_FIDELITY_SOURCE_SEARCH_2026_05_15.md",
)
DEFAULT_MINIMUM_WAVEFORM_COVERAGE_FRACTION = 0.95


def build_source_bounded_candidate_grid(
    *,
    baseline_parameters: dict[str, float],
    parameter_names: tuple[str, ...],
    scale_values: tuple[float, ...],
) -> tuple[dict[str, Any], ...]:
    """Build multiplicative candidates around source/deck baseline values."""

    if not parameter_names:
        raise ValueError("at least one parameter is required")
    scales = tuple(_positive_float(value, "scale") for value in scale_values)
    return build_source_bounded_candidate_grid_from_parameter_scales(
        baseline_parameters=baseline_parameters,
        parameter_names=parameter_names,
        parameter_scale_values={name: scales for name in parameter_names},
    )


def build_source_bounded_candidate_grid_from_parameter_scales(
    *,
    baseline_parameters: dict[str, float],
    parameter_names: tuple[str, ...],
    parameter_scale_values: dict[str, tuple[float, ...]],
) -> tuple[dict[str, Any], ...]:
    """Build candidates with parameter-specific multiplicative scale lists."""

    if not parameter_names:
        raise ValueError("at least one parameter is required")
    scale_sequences: list[tuple[float, ...]] = []
    for name in parameter_names:
        if name not in parameter_scale_values:
            raise ValueError(f"missing scale list for parameter {name}")
        scales = tuple(
            _positive_float(value, f"{name} scale")
            for value in parameter_scale_values[name]
        )
        if not scales:
            raise ValueError(f"scale list for {name} must not be empty")
        scale_sequences.append(scales)

    candidates: list[dict[str, Any]] = []
    for index, scale_tuple in enumerate(
        itertools.product(*scale_sequences)
    ):
        values = dict(baseline_parameters)
        factors: dict[str, float] = {}
        for name, scale in zip(parameter_names, scale_tuple, strict=True):
            if name not in baseline_parameters:
                raise ValueError(f"missing baseline parameter {name}")
            baseline = _positive_float(baseline_parameters[name], name)
            values[name] = baseline * scale
            factors[name] = scale
        candidates.append(
            {
                "candidate_id": f"candidate_{index:04d}",
                "parameter_values": values,
                "parameter_factors": factors,
                "parameter_scale_values": {
                    name: list(parameter_scale_values[name])
                    for name in parameter_names
                },
                "source_bound_policy": "multiplicative_scales_around_source_or_deck_value",
            }
        )
    return tuple(candidates)


def score_current_history_against_targets(
    *,
    current_history: list[dict[str, Any]],
    target_observables: dict[str, Any],
) -> dict[str, Any]:
    """Score retained circuit-current history against source observables."""

    if not current_history:
        return {
            "status": "no_current_history",
            "score": math.inf,
            "metrics": {},
            "usable": False,
        }

    signed_currents = [float(row["current_A"]) for row in current_history]
    currents = [abs(value) for value in signed_currents]
    times = [
        float(row.get("time_s", float(row.get("time_us", 0.0)) * 1.0e-6))
        for row in current_history
    ]
    peak_index = max(range(len(currents)), key=lambda idx: currents[idx])
    simulated_peak_A = currents[peak_index]
    simulated_peak_time_s = times[peak_index]
    peak_at_final_sample = peak_index == len(currents) - 1
    metrics: dict[str, Any] = {
        "simulation_peak_current_A": simulated_peak_A,
        "simulation_peak_time_s": simulated_peak_time_s,
        "simulation_point_count": len(current_history),
        "peak_at_final_sample": peak_at_final_sample,
        "horizon_covers_peak_candidate": not peak_at_final_sample,
    }
    terms: list[float] = []

    target_peak = _optional_positive_float(target_observables.get("peak_current_A"))
    if target_peak is not None:
        error = _relative_error(simulated_peak_A, target_peak)
        metrics["target_peak_current_A"] = target_peak
        metrics["peak_current_relative_error"] = error
        terms.append(error * error)

    target_peak_time = _optional_positive_float(target_observables.get("peak_time_s"))
    if target_peak_time is not None:
        error = _relative_error(simulated_peak_time_s, target_peak_time)
        metrics["target_peak_time_s"] = target_peak_time
        metrics["peak_time_relative_error"] = error
        terms.append(error * error)

    waveform_metrics = _waveform_shape_metrics(
        currents_A=signed_currents,
        times_s=times,
        target_observables=target_observables,
    )
    metrics.update(waveform_metrics)
    waveform_nrmse = waveform_metrics.get("waveform_nrmse_fraction")
    if (
        waveform_metrics.get("waveform_score_included") is True
        and waveform_nrmse is not None
    ):
        terms.append(float(waveform_nrmse) * float(waveform_nrmse))

    if not terms:
        return {
            "status": "no_supported_target_observables",
            "score": math.inf,
            "metrics": metrics,
            "usable": False,
        }

    score = math.sqrt(sum(terms) / len(terms))
    return {
        "status": "scored_against_source_observables",
        "score": score,
        "metrics": metrics,
        "usable": math.isfinite(score),
    }


def classify_inverse_calibration_results(
    *,
    candidate_results: tuple[dict[str, Any], ...],
    parameter_names: tuple[str, ...],
    near_best_relative_margin: float = 0.10,
    narrow_range_fraction: float = 0.10,
    accepted_fit_score_threshold: float | None = None,
) -> dict[str, Any]:
    """Classify whether candidate scores identify unique parameters."""

    usable = [
        result
        for result in candidate_results
        if result.get("case_status") == "completed_engineering_candidate_run"
        and result.get("finite_state_all_finite") is not False
        and result.get("scoring", {}).get("usable") is True
        and math.isfinite(float(result["scoring"]["score"]))
    ]
    if not usable:
        return {
            "status": "no_conclusion_no_usable_completed_candidates",
            "best_candidate_id": None,
            "near_best_candidate_count": 0,
            "parameter_intervals": {},
            "can_conclude_unique_parameters": False,
        }

    ranked = sorted(usable, key=lambda result: float(result["scoring"]["score"]))
    best = ranked[0]
    best_score = float(best["scoring"]["score"])
    threshold = best_score * (1.0 + float(near_best_relative_margin))
    if best_score == 0.0:
        threshold = float(near_best_relative_margin)
    near_best = [
        result
        for result in ranked
        if float(result["scoring"]["score"]) <= threshold
    ]

    intervals: dict[str, Any] = {}
    all_narrow = True
    for name in parameter_names:
        values = [
            float(result["candidate"]["parameter_values"][name])
            for result in near_best
        ]
        baseline = float(ranked[0]["candidate"]["baseline_parameters"][name])
        value_min = min(values)
        value_max = max(values)
        span_fraction = (value_max - value_min) / max(abs(baseline), 1.0e-300)
        narrow = span_fraction <= narrow_range_fraction
        all_narrow = all_narrow and narrow
        intervals[name] = {
            "min": value_min,
            "max": value_max,
            "baseline": baseline,
            "span_fraction_of_baseline": span_fraction,
            "narrow_at_configured_threshold": narrow,
        }

    if len(near_best) == 1:
        if accepted_fit_score_threshold is None:
            status = "separated_candidate_grid_without_accepted_fit_tolerance"
        elif best_score <= accepted_fit_score_threshold:
            status = "uniquely_inferred_on_candidate_grid"
        else:
            status = "separated_candidate_grid_but_fit_score_exceeds_threshold"
    elif all_narrow:
        status = "range_constrained_on_candidate_grid"
    else:
        status = "underdetermined_or_correlated_on_candidate_grid"
    horizon_limited_reasons = {
        result["candidate"]["candidate_id"]: reasons
        for result in near_best
        if (reasons := _candidate_horizon_limited_reasons(result))
    }
    horizon_limited_candidates = list(horizon_limited_reasons)
    if horizon_limited_candidates:
        status = "horizon_limited_requires_longer_run"

    return {
        "status": status,
        "best_candidate_id": best["candidate"]["candidate_id"],
        "best_score": best_score,
        "near_best_score_threshold": threshold,
        "near_best_candidate_count": len(near_best),
        "near_best_candidate_ids": [
            result["candidate"]["candidate_id"] for result in near_best
        ],
        "horizon_limited_candidate_ids": horizon_limited_candidates,
        "horizon_limited_reasons": horizon_limited_reasons,
        "parameter_intervals": intervals,
        "accepted_fit_score_threshold": accepted_fit_score_threshold,
        "fit_score_within_accepted_threshold": (
            None
            if accepted_fit_score_threshold is None
            else best_score <= accepted_fit_score_threshold
        ),
        "best_candidate_grid_separated": len(near_best) == 1,
        "can_conclude_unique_parameters": status == "uniquely_inferred_on_candidate_grid",
        "acceptance_policy": "non_promoting_experimental_inverse_fit",
    }


def build_experimental_inverse_calibration_packet(
    *,
    declared_scope: str,
    device_name: str,
    target_observables: dict[str, Any],
    candidate_results: tuple[dict[str, Any], ...],
    parameter_names: tuple[str, ...],
) -> dict[str, Any]:
    """Return a non-promoting inverse-calibration packet."""

    completed = [
        result
        for result in candidate_results
        if result.get("case_status") == "completed_engineering_candidate_run"
    ]
    finite = [
        result
        for result in completed
        if result.get("finite_state_all_finite") is True
    ]
    ranked = sorted(
        (
            result
            for result in completed
            if result.get("finite_state_all_finite") is not False
            if result.get("scoring", {}).get("usable") is True
        ),
        key=lambda result: float(result["scoring"]["score"]),
    )
    identifiability = classify_inverse_calibration_results(
        candidate_results=candidate_results,
        parameter_names=parameter_names,
    )
    target_waveform_available = isinstance(target_observables.get("waveform"), dict)
    return {
        "task_id": "experimental_inverse_calibration",
        "status": CALIBRATION_STATUS,
        "declared_scope": declared_scope,
        "device_name": device_name,
        "source_policy": {
            "source_truth": "typed local source targets and user-verified workbook observables",
            "reduced_models_used": False,
            "measured_waveforms_used_as_drive": False,
            "measured_waveforms_used_as_inverse_score": target_waveform_available,
            "calibration_changes_are_candidate_parameters_only": True,
            "calibration_is_non_promoting_experimental_fit": True,
        },
        "target_observables": target_observables,
        "parameter_names": list(parameter_names),
        "candidate_count": len(candidate_results),
        "completed_candidate_count": len(completed),
        "finite_candidate_count": len(finite),
        "blocked_candidate_count": len(candidate_results) - len(completed),
        "best_candidates": [
            _candidate_result_summary(result) for result in ranked[:5]
        ],
        "identifiability": identifiability,
        "parameter_sensitivity": _parameter_sensitivity(
            candidate_results=tuple(candidate_results),
            parameter_names=parameter_names,
        ),
        "horizon_recommendation": _horizon_recommendation(
            target_observables=target_observables,
            candidate_results=tuple(candidate_results),
            identifiability=identifiability,
        ),
        "candidate_results": list(candidate_results),
        "can_support_first_principles_acceptance": False,
    }


def _candidate_result_summary(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "candidate_id": result["candidate"]["candidate_id"],
        "parameter_values": result["candidate"]["parameter_values"],
        "parameter_factors": result["candidate"]["parameter_factors"],
        "score": result["scoring"]["score"],
        "metrics": result["scoring"]["metrics"],
        "duration_request_satisfied": result.get("duration_request_satisfied"),
        "finite_state_all_finite": result.get("finite_state_all_finite"),
    }


def _parameter_sensitivity(
    *,
    candidate_results: tuple[dict[str, Any], ...],
    parameter_names: tuple[str, ...],
) -> dict[str, Any]:
    completed = [
        result
        for result in candidate_results
        if result.get("case_status") == "completed_engineering_candidate_run"
        and result.get("scoring", {}).get("usable") is True
    ]
    sensitivities: dict[str, Any] = {}
    for name in parameter_names:
        factor_values = sorted({
            float(result["candidate"]["parameter_factors"][name])
            for result in completed
            if name in result.get("candidate", {}).get("parameter_factors", {})
        })
        if len(factor_values) <= 1:
            sensitivities[name] = {
                "status": "not_varied_in_candidate_grid",
                "factor_values": factor_values,
                "can_infer_parameter_sensitivity": False,
            }
            continue
        groups = _conditional_parameter_groups(
            completed,
            parameter_name=name,
            parameter_names=parameter_names,
        )
        varied_groups = [
            group
            for group in groups
            if len({
                float(item["candidate"]["parameter_factors"][name])
                for item in group["candidate_results"]
            })
            > 1
        ]
        if not varied_groups:
            sensitivities[name] = {
                "status": "not_varied_with_other_parameters_held_fixed",
                "factor_values": factor_values,
                "can_infer_parameter_sensitivity": False,
            }
            continue
        group_packets = [
            _parameter_group_sensitivity_packet(name, group)
            for group in varied_groups
        ]
        scored_observed = any(
            packet["effect_observed_on_scored_metrics"]
            for packet in group_packets
        )
        runtime_observed = any(
            packet["effect_observed_on_runtime_metrics"]
            for packet in group_packets
        )
        runtime_available = any(
            packet["runtime_metric_available"]
            for packet in group_packets
        )
        sensitivities[name] = {
            "status": (
                "observed_effect_on_scored_metrics"
                if scored_observed
                else (
                    "observed_effect_on_candidate_runtime_metrics_not_scored_current"
                    if runtime_observed
                    else (
                        "no_observed_effect_on_scored_or_runtime_metrics"
                        if runtime_available
                        else "no_observed_effect_on_scored_metrics"
                    )
                )
            ),
            "factor_values": factor_values,
            "conditional_group_count": len(group_packets),
            "effect_observed_group_count": sum(
                1 for packet in group_packets if packet["effect_observed"]
            ),
            "scored_effect_observed_group_count": sum(
                1
                for packet in group_packets
                if packet["effect_observed_on_scored_metrics"]
            ),
            "runtime_effect_observed_group_count": sum(
                1
                for packet in group_packets
                if packet["effect_observed_on_runtime_metrics"]
            ),
            "groups": group_packets,
            "can_infer_parameter_sensitivity": scored_observed or runtime_observed,
        }
    return {
        "status": "experimental_candidate_grid_sensitivity_not_validation",
        "parameters": sensitivities,
        "can_support_first_principles_acceptance": False,
    }


def _conditional_parameter_groups(
    candidate_results: list[dict[str, Any]],
    *,
    parameter_name: str,
    parameter_names: tuple[str, ...],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[tuple[str, float], ...], list[dict[str, Any]]] = {}
    for result in candidate_results:
        factors = result.get("candidate", {}).get("parameter_factors", {})
        if parameter_name not in factors:
            continue
        key = tuple(
            (name, float(factors[name]))
            for name in parameter_names
            if name != parameter_name and name in factors
        )
        grouped.setdefault(key, []).append(result)
    return [
        {
            "held_fixed_factors": dict(key),
            "candidate_results": values,
        }
        for key, values in grouped.items()
    ]


def _parameter_group_sensitivity_packet(
    parameter_name: str,
    group: dict[str, Any],
) -> dict[str, Any]:
    results = group["candidate_results"]
    metric_ranges = {
        "score": _range_for_values([
            float(result["scoring"]["score"])
            for result in results
            if result.get("scoring", {}).get("score") is not None
        ]),
        "simulation_peak_current_A": _range_for_metric(
            results,
            "simulation_peak_current_A",
        ),
        "simulation_peak_time_s": _range_for_metric(
            results,
            "simulation_peak_time_s",
        ),
        "waveform_nrmse_fraction": _range_for_metric(
            results,
            "waveform_nrmse_fraction",
        ),
    }
    runtime_metric_ranges = {
        "j_dot_e_power_W_max_abs": _range_for_plasma_loading_metric(
            results,
            "j_dot_e_power_W_max_abs",
        ),
        "j_dot_e_energy_trapezoid_J": _range_for_plasma_loading_metric(
            results,
            "j_dot_e_energy_trapezoid_J",
        ),
        "field_energy_delta_J": _range_for_plasma_loading_metric(
            results,
            "field_energy_delta_J",
        ),
        "field_energy_J_final": _range_for_plasma_loading_metric(
            results,
            "field_energy_J_final",
        ),
        "circuit_current_A_final": _range_for_plasma_loading_metric(
            results,
            "circuit_current_A_final",
        ),
        "circuit_terminal_voltage_V_final_step": (
            _range_for_plasma_loading_metric(
                results,
                "circuit_terminal_voltage_V_final_step",
            )
        ),
        "circuit_active_power_W_final_step": _range_for_plasma_loading_metric(
            results,
            "circuit_active_power_W_final_step",
        ),
        "electron_density_m3_max_retained": _range_for_plasma_loading_metric(
            results,
            "electron_density_m3_max_retained",
        ),
        "neutral_density_m3_max_retained": _range_for_plasma_loading_metric(
            results,
            "neutral_density_m3_max_retained",
        ),
        "source_backed_sigma_S_m_max_retained": (
            _range_for_plasma_loading_metric(
                results,
                "source_backed_sigma_S_m_max_retained",
            )
        ),
        "source_backed_resistivity_ohm_m_max_retained": (
            _range_for_plasma_loading_metric(
                results,
                "source_backed_resistivity_ohm_m_max_retained",
            )
        ),
        "conductivity_cfl_limited_fraction_max_retained": (
            _range_for_plasma_loading_metric(
                results,
                "conductivity_cfl_limited_fraction_max_retained",
            )
        ),
        "ionization_fraction_max_retained": _range_for_plasma_loading_metric(
            results,
            "ionization_fraction_max_retained",
        ),
    }
    scored_effect_observed = any(
        _range_exceeds_numeric_noise(packet)
        for packet in metric_ranges.values()
    )
    runtime_effect_observed = any(
        _range_exceeds_numeric_noise(packet)
        for packet in runtime_metric_ranges.values()
    )
    runtime_metric_available = any(
        packet["range"] is not None for packet in runtime_metric_ranges.values()
    )
    return {
        "held_fixed_factors": group["held_fixed_factors"],
        "varied_parameter": parameter_name,
        "varied_factor_values": sorted({
            float(result["candidate"]["parameter_factors"][parameter_name])
            for result in results
        }),
        "candidate_count": len(results),
        "metric_ranges": metric_ranges,
        "runtime_metric_ranges": runtime_metric_ranges,
        "runtime_metric_available": runtime_metric_available,
        "effect_observed_on_scored_metrics": scored_effect_observed,
        "effect_observed_on_runtime_metrics": runtime_effect_observed,
        "effect_observed": scored_effect_observed or runtime_effect_observed,
    }


def _range_for_metric(
    candidate_results: list[dict[str, Any]],
    metric_name: str,
) -> dict[str, Any]:
    values: list[float] = []
    for result in candidate_results:
        value = (
            result.get("scoring", {})
            .get("metrics", {})
            .get(metric_name)
        )
        if value is None:
            continue
        values.append(float(value))
    return _range_for_values(values)


def _range_for_plasma_loading_metric(
    candidate_results: list[dict[str, Any]],
    metric_name: str,
) -> dict[str, Any]:
    values: list[float] = []
    for result in candidate_results:
        value = result.get("plasma_loading_summary", {}).get(metric_name)
        if value is None:
            continue
        values.append(float(value))
    return _range_for_values(values)


def _range_for_values(values: list[float]) -> dict[str, Any]:
    finite = [value for value in values if math.isfinite(value)]
    if not finite:
        return {
            "min": None,
            "max": None,
            "range": None,
            "relative_range": None,
        }
    value_min = min(finite)
    value_max = max(finite)
    value_range = value_max - value_min
    return {
        "min": value_min,
        "max": value_max,
        "range": value_range,
        "relative_range": value_range / max(max(abs(value_min), abs(value_max)), 1.0e-300),
    }


def _range_exceeds_numeric_noise(packet: dict[str, Any]) -> bool:
    value_range = packet.get("range")
    relative_range = packet.get("relative_range")
    if value_range is None or relative_range is None:
        return False
    return bool(abs(float(value_range)) > 1.0e-12 and abs(float(relative_range)) > 1.0e-9)


def _relative_error(value: float, target: float) -> float:
    return abs(float(value) - float(target)) / max(abs(float(target)), 1.0e-300)


def _waveform_shape_metrics(
    *,
    currents_A: list[float],
    times_s: list[float],
    target_observables: dict[str, Any],
) -> dict[str, Any]:
    waveform = target_observables.get("waveform")
    if not isinstance(waveform, dict):
        return {"waveform_target_present": False}

    target_time_us = _np_array_or_empty(waveform.get("time_us"))
    target_current_kA = _np_array_or_empty(waveform.get("current_kA"))
    if target_time_us.size == 0 or target_current_kA.size == 0:
        return {
            "waveform_target_present": True,
            "waveform_status": "target_waveform_missing_arrays",
            "waveform_score_included": False,
        }
    if target_time_us.shape != target_current_kA.shape:
        return {
            "waveform_target_present": True,
            "waveform_status": "target_waveform_shape_mismatch",
            "waveform_score_included": False,
            "waveform_target_point_count": int(target_time_us.size),
        }

    finite_target = np.isfinite(target_time_us) & np.isfinite(target_current_kA)
    target_time_us = target_time_us[finite_target]
    target_current_kA = target_current_kA[finite_target]
    if target_time_us.size == 0:
        return {
            "waveform_target_present": True,
            "waveform_status": "target_waveform_no_finite_samples",
            "waveform_score_included": False,
        }

    order = np.argsort(target_time_us)
    target_time_us = target_time_us[order]
    target_current_kA = target_current_kA[order]
    sim_time_us = np.asarray(times_s, dtype=float) * 1.0e6
    sim_current_kA = np.asarray(currents_A, dtype=float) * 1.0e-3
    finite_sim = np.isfinite(sim_time_us) & np.isfinite(sim_current_kA)
    sim_time_us = sim_time_us[finite_sim]
    sim_current_kA = sim_current_kA[finite_sim]

    target_min = float(target_time_us[0])
    target_max = float(target_time_us[-1])
    coverage_start_us = max(target_min, 0.0)
    coverage_end_us = target_max
    scored_target_duration_us = max(coverage_end_us - coverage_start_us, 0.0)
    if sim_time_us.size:
        simulation_start_us = float(np.min(sim_time_us))
        simulation_end_us = float(np.max(sim_time_us))
        covered_duration_us = max(
            0.0,
            min(simulation_end_us, coverage_end_us)
            - max(simulation_start_us, coverage_start_us),
        )
    else:
        covered_duration_us = 0.0
    temporal_coverage_fraction = (
        0.0
        if scored_target_duration_us <= 0.0
        else min(1.0, covered_duration_us / scored_target_duration_us)
    )
    required_coverage = float(
        target_observables.get(
            "minimum_waveform_coverage_fraction",
            DEFAULT_MINIMUM_WAVEFORM_COVERAGE_FRACTION,
        )
    )
    common_metrics: dict[str, Any] = {
        "waveform_target_present": True,
        "waveform_score_included": False,
        "waveform_target_point_count": int(target_time_us.size),
        "waveform_target_series_sha256": waveform.get("series_sha256"),
        "waveform_target_time_range_us": [target_min, target_max],
        "waveform_scored_time_range_us": [coverage_start_us, coverage_end_us],
        "waveform_pretrigger_time_excluded_from_coverage_us": max(0.0, -target_min),
        "waveform_temporal_coverage_fraction_of_target": temporal_coverage_fraction,
        "waveform_required_coverage_fraction": required_coverage,
        "waveform_coverage_horizon_limited": (
            temporal_coverage_fraction < required_coverage
        ),
    }
    if sim_time_us.size == 0:
        return common_metrics | {"waveform_status": "simulation_waveform_empty"}

    overlap = (sim_time_us >= target_min) & (sim_time_us <= target_max)
    overlap_time_us = sim_time_us[overlap]
    overlap_sim_kA = sim_current_kA[overlap]
    minimum_points = int(target_observables.get("minimum_waveform_overlap_points", 3))
    if overlap_time_us.size < minimum_points:
        return common_metrics | {
            "waveform_status": "insufficient_waveform_overlap",
            "waveform_overlap_point_count": int(overlap_time_us.size),
            "waveform_minimum_overlap_points": minimum_points,
        }

    overlap_target_kA = np.interp(
        overlap_time_us,
        target_time_us,
        target_current_kA,
    )
    residual_kA = overlap_sim_kA - overlap_target_kA
    target_peak_kA = max(_peak_abs_array(target_current_kA), 1.0e-300)
    rmse_kA = float(np.sqrt(np.mean(residual_kA * residual_kA)))
    nrmse = rmse_kA / target_peak_kA
    return common_metrics | {
        "waveform_status": "scored_waveform_shape_overlap",
        "waveform_score_included": math.isfinite(nrmse),
        "waveform_overlap_point_count": int(overlap_time_us.size),
        "waveform_minimum_overlap_points": minimum_points,
        "waveform_rmse_kA": rmse_kA,
        "waveform_mae_kA": float(np.mean(np.abs(residual_kA))),
        "waveform_mean_signed_error_kA": float(np.mean(residual_kA)),
        "waveform_max_abs_error_kA": float(np.max(np.abs(residual_kA))),
        "waveform_nrmse_fraction": float(nrmse),
    }


def _candidate_horizon_limited_reasons(result: dict[str, Any]) -> list[str]:
    metrics = result.get("scoring", {}).get("metrics", {})
    if not isinstance(metrics, dict):
        return []
    reasons: list[str] = []
    if metrics.get("peak_at_final_sample") is True:
        reasons.append("peak_at_final_sample")
    if metrics.get("waveform_coverage_horizon_limited") is True:
        reasons.append("waveform_temporal_coverage_below_required")
    return reasons


def _horizon_recommendation(
    *,
    target_observables: dict[str, Any],
    candidate_results: tuple[dict[str, Any], ...],
    identifiability: dict[str, Any],
) -> dict[str, Any]:
    horizon_limited_ids = tuple(identifiability.get("horizon_limited_candidate_ids", ()))
    if not horizon_limited_ids:
        return {
            "status": "no_horizon_extension_required_by_current_classifier",
            "recommended_target_time_s": None,
        }

    limited = [
        result
        for result in candidate_results
        if result.get("candidate", {}).get("candidate_id") in horizon_limited_ids
    ]
    current_horizon_s = max(
        (float(result.get("final_time_s", 0.0)) for result in limited),
        default=0.0,
    )
    recommendations: list[float] = []
    reasons: list[str] = []
    for result in limited:
        metrics = result.get("scoring", {}).get("metrics", {})
        if not isinstance(metrics, dict):
            continue
        if metrics.get("peak_at_final_sample") is True:
            recommendations.append(max(current_horizon_s * 1.25, 0.0))
            reasons.append("extend_25_percent_beyond_terminal_candidate_peak")
        if metrics.get("waveform_coverage_horizon_limited") is True:
            target_range = target_observables.get("target_time_range_us")
            if (
                isinstance(target_range, (list, tuple))
                and len(target_range) == 2
                and target_range[1] is not None
            ):
                target_end_s = max(0.0, float(target_range[1]) * 1.0e-6)
                recommendations.append(target_end_s)
                reasons.append("cover_full_target_waveform_time_range")
    target_peak_time = _optional_positive_float(target_observables.get("peak_time_s"))
    if target_peak_time is not None:
        recommendations.append(target_peak_time * 1.25)

    recommended = max(recommendations, default=current_horizon_s * 1.25)
    return {
        "status": "extend_runtime_before_parameter_conclusion",
        "horizon_limited_candidate_ids": list(horizon_limited_ids),
        "current_horizon_s": current_horizon_s,
        "recommended_target_time_s": recommended,
        "estimated_step_multiplier": (
            None if current_horizon_s <= 0.0 else recommended / current_horizon_s
        ),
        "reasons": sorted(set(reasons)),
        "can_conclude_without_extension": False,
    }


def _np_array_or_empty(value: Any) -> Any:
    try:
        return np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        return np.asarray((), dtype=float)


def _peak_abs_array(values: Any) -> float:
    array = np.asarray(values, dtype=float)
    return 0.0 if array.size == 0 else float(np.max(np.abs(array)))


def _positive_float(value: float, name: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be positive and finite")
    return result


def _optional_positive_float(value: Any) -> float | None:
    if value is None:
        return None
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        return None
    return result
