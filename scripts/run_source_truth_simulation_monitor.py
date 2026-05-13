#!/usr/bin/env python3
"""Run source-scoped simulator monitoring across presets and waveform devices.

This is an engineering monitor.  It runs the simulator surfaces that are cheap
enough for a local audit, records failures/nonfinite outputs, and labels
scientific authority from the local device registry.  It does not promote any
draft or reconstructed evidence to accepted validation status.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app_engine import run_simulation_core  # noqa: E402
from dpf.circuit.rlc_solver import RLCSolver  # noqa: E402
from dpf.constants import k_B, m_D2  # noqa: E402
from dpf.core.bases import CouplingState  # noqa: E402
from dpf.fluid.snowplow import SnowplowModel  # noqa: E402
from dpf.presets import get_preset, list_presets  # noqa: E402
from dpf.validation._calibration_data import _DEFAULT_DEVICE_PCF  # noqa: E402
from dpf.validation.experimental_comparison import nrmse_peak  # noqa: E402
from dpf.validation.experimental_devices import (  # noqa: E402
    DEVICES,
    ExperimentalDevice,
    get_validation_ready_devices,
)
from dpf.validation.kr_targets import (  # noqa: E402
    faeton_i_high_voltage_dpf_targets,
    lee_course_nx2_neon_phase_timing_example_targets,
    mjolnir_first_experiments_targets,
)

CURRENT_NRMSE_FENCE = 0.35
DEFAULT_OUTPUT_STEM = "SOURCE_TRUTH_SIMULATION_MONITOR_2026_05_12"
SOURCE_CONFIG_REL_TOL = 5e-3
TOP_LEVEL_MONITOR_CATEGORIES = (
    "operational_failure",
    "source_gap",
    "model_coverage_gap",
    "numerical_verification_gap",
    "validation_ready_accuracy_failure",
)


@dataclass
class CircuitRun:
    t: np.ndarray
    current: np.ndarray
    summary: dict[str, Any]


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return [_json_safe(v) for v in value.tolist()]
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return str(value)
    return value


def _array_nonfinite_counts(result: dict[str, Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for key, value in result.items():
        if isinstance(value, np.ndarray) and np.issubdtype(value.dtype, np.number):
            n_bad = int(np.size(value) - np.count_nonzero(np.isfinite(value)))
            if n_bad:
                counts[key] = n_bad
    return counts


def _source_state(device_name: str, dev: ExperimentalDevice) -> dict[str, Any]:
    validation_ready = device_name in get_validation_ready_devices()
    blockers: list[str] = []
    if dev.kr_status != "verified":
        blockers.append(f"kr_status={dev.kr_status}")
    if dev.reliability != "measured":
        blockers.append(f"reliability={dev.reliability}")
    if dev.waveform_t is None or dev.waveform_I is None:
        blockers.append("waveform_missing")
    if dev.waveform_provenance != "measured":
        blockers.append(f"waveform_provenance={dev.waveform_provenance or 'unset'}")
    if dev.waveform_kr_status != "verified":
        blockers.append(f"waveform_kr_status={dev.waveform_kr_status}")
    return {
        "validation_ready": validation_ready,
        "kr_status": dev.kr_status,
        "reliability": dev.reliability,
        "waveform_provenance": dev.waveform_provenance,
        "waveform_kr_status": dev.waveform_kr_status,
        "blockers": blockers,
    }


def _source_config_flags(
    preset_config: dict[str, Any],
    dev: ExperimentalDevice,
    *,
    rel_tol: float = SOURCE_CONFIG_REL_TOL,
) -> list[str]:
    flags: list[str] = []
    circuit = preset_config.get("circuit", {})
    snowplow = preset_config.get("snowplow", {})

    def compare(path: str, observed: Any, expected: Any) -> None:
        if expected is None:
            return
        if observed is None:
            flags.append(f"{path}_missing_expected={expected:.6g}")
            return
        observed_f = float(observed)
        expected_f = float(expected)
        scale = max(abs(expected_f), 1e-30)
        if abs(observed_f - expected_f) / scale > rel_tol:
            flags.append(
                f"{path}_mismatch_observed={observed_f:.6g}_expected={expected_f:.6g}"
            )

    def compare_faeton_two_step_radial_fit() -> None:
        primary = snowplow.get("radial_current_fraction", snowplow.get("current_fraction", 0.7))
        secondary = snowplow.get("radial_current_fraction_2")
        if primary is None or secondary is None:
            flags.append("snowplow.radial_current_fraction_two_step_missing_for_faeton")
            return

        targets = faeton_i_high_voltage_dpf_targets()
        shots = targets["current_waveform_targets"]["table_3_shots"]
        primary_f = float(primary)
        secondary_f = float(secondary)
        matched_shots = [
            int(shot["shot"])
            for shot in shots
            if abs(primary_f - float(shot["fcr"])) <= rel_tol * max(abs(float(shot["fcr"])), 1e-30)
            and abs(secondary_f - float(shot["fcr2"])) <= rel_tol * max(abs(float(shot["fcr2"])), 1e-30)
        ]
        if not matched_shots:
            allowed = ",".join(f"{shot['fcr']:.6g}/{shot['fcr2']:.6g}" for shot in shots)
            flags.append(
                "snowplow.radial_current_fraction_two_step_not_in_faeton_table3_"
                f"observed={primary_f:.6g}/{secondary_f:.6g}_allowed={allowed}"
            )

        transition = snowplow.get("radial_transition_time")
        phase_timing = targets.get("phase_timing", {})
        if transition is not None and not phase_timing.get("absolute_phase_times_available_in_extract", True):
            flags.append(
                "snowplow.radial_transition_time_not_in_faeton_kr_extract_"
                f"observed={float(transition):.6g}"
            )

    compare("rho0", preset_config.get("rho0"), dev.fill_pressure_torr * 133.322 * m_D2 / (k_B * 300.0))
    compare("circuit.C", circuit.get("C"), dev.capacitance)
    compare("circuit.V0", circuit.get("V0"), dev.voltage)
    compare("circuit.L0", circuit.get("L0"), dev.inductance)
    compare("circuit.R0", circuit.get("R0"), dev.resistance)
    compare("circuit.anode_radius", circuit.get("anode_radius"), dev.anode_radius)
    compare("circuit.cathode_radius", circuit.get("cathode_radius"), dev.cathode_radius)

    if snowplow:
        compare("snowplow.anode_length", snowplow.get("anode_length"), dev.anode_length)
        compare(
            "snowplow.fill_pressure_Pa",
            snowplow.get("fill_pressure_Pa", 400.0),
            dev.fill_pressure_torr * 133.322,
        )
        compare("snowplow.current_fraction", snowplow.get("current_fraction", 0.7), dev.lee_fc)
        compare("snowplow.mass_fraction", snowplow.get("mass_fraction", 0.15), dev.lee_fm)
        compare(
            "snowplow.radial_mass_fraction",
            snowplow.get("radial_mass_fraction", snowplow.get("mass_fraction", 0.15)),
            dev.lee_fmr,
        )
        if dev.name == "FAETON-I" and snowplow.get("radial_current_fraction_2") is not None:
            compare_faeton_two_step_radial_fit()
        else:
            compare(
                "snowplow.radial_current_fraction",
                snowplow.get("radial_current_fraction", snowplow.get("current_fraction", 0.7)),
                dev.lee_fcr,
            )
    return flags


def _top_level_monitor_categories(row: dict[str, Any]) -> list[str]:
    """Classify monitor rows into the source-truth dashboard categories."""
    categories: list[str] = []
    status = str(row.get("workflow_status", ""))
    if status == "broken" or row.get("error") or row.get("nonfinite_counts"):
        nonfinite_counts = row.get("nonfinite_counts")
        if not isinstance(nonfinite_counts, dict) or sum(nonfinite_counts.values()) > 0:
            categories.append("operational_failure")
    if row.get("source_gap_flags") or row.get("source_config_flags"):
        categories.append("source_gap")
    if row.get("model_coverage_flags"):
        categories.append("model_coverage_gap")
    if row.get("numerical_verification_flags"):
        categories.append("numerical_verification_gap")
    validation_ready = bool(
        row.get("source_state", {}).get("validation_ready")
        or row.get("reference_source_state", {}).get("validation_ready")
    )
    if validation_ready and row.get("accuracy_flags"):
        categories.append("validation_ready_accuracy_failure")
    return [category for category in TOP_LEVEL_MONITOR_CATEGORIES if category in categories]


def _source_gap_flags(device_name: str, dev: ExperimentalDevice, state: dict[str, Any]) -> list[str]:
    flags: list[str] = []
    if state.get("validation_ready"):
        return flags

    blockers = set(state.get("blockers", []))
    if dev.reliability == "reference_only":
        flags.append("reference_only_device_not_scientific_validation_target")
    if "waveform_missing" in blockers:
        flags.append("measured_current_waveform_missing")
    if dev.waveform_provenance == "reconstructed":
        flags.append("waveform_reconstructed_not_digitized")
    if getattr(dev, "waveform_kr_status", "unverified") != "verified":
        flags.append(f"waveform_kr_status={getattr(dev, 'waveform_kr_status', 'unverified')}")

    if device_name == "NX2":
        target = lee_course_nx2_neon_phase_timing_example_targets()
        missing = ",".join(str(item) for item in target["missing_for_predictive_tier2"])
        flags.append(f"nx2_course_example_not_same_shot_deuterium_target_missing={missing}")
    return flags


def _model_coverage_flags(device_name: str, dev: ExperimentalDevice) -> list[str]:
    flags: list[str] = []
    if device_name == "MJOLNIR":
        target = mjolnir_first_experiments_targets()
        current_targets = target["current_waveform_targets"]
        if current_targets.get("restrike_model_required_to_match_current_traces") and (
            getattr(dev, "lee_fcr2", None) is None
            or getattr(dev, "lee_radial_transition_time", None) is None
        ):
            flags.append(
                "mjolnir_restrike_current_trace_model_required_by_kr_"
                "but_no_accepted_timing_or_magnitude_parameters"
            )
    return flags


def _simulate_circuit_device(device_name: str, dev: ExperimentalDevice) -> CircuitRun:
    p_pa = dev.fill_pressure_torr * 133.322
    rho0 = (p_pa / (k_B * 300.0)) * m_D2
    circuit = RLCSolver(
        C=dev.capacitance,
        V0=dev.voltage,
        L0=dev.inductance,
        R0=dev.resistance,
        anode_radius=dev.anode_radius,
        cathode_radius=dev.cathode_radius,
        crowbar_enabled=True,
        crowbar_mode="voltage_zero",
        crowbar_resistance=dev.crowbar_resistance,
    )
    fc = dev.lee_fc or 0.7
    fm = dev.lee_fm or 0.13
    fmr = dev.lee_fmr or 0.1
    fcr = dev.lee_fcr or fc
    pcf = _DEFAULT_DEVICE_PCF.get(device_name, 1.0)
    snowplow = SnowplowModel(
        anode_radius=dev.anode_radius,
        cathode_radius=dev.cathode_radius,
        fill_density=rho0,
        anode_length=dev.anode_length,
        mass_fraction=fm,
        current_fraction=fc,
        radial_mass_fraction=fmr,
        radial_current_fraction=fcr,
        radial_current_fraction_2=getattr(dev, "lee_fcr2", None),
        radial_transition_time=getattr(dev, "lee_radial_transition_time", None),
        fill_pressure_Pa=p_pa,
        pinch_column_fraction=pcf,
    )

    waveform_end = float(dev.waveform_t[-1]) if dev.waveform_t is not None else 0.0
    sim_time = max(8e-6, 2.5 * (dev.current_rise_time or 0.0), 1.2 * waveform_end)
    sim_time = min(sim_time, 40e-6)
    dt = min(1e-9, sim_time / 10000.0)
    n_steps = int(sim_time / dt)
    coupling = CouplingState()
    t_values = [0.0]
    currents = [0.0]
    phases: set[str] = set()
    first_nonfinite: dict[str, Any] | None = None
    started = time.perf_counter()
    t = 0.0

    for step in range(n_steps):
        sp_result = snowplow.step(dt, coupling.current)
        phases.add(str(sp_result.get("phase", snowplow.phase)))
        coupling.Lp = sp_result["L_plasma"]
        coupling.dL_dt = sp_result["dL_dt"]
        coupling.R_plasma = sp_result.get("R_plasma", 0.0)
        coupling = circuit.step(coupling, 0.0, dt)
        t += dt
        t_values.append(t)
        currents.append(coupling.current)
        checks = {
            "current": coupling.current,
            "voltage": circuit.voltage,
            "Lp": coupling.Lp,
            "dL_dt": coupling.dL_dt,
            "R_plasma": coupling.R_plasma,
        }
        for field, val in checks.items():
            if not np.isfinite(val):
                first_nonfinite = {
                    "step": step,
                    "time_s": t,
                    "field": field,
                    "value": str(val),
                }
                break
        if first_nonfinite:
            break

    t_arr = np.asarray(t_values)
    i_arr = np.asarray(currents)
    peak_idx = int(np.argmax(np.abs(i_arr))) if len(i_arr) else 0
    summary = {
        "sim_time_s": sim_time,
        "dt_s": dt,
        "steps_requested": n_steps,
        "steps_completed": len(i_arr) - 1,
        "elapsed_s": time.perf_counter() - started,
        "peak_current_A": float(np.max(np.abs(i_arr))) if len(i_arr) else 0.0,
        "peak_current_time_s": float(t_arr[peak_idx]) if len(t_arr) else 0.0,
        "phases_seen": sorted(phases),
        "first_nonfinite": first_nonfinite,
        "fc": fc,
        "fm": fm,
        "fmr": fmr,
        "fcr": fcr,
        "fcr2": getattr(dev, "lee_fcr2", None),
        "radial_transition_time_s": getattr(dev, "lee_radial_transition_time", None),
        "pinch_column_fraction": pcf,
    }
    return CircuitRun(t=t_arr, current=i_arr, summary=summary)


def _device_metrics(device_name: str, dev: ExperimentalDevice) -> dict[str, Any]:
    state = _source_state(device_name, dev)
    source_gaps = _source_gap_flags(device_name, dev, state)
    model_gaps = _model_coverage_flags(device_name, dev)
    row: dict[str, Any] = {
        "device": device_name,
        "source_state": state,
        "reference": dev.reference,
        "lee_reference": dev.lee_reference,
        "source_gap_flags": source_gaps,
        "model_coverage_flags": model_gaps,
    }
    try:
        run = _simulate_circuit_device(device_name, dev)
    except Exception as exc:
        row.update({"workflow_status": "broken", "error": repr(exc)})
        row["top_level_categories"] = _top_level_monitor_categories(row)
        return row

    summary = run.summary
    row["summary"] = summary
    if summary.get("first_nonfinite"):
        row["workflow_status"] = "broken"
        row["top_level_categories"] = _top_level_monitor_categories(row)
        return row

    i_peak = float(summary["peak_current_A"])
    t_peak = float(summary["peak_current_time_s"])
    peak_err = abs(i_peak - dev.peak_current) / max(abs(dev.peak_current), 1e-30)
    timing_err = abs(t_peak - dev.current_rise_time) / max(abs(dev.current_rise_time), 1e-30)
    row["peak_current_error_rel"] = peak_err
    row["timing_error_rel"] = timing_err

    flags: list[str] = []
    peak_two_sigma = 2.0 * dev.peak_current_uncertainty if dev.peak_current_uncertainty else None
    time_two_sigma = 2.0 * dev.rise_time_uncertainty if dev.rise_time_uncertainty else None
    if peak_two_sigma is not None and peak_err > peak_two_sigma:
        flags.append(f"peak_current_error>{peak_two_sigma:.3f}_two_sigma")
    if time_two_sigma is not None and timing_err > time_two_sigma:
        flags.append(f"timing_error>{time_two_sigma:.3f}_two_sigma")

    if dev.waveform_t is not None and dev.waveform_I is not None:
        row["waveform_points"] = int(len(dev.waveform_t))
        row["nrmse_full"] = nrmse_peak(run.t, run.current, dev.waveform_t, dev.waveform_I)
        row["nrmse_rise"] = nrmse_peak(
            run.t,
            run.current,
            dev.waveform_t,
            dev.waveform_I,
            max_time=dev.current_rise_time,
        )
        row["nrmse_to_dip"] = nrmse_peak(
            run.t,
            run.current,
            dev.waveform_t,
            dev.waveform_I,
            truncate_at_dip=True,
        )
        if row["nrmse_full"] > CURRENT_NRMSE_FENCE:
            flags.append(f"nrmse_full>{CURRENT_NRMSE_FENCE:.2f}_pipeline_fence")
    else:
        row["waveform_points"] = 0

    row["accuracy_flags"] = flags
    if flags and state["validation_ready"]:
        row["workflow_status"] = "accuracy_review_needed"
    elif model_gaps:
        row["workflow_status"] = "model_coverage_review_needed"
    elif source_gaps:
        row["workflow_status"] = "source_gap_review_needed"
    elif flags:
        row["workflow_status"] = "engineering_issue_nonaccepting"
    elif state["validation_ready"]:
        row["workflow_status"] = "within_current_pipeline"
    else:
        row["workflow_status"] = "engineering_only_nonaccepting"
    row["top_level_categories"] = _top_level_monitor_categories(row)
    return row


def _device_to_preset_map() -> dict[str, str]:
    mapping: dict[str, str] = {}
    for preset in list_presets():
        device = preset.get("device") or ""
        name = preset.get("name") or ""
        if device and name and device not in mapping:
            mapping[device] = name
    mapping.update(
        {
            "PF-1000": "pf1000",
            "PF-1000-Gribkov": "pf1000",
            "PF-1000-16kV": "pf1000_akel",
            "PF-1000-20kV": "pf1000_20kv",
            "UNU-ICTP": "unu_ictp",
            "NX2": "nx2",
            "POSEIDON-60kV": "poseidon_60kv",
            "FAETON-I": "faeton",
            "MJOLNIR": "mjolnir",
        }
    )
    return mapping


def _sim_time_for_preset(preset: dict[str, Any]) -> float:
    dev = _reference_device_for_preset(preset)
    if dev is None:
        return 5.0
    waveform_end_us = float(dev.waveform_t[-1] * 1e6) if dev.waveform_t is not None else 0.0
    return min(max(2.5 * dev.current_rise_time * 1e6, 1.2 * waveform_end_us, 2.0), 40.0)


def _reference_device_name_for_preset(preset: dict[str, Any]) -> str | None:
    name = str(preset.get("name") or "")
    if name == "pf1000_akel":
        return "PF-1000-16kV"
    if name == "pf1000_20kv":
        return "PF-1000-20kV"
    device_name = str(preset.get("device") or "")
    return device_name if device_name in DEVICES else None


def _reference_device_for_preset(preset: dict[str, Any]) -> ExperimentalDevice | None:
    name = _reference_device_name_for_preset(preset)
    return DEVICES.get(name) if name else None


def _preset_monitor() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for preset in list_presets():
        name = str(preset["name"])
        sim_time_us = _sim_time_for_preset(preset)
        ref_name = _reference_device_name_for_preset(preset)
        ref_dev = DEVICES.get(ref_name) if ref_name else None
        ref_state = _source_state(ref_name, ref_dev) if ref_name and ref_dev else None
        source_gaps = _source_gap_flags(ref_name, ref_dev, ref_state) if ref_name and ref_dev and ref_state else []
        model_gaps = _model_coverage_flags(ref_name, ref_dev) if ref_name and ref_dev else []
        preset_config = get_preset(name)
        source_config = _source_config_flags(preset_config, ref_dev) if ref_dev else []
        row: dict[str, Any] = {
            "preset": name,
            "device": preset.get("device", ""),
            "reference_device": ref_name or "",
            "reference_source_state": ref_state or {},
            "source_scope": preset.get("source_scope", ""),
            "source_scope_status": preset.get("source_scope_status", ""),
            "validation_status": preset.get("validation_status", ""),
            "can_support_validation_claims": preset.get("can_support_validation_claims"),
            "source_gap_flags": source_gaps,
            "model_coverage_flags": model_gaps,
            "source_config_flags": source_config,
            "sim_time_us": sim_time_us,
        }
        started = time.perf_counter()
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                result = run_simulation_core(name, sim_time_us=sim_time_us)
            warning_rows = [
                {
                    "category": item.category.__name__,
                    "message": str(item.message),
                    "filename": str(item.filename),
                    "lineno": item.lineno,
                }
                for item in caught
            ]
            nonfinite = _array_nonfinite_counts(result)
            accuracy_flags: list[str] = []
            ref_peak_error = None
            ref_timing_error = None
            if ref_dev is not None:
                ref_peak_error = abs(float(result.get("I_peak", 0.0)) * 1e6 - ref_dev.peak_current) / max(
                    abs(ref_dev.peak_current), 1e-30
                )
                ref_timing_error = abs(float(result.get("t_peak", 0.0)) * 1e-6 - ref_dev.current_rise_time) / max(
                    abs(ref_dev.current_rise_time), 1e-30
                )
                peak_two_sigma = (
                    2.0 * ref_dev.peak_current_uncertainty
                    if ref_dev.peak_current_uncertainty
                    else None
                )
                time_two_sigma = (
                    2.0 * ref_dev.rise_time_uncertainty
                    if ref_dev.rise_time_uncertainty
                    else None
                )
                if peak_two_sigma is not None and ref_peak_error > peak_two_sigma:
                    accuracy_flags.append(f"preset_peak_error>{peak_two_sigma:.3f}_two_sigma")
                if time_two_sigma is not None and ref_timing_error > time_two_sigma:
                    accuracy_flags.append(f"preset_timing_error>{time_two_sigma:.3f}_two_sigma")
            row.update(
                {
                    "workflow_status": "broken" if nonfinite else "completed",
                    "elapsed_s": time.perf_counter() - started,
                    "I_peak_MA": float(result.get("I_peak", 0.0)),
                    "t_peak_us": float(result.get("t_peak", 0.0)),
                    "n_steps": int(result.get("n_steps", 0)),
                    "dt_ns": float(result.get("dt_ns", 0.0)),
                    "dip_pct": float(result.get("dip_pct", 0.0)),
                    "nonfinite_counts": nonfinite,
                    "engine_status": result.get("engine_status", "completed"),
                    "reference_peak_error_rel": ref_peak_error,
                    "reference_timing_error_rel": ref_timing_error,
                    "accuracy_flags": accuracy_flags,
                    "warnings": warning_rows,
                }
            )
            if row["I_peak_MA"] <= 0:
                row["workflow_status"] = "broken"
                row["error"] = "nonpositive_peak_current"
            elif accuracy_flags and ref_state and ref_state.get("validation_ready"):
                row["workflow_status"] = "accuracy_review_needed"
            elif model_gaps:
                row["workflow_status"] = "model_coverage_review_needed"
            elif source_gaps and accuracy_flags:
                row["workflow_status"] = "source_gap_review_needed"
            elif source_config:
                row["workflow_status"] = "source_config_review_needed"
            elif warning_rows:
                row["workflow_status"] = "completed_with_warnings"
        except Exception as exc:
            row.update(
                {
                    "workflow_status": "broken",
                    "elapsed_s": time.perf_counter() - started,
                    "error": repr(exc),
                }
            )
        row["top_level_categories"] = _top_level_monitor_categories(row)
        rows.append(row)
        print(
            f"preset {name:18s} {row['workflow_status']:>10s} "
            f"I={row.get('I_peak_MA', 0.0):.3f} MA t={row.get('t_peak_us', 0.0):.3f} us"
        )
    return rows


def _pytest_lanes() -> list[dict[str, Any]]:
    lanes = [
        [
            sys.executable,
            "-m",
            "pytest",
            "tests/test_validation_ci.py",
            "-q",
            "-o",
            "addopts=",
        ],
        [
            sys.executable,
            "-m",
            "pytest",
            "tests/test_mhd_acceptance.py",
            "-q",
            "-rsx",
            "-o",
            "addopts=",
        ],
        [
            sys.executable,
            "-m",
            "pytest",
            "tests/test_quality_assessment.py::TestQualityAssessment::test_scientific_accuracy_gap_report_lists_remaining_work",
            "tests/test_mhd_physics_integration.py::test_predictive_readiness_exported_and_blocks_unvalidated_claims",
            "tests/test_akel_digitization_source_integrity.py",
            "tests/test_preset_source_scope.py",
            "tests/test_unreviewed_physics_metadata.py",
            "-q",
            "-o",
            "addopts=",
        ],
    ]
    rows = []
    for cmd in lanes:
        started = time.perf_counter()
        print(f"pytest lane: {' '.join(cmd[3:])}")
        proc = subprocess.run(
            cmd,
            cwd=REPO_ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        output = proc.stdout.strip()
        rows.append(
            {
                "command": cmd,
                "returncode": proc.returncode,
                "elapsed_s": time.perf_counter() - started,
                "status": "passed" if proc.returncode == 0 else "failed",
                "output_tail": "\n".join(output.splitlines()[-60:]),
            }
        )
        print(f"  -> rc={proc.returncode} elapsed={rows[-1]['elapsed_s']:.1f}s")
    return rows


def _write_markdown(report: dict[str, Any], path: Path) -> None:
    device_rows = report["circuit_waveform_devices"]
    preset_rows = report["preset_runs"]
    pytest_rows = report.get("pytest_lanes", [])
    lines: list[str] = []
    lines.append("# Source-Truth Simulation Monitor")
    lines.append("")
    lines.append(f"- Generated: `{report['generated_at']}`")
    lines.append("- Scientific authority: local `KnowledgeReference/` and registry source metadata only.")
    lines.append("- Boundary: this monitor does not accept draft digitizations or issue certificates.")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    for key, value in report["summary"].items():
        lines.append(f"- `{key}`: `{value}`")
    lines.append("")
    lines.append("## Top-Level Categories")
    lines.append("")
    category_counts = report["summary"].get("top_level_category_counts", {})
    for category in report.get("top_level_categories", []):
        lines.append(f"- `{category}`: `{category_counts.get(category, 0)}`")
    lines.append("")
    lines.append("## Circuit/Waveform Devices")
    lines.append("")
    lines.append("| Device | Source State | Workflow | Ipeak Err | Timing Err | NRMSE | Accuracy | Source Gap | Model Coverage |")
    lines.append("| --- | --- | --- | ---: | ---: | ---: | --- | --- | --- |")
    for row in device_rows:
        state = row.get("source_state", {})
        flags = ", ".join(row.get("accuracy_flags", [])) or "-"
        source_gaps = ", ".join(row.get("source_gap_flags", [])) or "-"
        model_gaps = ", ".join(row.get("model_coverage_flags", [])) or "-"
        source = "ready" if state.get("validation_ready") else "nonaccepting"
        if state.get("blockers"):
            source += ": " + ", ".join(state["blockers"])
        lines.append(
            "| {device} | {source} | {status} | {ipeak:.3%} | {timing:.3%} | {nrmse} | {flags} | {source_gaps} | {model_gaps} |".format(
                device=row["device"],
                source=source,
                status=row.get("workflow_status", "unknown"),
                ipeak=float(row.get("peak_current_error_rel", 0.0)),
                timing=float(row.get("timing_error_rel", 0.0)),
                nrmse=(
                    f"{float(row['nrmse_full']):.3f}"
                    if "nrmse_full" in row
                    else "-"
                ),
                flags=flags,
                source_gaps=source_gaps,
                model_gaps=model_gaps,
            )
        )
    lines.append("")
    lines.append("## Preset Runs")
    lines.append("")
    lines.append("| Preset | Reference | Source Scope Status | Workflow | Ipeak MA | Ipeak Err | tpeak us | Nonfinite | Warnings | Source Gap | Model Coverage | Source Config |")
    lines.append("| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |")
    for row in preset_rows:
        config_flags = row.get("source_config_flags", [])
        source_gaps = row.get("source_gap_flags", [])
        model_gaps = row.get("model_coverage_flags", [])
        lines.append(
            "| {preset} | {reference} | {scope} | {status} | {ipeak:.3f} | {ipeak_err} | {tpeak:.3f} | {nonfinite} | {warnings} | {source_gap} | {model_gap} | {source_config} |".format(
                preset=row["preset"],
                reference=row.get("reference_device", "") or row.get("device", ""),
                scope=row.get("source_scope_status", ""),
                status=row.get("workflow_status", "unknown"),
                ipeak=float(row.get("I_peak_MA", 0.0)),
                ipeak_err=(
                    f"{float(row['reference_peak_error_rel']):.3%}"
                    if row.get("reference_peak_error_rel") is not None
                    else "-"
                ),
                tpeak=float(row.get("t_peak_us", 0.0)),
                nonfinite=sum(row.get("nonfinite_counts", {}).values()),
                warnings=len(row.get("warnings", [])),
                source_gap=", ".join(source_gaps) if source_gaps else "-",
                model_gap=", ".join(model_gaps) if model_gaps else "-",
                source_config=", ".join(config_flags) if config_flags else "-",
            )
        )
    lines.append("")
    if pytest_rows:
        lines.append("## Pytest Lanes")
        lines.append("")
        for lane in pytest_rows:
            lines.append(f"- `{ ' '.join(lane['command']) }`")
            lines.append(f"  - status: `{lane['status']}`, elapsed_s: `{lane['elapsed_s']:.1f}`")
            lines.append("  - output tail:")
            lines.append("")
            lines.append("```text")
            lines.append(lane["output_tail"])
            lines.append("```")
            lines.append("")
    lines.append("## Findings")
    lines.append("")
    for finding in report["findings"]:
        lines.append(f"- {finding}")
    lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _build_findings(report: dict[str, Any]) -> list[str]:
    findings: list[str] = []
    broken_presets = [
        row for row in report["preset_runs"] if row.get("workflow_status") == "broken"
    ]
    inaccurate_presets = [
        row
        for row in report["preset_runs"]
        if row.get("workflow_status") == "accuracy_review_needed"
    ]
    source_gap_presets = [
        row
        for row in report["preset_runs"]
        if row.get("workflow_status") == "source_gap_review_needed"
    ]
    model_coverage_presets = [
        row
        for row in report["preset_runs"]
        if row.get("workflow_status") == "model_coverage_review_needed"
    ]
    warning_presets = [
        row for row in report["preset_runs"] if row.get("warnings")
    ]
    source_config_presets = [
        row for row in report["preset_runs"] if row.get("source_config_flags")
    ]
    accuracy_review = [
        row
        for row in report["circuit_waveform_devices"]
        if row.get("workflow_status") == "accuracy_review_needed"
    ]
    source_gap_devices = [
        row
        for row in report["circuit_waveform_devices"]
        if row.get("workflow_status") == "source_gap_review_needed"
    ]
    model_coverage_devices = [
        row
        for row in report["circuit_waveform_devices"]
        if row.get("workflow_status") == "model_coverage_review_needed"
    ]
    nonaccepting = [
        row
        for row in report["circuit_waveform_devices"]
        if not row.get("source_state", {}).get("validation_ready")
    ]
    if broken_presets:
        findings.append(
            "Broken preset workflow(s): "
            + ", ".join(f"{row['preset']} ({row.get('error', 'nonfinite')})" for row in broken_presets)
        )
    else:
        findings.append("All preset app-engine simulations completed without nonfinite arrays.")
    if inaccurate_presets:
        findings.append(
            "Validation-ready preset accuracy review needed for: "
            + ", ".join(
                f"{row['preset']} vs {row.get('reference_device', 'reference')} "
                f"[{', '.join(row.get('accuracy_flags', []))}]"
                for row in inaccurate_presets
            )
        )
    else:
        findings.append("No validation-ready preset crossed the current monitor accuracy flags.")
    if source_gap_presets:
        findings.append(
            "Preset source-gap review needed for: "
            + "; ".join(
                f"{row['preset']} vs {row.get('reference_device', 'reference')} "
                f"[{', '.join(row.get('source_gap_flags', []))}]"
                for row in source_gap_presets
            )
        )
    if model_coverage_presets:
        findings.append(
            "Preset model-coverage review needed for: "
            + "; ".join(
                f"{row['preset']} vs {row.get('reference_device', 'reference')} "
                f"[{', '.join(row.get('model_coverage_flags', []))}]"
                for row in model_coverage_presets
            )
        )
    if warning_presets:
        findings.append(
            "Preset runtime warnings captured for: "
            + ", ".join(
                f"{row['preset']} ({len(row.get('warnings', []))})"
                for row in warning_presets
            )
        )
    if source_config_presets:
        findings.append(
            "Preset source-config review needed for: "
            + "; ".join(
                f"{row['preset']} vs {row.get('reference_device', 'reference')} "
                f"[{', '.join(row.get('source_config_flags', []))}]"
                for row in source_config_presets
            )
        )
    if accuracy_review:
        findings.append(
            "Validation-ready waveform accuracy review needed for: "
            + ", ".join(f"{row['device']} [{', '.join(row.get('accuracy_flags', []))}]" for row in accuracy_review)
        )
    else:
        findings.append("No validation-ready waveform device crossed the current monitor accuracy flags.")
    if source_gap_devices:
        findings.append(
            "Waveform/device source-gap review needed for: "
            + "; ".join(
                f"{row['device']} [{', '.join(row.get('source_gap_flags', []))}]"
                for row in source_gap_devices
            )
        )
    if model_coverage_devices:
        findings.append(
            "Waveform/device model-coverage review needed for: "
            + "; ".join(
                f"{row['device']} [{', '.join(row.get('model_coverage_flags', []))}]"
                for row in model_coverage_devices
            )
        )
    if nonaccepting:
        findings.append(
            "Nonaccepting waveform/device evidence still simulated but not scored as accepted: "
            + ", ".join(row["device"] for row in nonaccepting)
        )
    failed_lanes = [
        row for row in report.get("pytest_lanes", []) if row.get("returncode") != 0
    ]
    if failed_lanes:
        findings.append(
            "Pytest monitor lane failure(s): "
            + "; ".join(" ".join(row["command"]) for row in failed_lanes)
        )
    elif report.get("pytest_lanes"):
        findings.append("All requested pytest monitor lanes returned success.")
    return findings


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--include-pytest-lanes", action="store_true")
    parser.add_argument("--output-stem", default=DEFAULT_OUTPUT_STEM)
    args = parser.parse_args()

    out_json = REPO_ROOT / "docs" / f"{args.output_stem}.json"
    out_md = REPO_ROOT / "docs" / f"{args.output_stem}.md"
    generated_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()

    print("Running source-scoped circuit waveform monitor...")
    device_rows = []
    for name in sorted(DEVICES):
        row = _device_metrics(name, DEVICES[name])
        device_rows.append(row)
        print(
            f"device {name:18s} {row.get('workflow_status', 'unknown'):>28s} "
            f"Ierr={float(row.get('peak_current_error_rel', 0.0)):.3%} "
            f"NRMSE={float(row.get('nrmse_full', 0.0)):.3f}"
        )

    print("Running app-engine preset monitor...")
    preset_rows = _preset_monitor()

    pytest_rows = _pytest_lanes() if args.include_pytest_lanes else []
    category_counts = {
        category: sum(
            1
            for row in [*device_rows, *preset_rows]
            if category in row.get("top_level_categories", [])
        )
        for category in TOP_LEVEL_MONITOR_CATEGORIES
    }
    if any(row.get("returncode") != 0 for row in pytest_rows):
        category_counts["numerical_verification_gap"] += 1

    summary = {
        "device_count": len(device_rows),
        "validation_ready_device_count": sum(
            1 for row in device_rows if row.get("source_state", {}).get("validation_ready")
        ),
        "preset_count": len(preset_rows),
        "broken_preset_count": sum(1 for row in preset_rows if row.get("workflow_status") == "broken"),
        "accuracy_review_preset_count": sum(
            1 for row in preset_rows if row.get("workflow_status") == "accuracy_review_needed"
        ),
        "source_gap_review_preset_count": sum(
            1 for row in preset_rows if row.get("workflow_status") == "source_gap_review_needed"
        ),
        "model_coverage_review_preset_count": sum(
            1 for row in preset_rows if row.get("workflow_status") == "model_coverage_review_needed"
        ),
        "warning_preset_count": sum(1 for row in preset_rows if row.get("warnings")),
        "source_config_review_preset_count": sum(
            1 for row in preset_rows if row.get("source_config_flags")
        ),
        "accuracy_review_device_count": sum(
            1
            for row in device_rows
            if row.get("workflow_status") == "accuracy_review_needed"
        ),
        "source_gap_review_device_count": sum(
            1 for row in device_rows if row.get("workflow_status") == "source_gap_review_needed"
        ),
        "model_coverage_review_device_count": sum(
            1 for row in device_rows if row.get("workflow_status") == "model_coverage_review_needed"
        ),
        "pytest_lane_count": len(pytest_rows),
        "pytest_failed_lane_count": sum(1 for row in pytest_rows if row.get("returncode") != 0),
        "top_level_category_counts": category_counts,
    }
    report = {
        "generated_at": generated_at,
        "repo_root": str(REPO_ROOT),
        "monitor_scope": {
            "science_authority": "local KnowledgeReference/ plus registry source metadata only",
            "validation_boundary": (
                "engineering monitor; does not accept draft digitizations, "
                "reconstructed waveforms, reference-only devices, or MHD/RADPF gates"
            ),
        },
        "top_level_categories": list(TOP_LEVEL_MONITOR_CATEGORIES),
        "summary": summary,
        "circuit_waveform_devices": device_rows,
        "preset_runs": preset_rows,
        "pytest_lanes": pytest_rows,
    }
    report["findings"] = _build_findings(report)

    out_json.write_text(
        json.dumps(_json_safe(report), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_markdown(report, out_md)
    print(f"wrote {out_json}")
    print(f"wrote {out_md}")
    return 1 if summary["broken_preset_count"] or summary["pytest_failed_lane_count"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
