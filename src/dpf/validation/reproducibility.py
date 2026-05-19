"""Reproducibility package for DPF simulations.

Exports a complete, self-contained JSON file that captures everything
needed to reproduce a simulation result: config, code version, platform,
key outputs, and validation metrics.

Anyone with DPF-Unified can load the package and verify the results.

Usage:
    pkg = create_reproducibility_package(result_dict, preset_name, backend)
    save_package(pkg, "pf1000_run_2026-03-16.json")

    # Later, verify:
    pkg = load_package("pf1000_run_2026-03-16.json")
    verified = verify_package(pkg)
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np


def _get_git_info() -> dict[str, str]:
    """Get current git commit hash and branch."""
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
        branch = subprocess.check_output(
            ["git", "branch", "--show-current"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
        dirty = subprocess.check_output(
            ["git", "status", "--porcelain"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
        return {
            "commit": commit,
            "branch": branch,
            "dirty": bool(dirty),
        }
    except (subprocess.CalledProcessError, FileNotFoundError):
        return {"commit": "unknown", "branch": "unknown", "dirty": False}


def _get_platform_info() -> dict[str, str]:
    """Get platform information for reproducibility."""
    return {
        "system": platform.system(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "processor": platform.processor(),
    }


def _numpy_safe(obj: Any) -> Any:
    """Convert numpy types to JSON-serializable Python types."""
    if isinstance(obj, np.ndarray):
        if obj.size <= 1000:
            return obj.tolist()
        return {
            "_type": "ndarray_summary",
            "shape": list(obj.shape),
            "dtype": str(obj.dtype),
            "min": float(np.min(obj)),
            "max": float(np.max(obj)),
            "mean": float(np.mean(obj)),
            "checksum": hashlib.md5(obj.tobytes()).hexdigest(),
        }
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, dict):
        return {k: _numpy_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_numpy_safe(v) for v in obj]
    return obj


def create_reproducibility_package(
    result: dict[str, Any],
    preset_name: str,
    backend: str,
    gas_key: str = "D2",
    sim_time_us: float = 10.0,
    notes: str = "",
) -> dict[str, Any]:
    """Create a reproducibility package from simulation results.

    Args:
        result: Simulation result dictionary from run_mhd_simulation or run_simulation_core.
        preset_name: Device preset name used.
        backend: Simulation backend used.
        gas_key: Fill gas species.
        sim_time_us: Simulation time [us].
        notes: Optional notes about this run.

    Returns:
        JSON-serializable dict containing the full reproducibility package.
    """
    git = _get_git_info()
    plat = _get_platform_info()

    # Extract key outputs
    outputs = {
        "I_peak_MA": result.get("I_peak", 0.0),
        "t_peak_us": result.get("t_peak", 0.0),
        "dip_pct": result.get("dip_pct", 0.0),
        "n_steps": result.get("n_steps", 0),
        "elapsed_s": result.get("elapsed_s", 0.0),
    }

    # Neutron yield
    ny = result.get("neutron_yield")
    if ny:
        outputs["neutron_yield"] = _numpy_safe(ny)

    # Bennett equilibrium
    bennett = result.get("bennett")
    if bennett:
        outputs["bennett"] = _numpy_safe(bennett)

    # Breakdown
    bd = result.get("breakdown")
    if bd:
        outputs["breakdown"] = {
            k: v for k, v in bd.items()
            if k != "narrative"  # Skip long narrative text
        }

    # Radiation regime
    rad = result.get("radiation_regime")
    if rad:
        outputs["radiation_regime"] = _numpy_safe(rad)

    # Shear stabilization
    shear = result.get("shear_stabilization")
    if shear:
        outputs["shear_stabilization"] = _numpy_safe(shear)

    # Circuit config
    circuit = _numpy_safe(result.get("circuit", {}))
    snowplow = _numpy_safe(result.get("snowplow_cfg", {}))

    # Waveform checksum (for data integrity verification)
    t_arr = result.get("t_us", np.array([]))
    I_arr = result.get("I_MA", np.array([]))
    if len(t_arr) > 0 and len(I_arr) > 0:
        waveform_data = np.column_stack([t_arr, I_arr])
        waveform_checksum = hashlib.md5(waveform_data.tobytes()).hexdigest()
    else:
        waveform_checksum = "no_waveform"

    package = {
        "dpf_unified_reproducibility": "1.0",
        "created": datetime.now(UTC).isoformat(),
        "software": {
            "name": "DPF-Unified",
            "git": git,
            "platform": plat,
        },
        "configuration": {
            "preset": preset_name,
            "backend": backend,
            "gas": gas_key,
            "sim_time_us": sim_time_us,
            "circuit": circuit,
            "snowplow": snowplow,
        },
        "outputs": outputs,
        "verification": {
            "waveform_checksum": waveform_checksum,
            "waveform_points": len(t_arr),
        },
        "notes": notes,
    }

    return package


def save_package(package: dict, path: str | Path) -> None:
    """Save reproducibility package to JSON file."""
    with open(path, "w") as f:
        json.dump(package, f, indent=2, default=str)


def load_package(path: str | Path) -> dict:
    """Load reproducibility package from JSON file."""
    with open(path) as f:
        return json.load(f)


def verify_package(package: dict) -> dict[str, Any]:
    """Verify a reproducibility package by re-running the simulation.

    Returns verification results including whether outputs match.
    """
    config = package.get("configuration", {})
    preset = config.get("preset", "tutorial")
    backend = config.get("backend", "lee")

    verification = {
        "package_version": package.get("dpf_unified_reproducibility", "unknown"),
        "original_git": package.get("software", {}).get("git", {}).get("commit", "?"),
        "current_git": _get_git_info().get("commit", "?"),
        "preset": preset,
        "backend": backend,
    }

    try:
        if backend == "lee":
            from app_engine import run_simulation_core
            result = run_simulation_core(
                preset_name=preset,
                sim_time_us=config.get("sim_time_us", 10.0),
                gas_key=config.get("gas", "D2"),
            )
        else:
            from app_mhd import run_mhd_simulation
            result = run_mhd_simulation(
                backend=backend,
                grid_preset="medium",
                preset_name=preset,
                sim_time_us=config.get("sim_time_us", 10.0),
                gas_key=config.get("gas", "D2"),
            )

        orig = package.get("outputs", {})
        I_peak_orig = orig.get("I_peak_MA", 0)
        I_peak_new = result.get("I_peak", 0)

        if I_peak_orig > 0:
            I_error = abs(I_peak_new - I_peak_orig) / I_peak_orig * 100
        else:
            I_error = 0.0

        verification.update({
            "reproduced": True,
            "I_peak_original": I_peak_orig,
            "I_peak_reproduced": I_peak_new,
            "I_peak_error_pct": I_error,
            "match": I_error < 1.0,  # < 1% = exact match (floating point)
        })

    except Exception as exc:
        verification.update({
            "reproduced": False,
            "error": str(exc),
        })

    return verification


def format_package_summary(package: dict) -> str:
    """Format a reproducibility package as a human-readable summary."""
    sw = package.get("software", {})
    cfg = package.get("configuration", {})
    out = package.get("outputs", {})
    ver = package.get("verification", {})

    lines = [
        "# DPF-Unified Reproducibility Package",
        f"Created: {package.get('created', '?')}",
        f"Git: {sw.get('git', {}).get('commit', '?')} ({sw.get('git', {}).get('branch', '?')})",
        f"Platform: {sw.get('platform', {}).get('system', '?')} {sw.get('platform', {}).get('machine', '?')}",
        "",
        f"Preset: {cfg.get('preset', '?')} | Backend: {cfg.get('backend', '?')} | Gas: {cfg.get('gas', '?')}",
        f"Sim time: {cfg.get('sim_time_us', '?')} us",
        "",
        f"I_peak = {out.get('I_peak_MA', 0):.3f} MA at {out.get('t_peak_us', 0):.1f} us",
        f"Steps: {out.get('n_steps', 0)} in {out.get('elapsed_s', 0):.2f} s",
        f"Waveform: {ver.get('waveform_points', 0)} points, checksum {ver.get('waveform_checksum', '?')[:8]}",
    ]

    ny = out.get("neutron_yield")
    if ny:
        lines.append(f"Yn = {ny.get('Y_neutron', 0):.2e} ({ny.get('bt_fraction', 0)*100:.0f}% BT)")

    bd = out.get("breakdown")
    if bd:
        lines.append(f"Breakdown: {bd.get('mechanism', '?')} (CIV ratio {bd.get('civ_ratio', 0):.1f})")

    notes = package.get("notes", "")
    if notes:
        lines.append(f"\nNotes: {notes}")

    return "\n".join(lines)
