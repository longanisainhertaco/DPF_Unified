"""Opt-in PF-1000 MLX endurance/regression stepping probe.

This file is intentionally narrow and is used for runtime/endurance evidence,
not for scientific acceptance. Set ``DPF_MLX_RUN_ENDURANCE=1`` to execute it.
Scientific PF-1000 gates stay in ``tests/test_mlx_pf1000.py`` as source-blocked
xfails until S1/S2 same-scope waveform evidence is accepted.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

_ENDURANCE_OPT_IN_ENV = "DPF_MLX_RUN_ENDURANCE"
_ENDURANCE_OPT_IN = os.environ.get(_ENDURANCE_OPT_IN_ENV) == "1"
_ENDURANCE_POLICY = {
    "lane": "endurance_regression",
    "scientific_status": "non_scientific",
    "source_status": "s1_s2_source_closure_blocked",
    "opt_in_env": _ENDURANCE_OPT_IN_ENV,
}

if not _ENDURANCE_OPT_IN:
    pytestmark = pytest.mark.skip(
        reason=f"PF-1000 endurance probe is opt-in; set {_ENDURANCE_OPT_IN_ENV}=1"
    )
    mlx = None
    HAS_MLX = False
    _METAL_GPU_AVAILABLE = False
else:
    mlx = pytest.importorskip("mlx.core", reason="MLX not installed")

    from dpf.config import SimulationConfig
    from dpf.engine import SimulationEngine
    from dpf.metal.mlx_device import HAS_MLX
    from dpf.presets import get_preset

    _METAL_GPU_AVAILABLE = HAS_MLX and (mlx.default_device().type == mlx.gpu)


def _make_pf1000_mlx_engine() -> SimulationEngine:
    preset_name = os.environ.get("DPF_MLX_PROBE_PRESET", "pf1000")
    preset = get_preset(preset_name)
    preset["fluid"] = {
        "backend": "mlx",
        "riemann_solver": "hll",
        "reconstruction": "plm",
        "time_integrator": "ssp_rk2",
        "precision": "float32",
        "use_ct": False,
    }
    preset["grid_shape"] = [32, 1, 64]
    preset["sim_time"] = 12e-6
    preset["radiation"] = {"bremsstrahlung_enabled": False, "fld_enabled": False}
    preset["collision"] = {"enabled": False}
    preset["nan_check_stride"] = 1
    preset["fail_fast_on_nonfinite"] = True
    return SimulationEngine(SimulationConfig(**preset))


def _memory_fields() -> list[str]:
    if mlx is None:
        return ["mlx_memory_unavailable=1"]
    fields: list[str] = []
    for attr, label in (
        ("get_active_memory", "mlx_active_MB"),
        ("get_cache_memory", "mlx_cache_MB"),
        ("get_peak_memory", "mlx_peak_MB"),
    ):
        getter = getattr(mlx, attr, None)
        if getter is not None:
            fields.append(f"{label}={float(getter()) / 1e6:.3f}")
    return fields or ["mlx_memory_unavailable=1"]


@pytest.mark.skipif(not _METAL_GPU_AVAILABLE, reason="Metal GPU not available")
@pytest.mark.slow
def test_pf1000_mlx_probe_steps() -> None:
    target_steps = int(os.environ.get("DPF_MLX_PROBE_STEPS", "10"))
    target_us_raw = os.environ.get("DPF_MLX_PROBE_TARGET_US")
    target_s = float(target_us_raw) * 1e-6 if target_us_raw is not None else None
    print_interval = int(os.environ.get("DPF_MLX_PROBE_PRINT_INTERVAL", "50"))
    print_start = int(os.environ.get("DPF_MLX_PROBE_PRINT_START", str(target_steps + 1)))
    print_start_interval = int(os.environ.get("DPF_MLX_PROBE_PRINT_START_INTERVAL", "1"))
    clear_cache_interval = int(os.environ.get("DPF_MLX_PROBE_CLEAR_CACHE_INTERVAL", "0"))
    memory_telemetry = True
    if memory_telemetry and hasattr(mlx, "reset_peak_memory"):
        mlx.reset_peak_memory()
    engine = _make_pf1000_mlx_engine()
    print(
        "policy",
        *[f"{key}={value}" for key, value in _ENDURANCE_POLICY.items()],
        f"target_us={(target_s * 1e6) if target_s is not None else -1.0:.6f}",
        f"cap_steps={target_steps}",
        "memory_telemetry=1",
        "nan_check_stride=1",
        "fail_fast_on_nonfinite=1",
        flush=True,
    )

    for step in range(target_steps):
        dt_fluid_before = float(engine.fluid._compute_dt(engine.state))
        dt_before = float(engine._compute_dt())
        result = engine.step()
        state = engine.state
        rho = np.asarray(state["rho"], dtype=np.float64)
        pressure = np.asarray(state["pressure"], dtype=np.float64)
        B = np.asarray(state["B"], dtype=np.float64)
        min_rho = float(np.nanmin(rho))
        min_p = float(np.nanmin(pressure))
        max_b = float(np.nanmax(np.abs(B)))
        snowplow = getattr(engine, "snowplow", None)
        coupling = getattr(engine, "_coupling", None)
        circuit_state = getattr(engine.circuit, "state", None)
        phase = getattr(snowplow, "phase", "none")
        z_cm = float(getattr(snowplow, "sheath_position", 0.0)) * 100.0
        r_cm = float(getattr(snowplow, "shock_radius", 0.0)) * 100.0
        lp_nh = float(getattr(coupling, "Lp", 0.0)) * 1.0e9
        dldt_nh_per_us = float(getattr(coupling, "dL_dt", 0.0) or 0.0) * 1.0e3
        r_plasma_mohm = float(getattr(coupling, "R_plasma", 0.0)) * 1.0e3
        sheath_p_pa = float(getattr(engine, "_last_sheath_pressure", 0.0))
        crowbar_fired = bool(getattr(circuit_state, "crowbar_fired", False))
        crowbar_time_us = (
            float(getattr(circuit_state, "crowbar_fire_time", -1.0)) * 1.0e6
            if crowbar_fired
            else -1.0
        )
        if (
            step == 0
            or (
                (step + 1) >= print_start
                and (step + 1 - print_start) % max(print_start_interval, 1) == 0
            )
            or (step + 1) % print_interval == 0
            or (target_s is not None and engine.time >= target_s)
            or result.finished
        ):
            print(
                "probe",
                f"step={step + 1}",
                f"t_us={engine.time * 1e6:.6f}",
                f"dt_fluid_ps={dt_fluid_before * 1e12:.6f}",
                f"dt_before_ps={dt_before * 1e12:.6f}",
                f"I_MA={engine.circuit.current / 1e6:.6f}",
                f"V_kV={engine.circuit.voltage / 1e3:.6f}",
                f"crowbar={int(crowbar_fired)}",
                f"crowbar_t_us={crowbar_time_us:.6f}",
                f"phase={phase}",
                f"z_cm={z_cm:.6f}",
                f"r_cm={r_cm:.6f}",
                f"Lp_nH={lp_nh:.6f}",
                f"dLdt_nH_per_us={dldt_nh_per_us:.6f}",
                f"Rplasma_mohm={r_plasma_mohm:.6f}",
                f"sheath_p_Pa={sheath_p_pa:.6e}",
                f"min_rho={min_rho:.6e}",
                f"min_p={min_p:.6e}",
                f"max_B={max_b:.6e}",
                *_memory_fields(),
                flush=True,
            )
        assert np.all(np.isfinite(rho)), (
            f"rho became non-finite at step {step + 1}, "
            f"t={engine.time * 1e6:.6f} us, dt_before={dt_before:.6e} s"
        )
        assert np.all(np.isfinite(pressure)), (
            f"pressure became non-finite at step {step + 1}, "
            f"t={engine.time * 1e6:.6f} us, dt_before={dt_before:.6e} s"
        )
        assert np.all(np.isfinite(B)), (
            f"B became non-finite at step {step + 1}, "
            f"t={engine.time * 1e6:.6f} us, dt_before={dt_before:.6e} s"
        )
        if clear_cache_interval > 0 and (step + 1) % clear_cache_interval == 0:
            clear_cache = getattr(mlx, "clear_cache", None)
            if clear_cache is not None:
                clear_cache()
        if result.finished or (target_s is not None and engine.time >= target_s):
            break
    cap_exhausted = target_s is not None and engine.time < target_s
    result_status = "CAP_EXHAUSTED" if cap_exhausted else "PASSED"
    print(
        "endurance_result",
        f"status={result_status}",
        f"scientific_status={_ENDURANCE_POLICY['scientific_status']}",
        f"source_status={_ENDURANCE_POLICY['source_status']}",
        f"target_us={(target_s * 1e6) if target_s is not None else -1.0:.6f}",
        f"cap_steps={target_steps}",
        f"steps={engine.step_count}",
        f"final_t_us={engine.time * 1e6:.6f}",
        f"cap_exhausted={int(cap_exhausted)}",
        *_memory_fields(),
        flush=True,
    )
    if target_s is not None:
        assert engine.time >= target_s, (
            f"target time {target_s * 1e6:.6f} us not reached within "
            f"{target_steps} steps; final t={engine.time * 1e6:.6f} us"
        )
