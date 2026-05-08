"""Standalone PF-1000 MLX stepping probe.

This bypasses pytest/conftest so native MLX/Metal aborts can be separated from
test-harness subprocess or import-probe behavior.  It assumes the caller is
running in a Metal-visible process.
"""

from __future__ import annotations

import os
from types import MethodType

import numpy as np

import mlx.core as mlx

os.environ.setdefault("DPF_MLX_ASSUME_AVAILABLE", "1")

from dpf.config import SimulationConfig  # noqa: E402
from dpf.engine import SimulationEngine  # noqa: E402
from dpf.presets import get_preset  # noqa: E402


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
    return SimulationEngine(SimulationConfig(**preset))


def main() -> int:
    if mlx.default_device().type != mlx.gpu:
        raise SystemExit(f"Metal GPU not available: {mlx.default_device()}")

    target_steps = int(os.environ.get("DPF_MLX_PROBE_STEPS", "10"))
    target_us_raw = os.environ.get("DPF_MLX_PROBE_TARGET_US")
    target_s = float(target_us_raw) * 1e-6 if target_us_raw is not None else None
    print_interval = int(os.environ.get("DPF_MLX_PROBE_PRINT_INTERVAL", "50"))
    print_start = int(os.environ.get("DPF_MLX_PROBE_PRINT_START", str(target_steps + 1)))
    print_start_interval = int(os.environ.get("DPF_MLX_PROBE_PRINT_START_INTERVAL", "1"))
    clear_cache_interval = int(os.environ.get("DPF_MLX_PROBE_CLEAR_CACHE_INTERVAL", "0"))
    memory_telemetry = os.environ.get("DPF_MLX_PROBE_MEMORY", "0") == "1"
    if memory_telemetry and hasattr(mlx, "reset_peak_memory"):
        mlx.reset_peak_memory()

    engine = _make_pf1000_mlx_engine()
    engine._nan_check_stride = 1
    peak_current = 0.0
    peak_time = 0.0

    def _fail_on_nonfinite(self: SimulationEngine, label: str) -> int:
        for key, arr in self.state.items():
            if not isinstance(arr, np.ndarray):
                continue
            bad = ~np.isfinite(arr)
            count = int(np.sum(bad))
            if count:
                first = tuple(int(i) for i in np.argwhere(bad)[0])
                value = arr[first]
                raise AssertionError(
                    f"{label}: {count} non-finite value(s) in {key}; "
                    f"first={first}, value={value}, step={self.step_count}, "
                    f"t={self.time * 1e6:.6f} us"
                )
        return 0

    engine._sanitize_state = MethodType(_fail_on_nonfinite, engine)

    for step in range(target_steps):
        dt_fluid_before = float(engine.fluid._compute_dt(engine.state))
        dt_before = float(engine._compute_dt())
        result = engine.step()
        current_abs = abs(float(engine.circuit.current))
        if current_abs > peak_current:
            peak_current = current_abs
            peak_time = float(engine.time)
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
        should_print = (
            step == 0
            or (
                (step + 1) >= print_start
                and (step + 1 - print_start) % max(print_start_interval, 1) == 0
            )
            or (step + 1) % print_interval == 0
            or (target_s is not None and engine.time >= target_s)
            or result.finished
        )
        if should_print:
            memory_fields: list[str] = []
            if memory_telemetry:
                for attr, label in (
                    ("get_active_memory", "mlx_active_MB"),
                    ("get_cache_memory", "mlx_cache_MB"),
                    ("get_peak_memory", "mlx_peak_MB"),
                ):
                    getter = getattr(mlx, attr, None)
                    if getter is not None:
                        memory_fields.append(f"{label}={float(getter()) / 1e6:.3f}")
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
                *memory_fields,
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

    if target_s is not None and engine.time < target_s:
        print(
            "CAP_EXHAUSTED",
            f"steps={target_steps}",
            f"target_us={target_s * 1e6:.6f}",
            f"final_t_us={engine.time * 1e6:.6f}",
            flush=True,
        )
        return 2
    print(
        "PASSED",
        f"steps={engine.step_count}",
        f"final_t_us={engine.time * 1e6:.6f}",
        f"peak_I_MA={peak_current / 1e6:.6f}",
        f"peak_t_us={peak_time * 1e6:.6f}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
