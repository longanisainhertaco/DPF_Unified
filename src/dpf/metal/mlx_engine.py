"""Pure-MLX DPF engine: circuit + snowplow + MHD with zero CPU sync.

Chains MLXCircuitSolver + MLXSnowplow + MLXMHDSolver into a single
discharge simulation with no numpy calls in the hot loop. Compatible
with mx.grad for differentiable calibration.

Usage:
    from dpf.metal.mlx_engine import run_mlx_discharge
    result = run_mlx_discharge(preset_name="pf1000", max_steps=5000)
"""
from __future__ import annotations

import time
from typing import Any

from dpf.metal.mlx_circuit import MLXCircuitSolver
from dpf.metal.mlx_snowplow import MLXSnowplow
from dpf.presets import get_preset


def run_mlx_discharge(
    preset_name: str = "pf1000",
    max_steps: int = 10000,
    fc: float | None = None,
    fm: float | None = None,
    V0_kV: float | None = None,
    pressure_torr: float | None = None,
    mode: str = "lee",
    grid_shape: tuple[int, int, int] = (32, 1, 64),
) -> dict[str, Any]:
    """Run a full DPF discharge through pure-MLX pipeline.

    Args:
        mode: "lee" for circuit+snowplow only (1ms/shot), or
              "mhd" for circuit+snowplow+MLX MHD solver (7min/shot).
        grid_shape: MHD grid resolution (only used in "mhd" mode).

    Returns dict with I_peak_MA, t_peak_us, n_steps, elapsed_s, and
    time-series arrays (I_MA, t_us, Lp_nH, phases).
    """
    preset = get_preset(preset_name)
    cc = preset["circuit"]
    sp_cfg = preset.get("snowplow", {})

    _fc = fc if fc is not None else sp_cfg.get("current_fraction", 0.7)
    _fm = fm if fm is not None else sp_cfg.get("mass_fraction", 0.15)
    _V0 = (V0_kV * 1e3) if V0_kV is not None else cc["V0"]
    _p_Pa = (pressure_torr * 133.322) if pressure_torr is not None else sp_cfg.get("fill_pressure_Pa", 400.0)

    # Fill density from ideal gas law: rho = p * m_D2 / (kB * T)
    _kB = 1.380649e-23
    _m_D2 = 6.69e-27  # D2 molecular mass
    rho0 = _p_Pa * _m_D2 / (_kB * 300.0)

    sim_time = preset.get("sim_time", 10e-6)

    # Build circuit
    circuit = MLXCircuitSolver(
        V0=_V0, C=cc["C"], L0=cc["L0"], R0=cc["R0"],
        crowbar_enabled=cc.get("crowbar_enabled", False),
        crowbar_mode=cc.get("crowbar_mode", "voltage_zero"),
        crowbar_time=cc.get("crowbar_time", 0.0),
        crowbar_resistance=cc.get("crowbar_resistance", 0.0),
    )

    # Build snowplow
    snowplow = MLXSnowplow(
        anode_radius=cc["anode_radius"],
        cathode_radius=cc["cathode_radius"],
        fill_density=rho0,
        anode_length=sp_cfg.get("anode_length", 0.16),
        mass_fraction=_fm,
        current_fraction=_fc,
        fill_pressure_Pa=_p_Pa,
        radial_mass_fraction=sp_cfg.get("radial_mass_fraction"),
        pinch_column_fraction=sp_cfg.get("pinch_column_fraction", 1.0),
    )

    # MHD solver (optional)
    mhd_solver = None
    mhd_state = None
    if mode == "mhd":
        import numpy as _np  # only for MHD state I/O, not in gradient path

        from dpf.metal.mlx_solver import MLXMHDSolver
        mhd_solver = MLXMHDSolver(
            grid_shape=grid_shape, dx=preset.get("dx", 1e-3),
            riemann_solver="hlls", reconstruction="plm",
            time_integrator="ssp_rk2", coordinates="cylindrical",
            r_inner=cc["anode_radius"],
            ion_mass=_m_D2 / 2.0,  # deuterium atom mass
        )
        nr, ny, nz = grid_shape
        mhd_state = {
            "rho": _np.full((nr, ny, nz), rho0, dtype=_np.float32),
            "velocity": _np.zeros((3, nr, ny, nz), dtype=_np.float32),
            "pressure": _np.full((nr, ny, nz), _p_Pa, dtype=_np.float32),
            "B": _np.zeros((3, nr, ny, nz), dtype=_np.float32),
            "Te": _np.full((nr, ny, nz), 300.0, dtype=_np.float32),
            "Ti": _np.full((nr, ny, nz), 300.0, dtype=_np.float32),
        }

    # Timestep from LC period
    import math
    L_total = cc["L0"] + 1e-9
    T_LC = 2.0 * math.pi * math.sqrt(L_total * cc["C"])
    dt = T_LC / 5000.0
    n_steps_max = min(max_steps, int(sim_time / dt) + 1)

    # Short-circuit current for divergence guard
    I_sc = _V0 / max(math.sqrt(cc["L0"] / cc["C"]), 1e-30)
    I_diverge = 10.0 * I_sc

    # Time series storage (Python lists — no numpy)
    times: list[float] = []
    currents: list[float] = []
    voltages: list[float] = []
    Lp_list: list[float] = []
    phases: list[str] = []

    t = 0.0
    t0_wall = time.perf_counter()
    I_peak = 0.0
    t_peak = 0.0

    for _step in range(n_steps_max):
        # Snowplow step
        sp_result = snowplow.step(dt, circuit.current)
        Lp = sp_result["L_plasma"]
        dLp_dt = sp_result["dL_dt"]
        R_plasma = sp_result.get("R_plasma", 0.0)

        # MHD step (if enabled)
        if mhd_solver is not None and mhd_state is not None:
            mhd_dt = mhd_solver._compute_dt(mhd_state)
            mhd_dt = min(mhd_dt, dt)
            mhd_state = mhd_solver.step(
                mhd_state, mhd_dt,
                current=circuit.current, voltage=circuit.voltage,
            )

        # Circuit step
        circuit.step(Lp=Lp, dLp_dt=dLp_dt, R_plasma=R_plasma, back_emf=0.0, dt=dt)
        t += dt

        # Record
        I_MA = circuit.current / 1e6
        times.append(t * 1e6)
        currents.append(I_MA)
        voltages.append(circuit.voltage / 1e3)
        Lp_list.append(Lp * 1e9)
        phases.append(sp_result["phase"])

        if abs(I_MA) > abs(I_peak):
            I_peak = I_MA
            t_peak = t * 1e6

        # Divergence guard
        if abs(circuit.current) > I_diverge:
            currents[-1] = currents[-2] if len(currents) > 1 else 0.0
            break

    elapsed = time.perf_counter() - t0_wall

    return {
        "preset": preset_name,
        "I_peak_MA": abs(I_peak),
        "t_peak_us": t_peak,
        "n_steps": len(times),
        "elapsed_s": round(elapsed, 3),
        "sim_time_us": sim_time * 1e6,
        "fc": _fc,
        "fm": _fm,
        "t_us": times,
        "I_MA": currents,
        "V_kV": voltages,
        "Lp_nH": Lp_list,
        "phases": phases,
    }
