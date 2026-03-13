"""Run PF-1000 Lee model simulation (snowplow + circuit) and output I(t).

This bypasses the full MHD engine and runs the 0D Lee model directly,
which is what produces the I(t) waveform shape (axial rundown + radial
implosion + pinch current dip).

Usage:
    python3 scripts/run_pf1000.py [--preset pf1000] [--output results/pf1000_It.npz]
"""
from __future__ import annotations

import argparse
import sys
import time as wall_time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from dpf.circuit.rlc_solver import RLCSolver
from dpf.core.bases import CouplingState
from dpf.fluid.snowplow import SnowplowModel
from dpf.presets import get_preset, get_preset_names


def run_lee_model(preset_name: str, sim_time: float | None = None) -> dict[str, np.ndarray]:
    """Run snowplow + circuit Lee model for a given preset.

    Returns dict with time_s, current_A, voltage_V, L_plasma_H, sheath_z_m,
    shock_r_m, phase arrays.
    """
    preset = get_preset(preset_name)
    circuit_cfg = preset["circuit"]
    snowplow_cfg = preset.get("snowplow", {})
    rho0 = preset["rho0"]
    t_end = sim_time or preset["sim_time"]

    circuit = RLCSolver(
        C=circuit_cfg["C"],
        V0=circuit_cfg["V0"],
        L0=circuit_cfg["L0"],
        R0=circuit_cfg.get("R0", 0.0),
        anode_radius=circuit_cfg["anode_radius"],
        cathode_radius=circuit_cfg["cathode_radius"],
        crowbar_enabled=circuit_cfg.get("crowbar_enabled", False),
        crowbar_mode=circuit_cfg.get("crowbar_mode", "voltage_zero"),
        crowbar_resistance=circuit_cfg.get("crowbar_resistance", 0.0),
    )

    has_snowplow = bool(snowplow_cfg)
    if has_snowplow:
        snowplow = SnowplowModel(
            anode_radius=circuit_cfg["anode_radius"],
            cathode_radius=circuit_cfg["cathode_radius"],
            fill_density=rho0,
            anode_length=snowplow_cfg.get("anode_length", 0.16),
            mass_fraction=snowplow_cfg.get("mass_fraction", 0.15),
            fill_pressure_Pa=snowplow_cfg.get("fill_pressure_Pa", 400.0),
            current_fraction=snowplow_cfg.get("current_fraction", 0.7),
            radial_mass_fraction=snowplow_cfg.get("radial_mass_fraction", None),
            pinch_column_fraction=snowplow_cfg.get("pinch_column_fraction", 1.0),
        )
    else:
        snowplow = None

    # LC period for timestep estimation
    L_total = circuit_cfg["L0"] + 1e-9  # small offset
    T_LC = 2 * np.pi * np.sqrt(L_total * circuit_cfg["C"])
    dt = T_LC / 5000  # ~5000 steps per LC period

    times = []
    currents = []
    voltages = []
    L_plasmas = []
    sheath_zs = []
    shock_rs = []
    phases = []

    t = 0.0
    step = 0
    coupling = CouplingState()

    t0_wall = wall_time.perf_counter()

    while t < t_end:
        # Snowplow provides L_plasma and dL/dt
        if snowplow is not None:
            sp_result = snowplow.step(dt, circuit.current)
            coupling.Lp = sp_result["L_plasma"]
            coupling.dL_dt = sp_result["dL_dt"]
            sheath_zs.append(sp_result["z_sheath"])
            shock_rs.append(sp_result["r_shock"])
            phases.append(sp_result["phase"])
        else:
            sheath_zs.append(0.0)
            shock_rs.append(0.0)
            phases.append("none")

        # Circuit step
        coupling = circuit.step(coupling, back_emf=0.0, dt=dt)

        t += dt
        step += 1

        times.append(t)
        currents.append(circuit.current)
        voltages.append(circuit.voltage)
        L_plasmas.append(coupling.Lp)

        if step % 5000 == 0:
            elapsed = wall_time.perf_counter() - t0_wall
            I_kA = circuit.current / 1e3
            phase = phases[-1] if phases else "?"
            print(f"  t={t*1e6:.2f} us, I={I_kA:.1f} kA, "
                  f"phase={phase}, wall={elapsed:.1f}s")

    elapsed = wall_time.perf_counter() - t0_wall
    print(f"Done: {step} steps in {elapsed:.1f}s ({step/elapsed:.0f} steps/s)")

    t_arr = np.array(times)
    I_arr = np.array(currents)
    I_peak = np.max(np.abs(I_arr))
    t_peak = t_arr[np.argmax(np.abs(I_arr))]
    print(f"I_peak = {I_peak/1e6:.3f} MA at t = {t_peak*1e6:.3f} us")

    if snowplow is not None:
        print(f"Snowplow final phase: {phases[-1]}")
        if snowplow.rundown_complete:
            print(f"Rundown complete at z = {snowplow.sheath_position*1e3:.1f} mm")
        if snowplow.pinch_complete:
            print(f"Pinch at r_min = {snowplow.shock_radius*1e3:.2f} mm")

    return {
        "time_s": t_arr,
        "current_A": I_arr,
        "voltage_V": np.array(voltages),
        "L_plasma_H": np.array(L_plasmas),
        "sheath_z_m": np.array(sheath_zs),
        "shock_r_m": np.array(shock_rs),
        "preset": preset_name,
        "V0_kV": circuit_cfg["V0"] / 1e3,
        "C_mF": circuit_cfg["C"] * 1e3,
        "E_kJ": 0.5 * circuit_cfg["C"] * circuit_cfg["V0"]**2 / 1e3,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="DPF Lee model I(t) simulation")
    parser.add_argument("--preset", type=str, default="pf1000",
                        choices=get_preset_names(),
                        help="Device preset name")
    parser.add_argument("--output", type=str, default=None,
                        help="Output .npz file (default: results/<preset>_It.npz)")
    parser.add_argument("--sim-time", type=float, default=None,
                        help="Override simulation time [s]")
    args = parser.parse_args()

    out_path = Path(args.output) if args.output else Path(f"results/{args.preset}_It.npz")

    preset = get_preset(args.preset)
    E_kJ = 0.5 * preset["circuit"]["C"] * preset["circuit"]["V0"]**2 / 1e3
    print(f"{args.preset}: V0={preset['circuit']['V0']/1e3:.0f} kV, "
          f"C={preset['circuit']['C']*1e3:.3f} mF, E={E_kJ:.0f} kJ")

    result = run_lee_model(args.preset, sim_time=args.sim_time)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(out_path), **result)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
