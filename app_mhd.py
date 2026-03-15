"""MHD backend connector for DPF web UI.

Wraps MetalMHDSolver and Python MHD engine to produce the same output
format as the Lee model runner, enabling backend switching in the UI.
"""
from __future__ import annotations

import logging
import time as wall_time
from typing import Any

import numpy as np

from app_engine import GAS_SPECIES, kB

logger = logging.getLogger(__name__)

BACKENDS = {
    "lee": "Lee Model (0D snowplow + circuit, <1s)",
    "metal_plm": "Metal GPU — PLM+HLL+SSP-RK2 (fast, ~6.5/10 fidelity)",
    "metal_weno5": "Metal GPU — WENO5+HLLD+SSP-RK3 (high fidelity, ~8.7/10)",
    "athena": "Athena++ — PPM+HLLD (reference C++ engine, ~9.0/10 fidelity)",
    "python": "Python MHD — NumPy/Numba (full physics, moderate speed)",
}

BACKEND_CONFIGS = {
    "metal_plm": {
        "reconstruction": "plm", "riemann_solver": "hll",
        "time_integrator": "ssp_rk2", "precision": "float32",
    },
    "metal_weno5": {
        "reconstruction": "weno5", "riemann_solver": "hlld",
        "time_integrator": "ssp_rk3", "precision": "float64",
    },
}

MHD_GRID_PRESETS = {
    "coarse": (16, 16, 32),
    "medium": (32, 32, 64),
    "fine": (64, 64, 128),
}


def run_mhd_simulation(
    backend: str,
    grid_preset: str,
    preset_name: str,
    sim_time_us: float,
    gas_key: str = "D2",
    V0_kV: float | None = None,
    pressure_torr: float | None = None,
    C_uF: float | None = None,
    L0_nH: float | None = None,
    R0_mOhm: float | None = None,
    anode_r_mm: float | None = None,
    cathode_r_mm: float | None = None,
    anode_len_mm: float | None = None,
    progress_fn=None,
) -> dict[str, Any]:
    """Run MHD simulation and return data in the same format as Lee model."""
    from dpf.presets import _PRESETS, get_preset

    preset = get_preset(preset_name)
    cc = preset["circuit"]
    sc = preset.get("snowplow", {})
    gas = GAS_SPECIES.get(gas_key, GAS_SPECIES["D2"])

    if V0_kV is not None and V0_kV > 0:
        cc["V0"] = V0_kV * 1e3
    if C_uF is not None and C_uF > 0:
        cc["C"] = C_uF * 1e-6
    if L0_nH is not None and L0_nH > 0:
        cc["L0"] = L0_nH * 1e-9
    if R0_mOhm is not None and R0_mOhm > 0:
        cc["R0"] = R0_mOhm * 1e-3
    if anode_r_mm is not None and anode_r_mm > 0:
        cc["anode_radius"] = anode_r_mm * 1e-3
    if cathode_r_mm is not None and cathode_r_mm > 0:
        cc["cathode_radius"] = cathode_r_mm * 1e-3

    a = cc["anode_radius"]
    b = cc["cathode_radius"]
    L_anode = sc.get("anode_length", 0.16)
    if anode_len_mm is not None and anode_len_mm > 0:
        L_anode = anode_len_mm * 1e-3

    p_pa = sc.get("fill_pressure_Pa", 400.0)
    if pressure_torr is not None and pressure_torr > 0:
        p_pa = pressure_torr * 133.322
    rho0 = p_pa * gas["m_mol"] / (kB * 300.0)

    grid_shape = MHD_GRID_PRESETS.get(grid_preset, (32, 32, 64))
    nr, ny, nz = grid_shape
    dr = (b - a) / nr
    dz = L_anode / nz

    t_end = sim_time_us * 1e-6
    meta = _PRESETS.get(preset_name, {}).get("_meta", {})
    E_bank = 0.5 * cc["C"] * cc["V0"] ** 2

    t0_wall = wall_time.perf_counter()

    if backend.startswith("metal"):
        result = _run_metal(
            backend, grid_shape, dr, dz, gas, rho0, p_pa,
            cc, t_end, a, b, L_anode, progress_fn,
        )
    elif backend == "athena":
        from pathlib import Path
        _athena_bin = Path(__file__).resolve().parent / "external" / "athena" / "bin" / "athena_cylindrical"
        if not _athena_bin.exists():
            import gradio as gr
            gr.Warning(
                "Athena++ binary not found — falling back to Python MHD engine. "
                "Build Athena++ with `cd external/athena && make -j8` for full fidelity."
            )
            logger.warning(
                "Athena++ binary not found at %s — falling back to Python engine", _athena_bin
            )
            backend = "python"
            result = _run_python_mhd(
                grid_shape, dr, dz, gas, rho0, p_pa,
                cc, t_end, a, b, L_anode, progress_fn,
            )
            result["backend"] = "python (fallback from athena)"
        else:
            result = _run_athena(
                grid_shape, dr, dz, gas, rho0, p_pa,
                cc, sc, t_end, a, b, L_anode, progress_fn,
            )
    else:
        result = _run_python_mhd(
            grid_shape, dr, dz, gas, rho0, p_pa,
            cc, t_end, a, b, L_anode, progress_fn,
        )

    elapsed = wall_time.perf_counter() - t0_wall

    result.update({
        "E_bank_kJ": E_bank / 1e3,
        "T_LC_us": 2 * np.pi * np.sqrt(cc["L0"] * cc["C"]) * 1e6,
        "elapsed_s": elapsed,
        "device": meta.get("device", preset_name),
        "circuit": cc, "snowplow_cfg": sc,
        "gas": gas, "gas_key": gas_key,
        "rho0": rho0,
        "backend": backend,
        "grid_shape": grid_shape,
    })

    # Compute neutron yield from MHD state for deuterium fills
    if gas.get("A") == 2 and gas.get("Z") == 1:
        final_state = result.get("final_state")
        if final_state is not None:
            try:
                from dpf.diagnostics.neutron_yield import neutron_yield_rate

                rho = final_state["rho"]
                ion_mass = gas["m_mol"]
                n_D = rho / ion_mass
                Ti = final_state.get("Ti", final_state["pressure"] * ion_mass / (2.0 * rho * kB))
                nr, ny_g, nz = rho.shape
                cell_vol = dr * (b - a) / nr * dz  # approximate cell volume
                _, total_rate = neutron_yield_rate(n_D, Ti, cell_vol)

                # Estimate confinement time from MHD evolution
                t_arr = result.get("t_us", np.array([]))
                if len(t_arr) > 1:
                    tau_pinch = (t_arr[-1] - t_arr[0]) * 1e-6  # total sim time in seconds
                else:
                    tau_pinch = t_end

                Y_thermo = total_rate * tau_pinch

                # Beam-target from circuit (if current available)
                I_arr = result.get("I_MA", np.array([]))
                Y_bt = 0.0
                if len(I_arr) > 0:
                    try:
                        from dpf.diagnostics.beam_target import beam_target_yield_rate
                        I_peak_A = float(np.max(np.abs(I_arr))) * 1e6
                        n_target = float(np.max(n_D))
                        L_pinch = L_anode * 0.3  # EMPIRICAL: ~30% of anode length
                        # V_pinch from dL/dt * I
                        L_arr = result.get("L_p_nH", np.array([]))
                        if len(L_arr) > 1 and len(t_arr) > 1:
                            dLdt = np.gradient(L_arr * 1e-9, t_arr * 1e-6)
                            V_pinch = float(np.max(np.abs(I_arr * 1e6 * dLdt)))
                        else:
                            V_pinch = 0.0
                        if V_pinch > 1e3:
                            bt_rate = beam_target_yield_rate(
                                I_peak_A, V_pinch, n_target, L_pinch, f_beam=0.14,
                            )
                            Y_bt = bt_rate * tau_pinch
                    except ImportError:
                        pass

                Y_total = Y_thermo + Y_bt
                if Y_total > 0:
                    result["neutron_yield"] = {
                        "Y_thermonuclear": float(Y_thermo),
                        "Y_beam_target": float(Y_bt),
                        "Y_neutron": float(Y_total),
                        "bt_fraction": float(Y_bt / Y_total) if Y_total > 0 else 0.0,
                        "V_pinch_kV": float(V_pinch / 1e3) if "V_pinch" in dir() else 0.0,
                        "tau_ns": float(tau_pinch * 1e9),
                    }
            except (ImportError, Exception) as exc:
                logger.debug("Neutron yield computation skipped: %s", exc)

    # Bennett equilibrium diagnostic — check if pinch achieves pressure balance
    final_state = result.get("final_state")
    I_arr = result.get("I_MA", np.array([]))
    if final_state is not None and len(I_arr) > 0:
        I_peak_A = float(np.max(np.abs(I_arr))) * 1e6
        rho_final = final_state["rho"]
        p_final = final_state["pressure"]
        B_final = final_state["B"]
        # Magnetic pressure at peak B location
        B2 = np.sum(B_final**2, axis=0)
        mu_0 = 4 * np.pi * 1e-7
        p_mag_max = float(np.max(B2)) / (2 * mu_0)
        p_kin_max = float(np.max(p_final))
        beta_pinch = p_kin_max / p_mag_max if p_mag_max > 0 else float("inf")
        # Bennett temperature from I^2 = (8*pi*N_L*kB*(Te+Ti))/mu_0
        # N_L = n * pi * r_p^2 — use peak density and anode radius as proxy
        n_peak = float(np.max(rho_final)) / gas["m_mol"]
        N_L = n_peak * np.pi * a**2
        if N_L > 0:
            T_bennett_K = mu_0 * I_peak_A**2 / (8 * np.pi * N_L * 2 * kB)
            T_bennett_keV = T_bennett_K * kB / (1000 * 1.602e-19)
        else:
            T_bennett_keV = 0.0
        result["bennett"] = {
            "beta_pinch": float(beta_pinch),
            "p_mag_max_Pa": float(p_mag_max),
            "p_kin_max_Pa": float(p_kin_max),
            "T_bennett_keV": float(T_bennett_keV),
            "source": "Bennett 1934, Russell 2025",
        }

    # Instability timing diagnostic (Goyon 2025, Eq. 4)
    # tau_m0 = 31.0 * R_imp^2 * sqrt(P_fill) / (CR * I_imp)
    # where R_imp = cathode radius [cm], CR = convergence ratio, I_imp = implosion current [MA]
    I_arr = result.get("I_MA", np.array([]))
    if len(I_arr) > 0:
        I_imp_MA = float(np.max(np.abs(I_arr)))
        R_imp_cm = b * 100  # cathode radius in cm
        P_fill_Torr = p_pa / 133.322
        CR = b / a if a > 0 else 10.0  # convergence ratio = cathode/anode radius
        if I_imp_MA > 0:
            tau_m0_ns = 31.0 * R_imp_cm**2 * np.sqrt(P_fill_Torr) / (CR * I_imp_MA)
            result["instability"] = {
                "tau_m0_ns": float(tau_m0_ns),
                "convergence_ratio": float(CR),
                "I_imp_MA": float(I_imp_MA),
                "source": "Goyon et al. 2025, Eq. 4",
            }

    # Synthetic interferometry diagnostic (Challenge 15)
    final_state = result.get("final_state")
    if final_state is not None:
        try:
            from dpf.diagnostics.interferometry import abel_transform, fringe_shift
            rho_final = final_state["rho"]
            ion_mass = gas["m_mol"]
            nz_mid = rho_final.shape[-1] // 2
            if rho_final.ndim == 3:
                ne_mid = rho_final[:, rho_final.shape[1] // 2, nz_mid] / ion_mass
            else:
                ne_mid = rho_final[:, nz_mid] / ion_mass
            r_arr = np.linspace(a + dr * 0.5, b - dr * 0.5, len(ne_mid))
            N_L = abel_transform(ne_mid, r_arr)
            fringes = fringe_shift(N_L)
            result["interferometry"] = {
                "r_mm": (r_arr * 1e3).tolist(),
                "ne_midplane_m3": ne_mid.tolist(),
                "line_integrated_m2": N_L.tolist(),
                "fringes_HeNe": fringes.tolist(),
                "peak_fringes": float(np.max(np.abs(fringes))),
            }
        except (ImportError, Exception):
            pass

    # Plasmoid detection (Challenge 14)
    if final_state is not None:
        try:
            from dpf.diagnostics.instability import detect_plasmoids
            plasmoid_result = detect_plasmoids(
                final_state["B"], final_state["rho"], dr, dz,
            )
            if plasmoid_result["n_plasmoids"] > 0 or plasmoid_result["magnetic_energy_J"] > 0:
                result["plasmoids"] = plasmoid_result
        except (ImportError, Exception):
            pass

    return result


def _run_metal(
    backend: str,
    grid_shape: tuple[int, int, int],
    dr: float, dz: float,
    gas: dict, rho0: float, p_pa: float,
    cc: dict, t_end: float,
    a: float, b: float, L_anode: float,
    progress_fn=None,
) -> dict[str, Any]:
    """Run Metal GPU MHD solver."""
    import torch

    from dpf.circuit.rlc_solver import RLCSolver
    from dpf.core.bases import CouplingState
    from dpf.metal.metal_solver import MetalMHDSolver

    cfg = BACKEND_CONFIGS.get(backend, BACKEND_CONFIGS["metal_plm"])

    use_mps = cfg["precision"] != "float64" and torch.backends.mps.is_available()
    device = "mps" if use_mps else "cpu"

    solver = MetalMHDSolver(
        grid_shape=grid_shape, dx=dr, dz=dz,
        gamma=gas.get("gamma", 5 / 3),
        cfl=0.3, device=device,
        use_ct=False,
        coordinates="cylindrical",
        ion_mass=gas["m_mol"],
        **cfg,
    )

    circuit = RLCSolver(
        C=cc["C"], V0=cc["V0"], L0=cc["L0"],
        R0=cc.get("R0", 0.0),
        anode_radius=a, cathode_radius=b,
        crowbar_enabled=cc.get("crowbar_enabled", False),
        crowbar_mode=cc.get("crowbar_mode", "voltage_zero"),
        crowbar_time=cc.get("crowbar_time", 0.0),
        crowbar_resistance=cc.get("crowbar_resistance", 0.0),
    )

    nr, ny, nz = grid_shape
    state = {
        "rho": np.full((nr, ny, nz), rho0),
        "velocity": np.zeros((3, nr, ny, nz)),
        "pressure": np.full((nr, ny, nz), p_pa),
        "B": np.zeros((3, nr, ny, nz)),
        "Te": np.full((nr, ny, nz), 300.0),
        "Ti": np.full((nr, ny, nz), 300.0),
        "psi": np.zeros((nr, ny, nz)),
    }
    # B starts at zero — no current flows until the circuit fires.
    # The circuit solver drives B_theta growth via dI/dt coupling.

    coupling = CouplingState()
    t = 0.0
    times, currents, voltages, L_plasmas = [], [], [], []
    E_cap, E_ind, E_res = [], [], []
    rho_max_arr, T_max_arr, B_max_arr = [], [], []
    mhd_snapshots = []
    snap_interval = max(1, int(t_end / (80 * solver.cfl * min(dr, dz) / 1e4)))

    step = 0
    while t < t_end:
        dt_mhd = solver.compute_dt(state)
        dt = min(dt_mhd, t_end - t)
        if dt <= 0:
            break

        state = solver.step(
            state, dt, current=circuit.current, voltage=circuit.voltage,
            anode_radius=a, cathode_radius=b, apply_electrode_bc=True,
        )
        coupling = circuit.step(coupling, back_emf=0.0, dt=dt)
        t += dt
        step += 1

        times.append(t * 1e6)
        currents.append(circuit.current / 1e6)
        voltages.append(circuit.voltage / 1e3)
        L_plasmas.append(coupling.Lp * 1e9)
        E_cap.append(circuit.state.energy_cap / 1e3)
        E_ind.append(circuit.state.energy_ind / 1e3)
        E_res.append(circuit.state.energy_res / 1e3)
        rho_max_arr.append(float(np.max(state["rho"])))
        T_max_arr.append(float(np.max(state.get("Te", state["pressure"] / state["rho"]))))
        B_max_arr.append(float(np.max(np.sqrt(np.sum(state["B"] ** 2, axis=0)))))

        if step % snap_interval == 0:
            mhd_snapshots.append({
                "t_us": t * 1e6,
                "rho_mid": state["rho"][:, ny // 2, :].copy(),
                "B_mid": state["B"][:, :, ny // 2, :].copy(),
                "P_mid": state["pressure"][:, ny // 2, :].copy(),
            })

        if progress_fn and step % 50 == 0:
            progress_fn(min(t / t_end, 1.0), desc=f"MHD t={t*1e6:.1f}us, step={step}")

    t_arr = np.array(times)
    I_arr = np.array(currents)
    I_peak_idx = int(np.argmax(np.abs(I_arr)))

    return {
        "t_us": t_arr, "I_MA": I_arr, "V_kV": np.array(voltages),
        "L_p_nH": np.array(L_plasmas),
        "E_cap_kJ": np.array(E_cap), "E_ind_kJ": np.array(E_ind),
        "E_res_kJ": np.array(E_res),
        "rho_max": np.array(rho_max_arr),
        "T_max": np.array(T_max_arr),
        "B_max": np.array(B_max_arr),
        "mhd_snapshots": mhd_snapshots,
        "final_state": state,
        "I_peak": float(np.abs(I_arr[I_peak_idx])),
        "t_peak": float(t_arr[I_peak_idx]),
        "n_steps": step,
        "has_snowplow": False,
        "has_mhd": True,
        "phases": ["mhd"] * len(times),
        "z_mm": np.zeros(len(times)),
        "r_mm": np.zeros(len(times)),
        "dip_pct": 0.0, "I_pre_dip": 0.0, "I_dip": 0.0, "t_dip": 0.0,
        "scaling": None, "crowbar_t": None,
        "snowplow_obj": None, "dt_ns": 0,
    }


_ATHENA_STEP_TIMEOUT_S = 30


def _run_athena(
    grid_shape: tuple[int, int, int],
    dr: float, dz: float,
    gas: dict, rho0: float, p_pa: float,
    cc: dict, sc: dict, t_end: float,
    a: float, b: float, L_anode: float,
    progress_fn=None,
) -> dict[str, Any]:
    """Run Athena++ C++ MHD solver via subprocess mode."""
    import concurrent.futures
    from pathlib import Path

    from dpf.athena_wrapper import AthenaPPSolver
    from dpf.config import (
        CircuitConfig,
        DiagnosticsConfig,
        FluidConfig,
        GeometryConfig,
        SimulationConfig,
        SnowplowConfig,
    )

    nr, ny, nz = grid_shape

    circuit_cfg = CircuitConfig(
        C=cc["C"], V0=cc["V0"], L0=cc["L0"],
        R0=cc.get("R0", 0.0),
        anode_radius=a, cathode_radius=b,
        crowbar_enabled=cc.get("crowbar_enabled", False),
        crowbar_mode=cc.get("crowbar_mode", "voltage_zero"),
        crowbar_time=cc.get("crowbar_time", 0.0),
        crowbar_resistance=cc.get("crowbar_resistance", 0.0),
    )

    sim_cfg = SimulationConfig(
        grid_shape=[nr, 1, nz],
        dx=dr,
        sim_time=t_end,
        rho0=rho0,
        T0=300.0,
        ion_mass=gas["m_mol"],
        circuit=circuit_cfg,
        geometry=GeometryConfig(type="cylindrical", dz=dz),
        fluid=FluidConfig(
            backend="athena",
            reconstruction="plm",
            riemann_solver="hlld",
            gamma=gas.get("gamma", 5 / 3),
            cfl=0.3,
            time_integrator="ssp_rk2",
        ),
        snowplow=SnowplowConfig(
            enabled=True,
            fill_pressure_Pa=p_pa,
            anode_length=L_anode,
            current_fraction=sc.get("current_fraction", 0.7),
            mass_fraction=sc.get("mass_fraction", 0.15),
        ),
        diagnostics=DiagnosticsConfig(hdf5_filename=":memory:"),
    )

    athena_bin = str(Path(__file__).resolve().parent / "external" / "athena" / "bin" / "athena_cylindrical")
    solver = AthenaPPSolver(sim_cfg, athena_binary=athena_bin, use_subprocess=True)

    state = solver.initial_state()
    t = 0.0
    times, currents, voltages, L_plasmas = [], [], [], []
    E_cap, E_ind, E_res = [], [], []
    rho_max_arr, T_max_arr, B_max_arr = [], [], []
    mhd_snapshots = []

    from dpf.circuit.rlc_solver import RLCSolver
    from dpf.core.bases import CouplingState

    circuit = RLCSolver(
        C=cc["C"], V0=cc["V0"], L0=cc["L0"],
        R0=cc.get("R0", 0.0),
        anode_radius=a, cathode_radius=b,
        crowbar_enabled=cc.get("crowbar_enabled", False),
        crowbar_mode=cc.get("crowbar_mode", "voltage_zero"),
        crowbar_time=cc.get("crowbar_time", 0.0),
        crowbar_resistance=cc.get("crowbar_resistance", 0.0),
    )
    coupling = CouplingState()

    step = 0
    while t < t_end:
        dt = solver._compute_dt(state)
        dt = min(dt, t_end - t)
        if dt <= 0:
            break

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as _pool:
            _fut = _pool.submit(
                solver.step, state, dt,
                current=circuit.current, voltage=circuit.voltage,
            )
            try:
                state = _fut.result(timeout=_ATHENA_STEP_TIMEOUT_S)
            except concurrent.futures.TimeoutError:
                logger.error(
                    "Athena++ step timed out after %ds at t=%.3e s — aborting",
                    _ATHENA_STEP_TIMEOUT_S, t,
                )
                break
        coupling = circuit.step(coupling, back_emf=0.0, dt=dt)
        t += dt
        step += 1

        times.append(t * 1e6)
        currents.append(circuit.current / 1e6)
        voltages.append(circuit.voltage / 1e3)
        L_plasmas.append(coupling.Lp * 1e9)
        E_cap.append(circuit.state.energy_cap / 1e3)
        E_ind.append(circuit.state.energy_ind / 1e3)
        E_res.append(circuit.state.energy_res / 1e3)
        rho_max_arr.append(float(np.max(state["rho"])))
        T_max_arr.append(float(np.max(state.get("Te", state["pressure"] / state["rho"]))))
        B_max_arr.append(float(np.max(np.sqrt(np.sum(state["B"] ** 2, axis=0)))))

        if step % 80 == 0:
            mhd_snapshots.append({
                "t_us": t * 1e6,
                "rho_mid": state["rho"][:, 0, :].copy(),
                "P_mid": state["pressure"][:, 0, :].copy(),
            })

        if progress_fn and step % 20 == 0:
            progress_fn(min(t / t_end, 1.0), desc=f"Athena++ t={t*1e6:.1f}us, step={step}")

    t_arr = np.array(times)
    I_arr = np.array(currents)
    I_peak_idx = int(np.argmax(np.abs(I_arr))) if len(I_arr) > 0 else 0

    return {
        "t_us": t_arr, "I_MA": I_arr, "V_kV": np.array(voltages),
        "L_p_nH": np.array(L_plasmas),
        "E_cap_kJ": np.array(E_cap), "E_ind_kJ": np.array(E_ind),
        "E_res_kJ": np.array(E_res),
        "rho_max": np.array(rho_max_arr),
        "T_max": np.array(T_max_arr),
        "B_max": np.array(B_max_arr),
        "mhd_snapshots": mhd_snapshots,
        "final_state": state,
        "I_peak": float(np.abs(I_arr[I_peak_idx])) if len(I_arr) > 0 else 0,
        "t_peak": float(t_arr[I_peak_idx]) if len(t_arr) > 0 else 0,
        "n_steps": step,
        "has_snowplow": False,
        "has_mhd": True,
        "phases": ["mhd"] * len(times),
        "z_mm": np.zeros(len(times)),
        "r_mm": np.zeros(len(times)),
        "dip_pct": 0.0, "I_pre_dip": 0.0, "I_dip": 0.0, "t_dip": 0.0,
        "scaling": None, "crowbar_t": None,
        "snowplow_obj": None, "dt_ns": 0,
    }


def _run_python_mhd(
    grid_shape: tuple[int, int, int],
    dr: float, dz: float,
    gas: dict, rho0: float, p_pa: float,
    cc: dict, t_end: float,
    a: float, b: float, L_anode: float,
    progress_fn=None,
) -> dict[str, Any]:
    """Run Python NumPy MHD solver (CylindricalMHDSolver)."""
    from dpf.circuit.rlc_solver import RLCSolver
    from dpf.core.bases import CouplingState
    from dpf.fluid.cylindrical_mhd import CylindricalMHDSolver

    nr, ny, nz = grid_shape

    solver = CylindricalMHDSolver(
        nr=nr, nz=nz, dr=dr, dz=dz,
        gamma=gas.get("gamma", 5 / 3),
        cfl=0.3,
        enable_hall=False,  # Hall MHD causes overflow on coarse grids; enable for fine grids only
        enable_resistive=True,
        ion_mass=gas["m_mol"],
    )

    circuit = RLCSolver(
        C=cc["C"], V0=cc["V0"], L0=cc["L0"],
        R0=cc.get("R0", 0.0),
        anode_radius=a, cathode_radius=b,
        crowbar_enabled=cc.get("crowbar_enabled", False),
        crowbar_mode=cc.get("crowbar_mode", "voltage_zero"),
        crowbar_time=cc.get("crowbar_time", 0.0),
        crowbar_resistance=cc.get("crowbar_resistance", 0.0),
    )

    # Stochastic IC for shot-to-shot reproducibility studies (Challenge 9)
    # Random density perturbation with controllable seed for reproducibility
    rng = np.random.default_rng()  # Different every shot — models real variability
    delta_rho = 0.01  # 1% perturbation amplitude
    noise = rng.normal(0, delta_rho, size=(nr, 1, nz))
    # Add both structured (m=0) and random components
    z_arr = np.linspace(0, L_anode, nz)
    m0_pert = 0.005 * np.sin(4 * np.pi * z_arr / L_anode)
    rho_3d = rho0 * (1.0 + noise + m0_pert[np.newaxis, np.newaxis, :])
    rho_3d = np.maximum(rho_3d, rho0 * 0.01)  # floor at 1% of fill

    state = {
        "rho": rho_3d,
        "velocity": np.zeros((3, nr, 1, nz)),
        "pressure": np.full((nr, 1, nz), p_pa),
        "B": np.zeros((3, nr, 1, nz)),
        "Te": np.full((nr, 1, nz), 300.0),
        "Ti": np.full((nr, 1, nz), 300.0),
        "psi": np.zeros((nr, 1, nz)),
    }

    coupling = CouplingState()
    t = 0.0
    times, currents, voltages, L_plasmas = [], [], [], []
    E_cap, E_ind, E_res = [], [], []
    rho_max_arr, T_max_arr, B_max_arr = [], [], []
    mhd_snapshots = []

    step = 0
    while t < t_end:
        dt_mhd = solver.compute_dt(state)
        dt = min(dt_mhd, t_end - t)
        if dt <= 0:
            break

        state = solver.step(
            state, dt, current=circuit.current, voltage=circuit.voltage,
        )

        # Radiation cooling (Frontier D): bremsstrahlung + line radiation
        if "Te" in state:
            try:
                from dpf.radiation.bremsstrahlung import apply_bremsstrahlung_losses
                rho_safe = np.where(state["rho"] > 0, state["rho"], 1.0)
                ne = rho_safe / gas["m_mol"]
                Z_eff = gas.get("Z", 1)
                state["Te"], _ = apply_bremsstrahlung_losses(
                    state["Te"], ne, dt, Z=Z_eff,
                )
                # Line + recombination radiation for high-Z fills (Ne, Ar, Kr, Xe)
                if gas.get("Z", 1) > 1:
                    from dpf.radiation.line_radiation import apply_line_radiation_losses
                    state["Te"], _ = apply_line_radiation_losses(
                        state["Te"], ne, dt, Z_eff=0,  # brems already applied above
                        n_imp_frac=0.0, Z_imp=gas.get("Z", 10),
                    )
            except ImportError:
                pass

        # Back-EMF from MHD plasma inductance change (Frontier A: full coupling)
        mhd_coupling = solver.coupling_interface()
        back_emf = 0.0
        if mhd_coupling.dL_dt is not None and abs(circuit.current) > 1.0:
            back_emf = mhd_coupling.dL_dt * circuit.current

        coupling = circuit.step(coupling, back_emf=back_emf, dt=dt)
        t += dt
        step += 1

        times.append(t * 1e6)
        currents.append(circuit.current / 1e6)
        voltages.append(circuit.voltage / 1e3)
        Lp_mhd = mhd_coupling.Lp if mhd_coupling.Lp > 0 else coupling.Lp
        L_plasmas.append(Lp_mhd * 1e9)
        E_cap.append(circuit.state.energy_cap / 1e3)
        E_ind.append(circuit.state.energy_ind / 1e3)
        E_res.append(circuit.state.energy_res / 1e3)
        rho_max_arr.append(float(np.max(state["rho"])))
        rho_safe = np.where(state["rho"] > 0, state["rho"], 1.0)
        T_max_arr.append(float(np.max(state.get("Te", state["pressure"] / rho_safe))))
        B_max_arr.append(float(np.max(np.sqrt(np.sum(state["B"] ** 2, axis=0)))))

        if step % 100 == 0:
            mhd_snapshots.append({
                "t_us": t * 1e6,
                "rho_mid": state["rho"][:, 0, :].copy(),
                "P_mid": state["pressure"][:, 0, :].copy(),
            })

        if progress_fn and step % 50 == 0:
            progress_fn(min(t / t_end, 1.0), desc=f"Python MHD t={t*1e6:.1f}us")

    t_arr = np.array(times)
    I_arr = np.array(currents)
    I_peak_idx = int(np.argmax(np.abs(I_arr))) if len(I_arr) > 0 else 0

    return {
        "t_us": t_arr, "I_MA": I_arr, "V_kV": np.array(voltages),
        "L_p_nH": np.array(L_plasmas),
        "E_cap_kJ": np.array(E_cap), "E_ind_kJ": np.array(E_ind),
        "E_res_kJ": np.array(E_res),
        "mhd_snapshots": mhd_snapshots,
        "final_state": state,
        "I_peak": float(np.abs(I_arr[I_peak_idx])) if len(I_arr) > 0 else 0,
        "t_peak": float(t_arr[I_peak_idx]) if len(t_arr) > 0 else 0,
        "n_steps": step,
        "has_snowplow": False,
        "has_mhd": True,
        "phases": ["mhd"] * len(times),
        "z_mm": np.zeros(len(times)),
        "r_mm": np.zeros(len(times)),
        "rho_max": np.array(rho_max_arr),
        "T_max": np.array(T_max_arr),
        "B_max": np.array(B_max_arr),
        "dip_pct": 0.0, "I_pre_dip": 0.0, "I_dip": 0.0, "t_dip": 0.0,
        "scaling": None, "crowbar_t": None,
        "snowplow_obj": None, "dt_ns": 0,
    }


def create_mhd_fields_fig(d: dict[str, Any]) -> Any:
    """Create 2D field plots from MHD snapshots with physical coordinates."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    snapshots = d.get("mhd_snapshots", [])
    if not snapshots:
        fig = go.Figure()
        fig.add_annotation(
            text="No MHD snapshots available", x=0.5, y=0.5,
            showarrow=False, font=dict(size=16, color="#aaa"),
        )
        fig.update_layout(height=400, template="plotly_dark")
        return fig

    snap = snapshots[-1]
    rho = snap["rho_mid"]
    P = snap["P_mid"]

    cc = d.get("circuit", {})
    sc = d.get("snowplow_cfg", {})
    a_m = cc.get("anode_radius", 0.01)
    b_m = cc.get("cathode_radius", 0.02)
    L_m = sc.get("anode_length", 0.16)
    nr, nz = rho.shape

    r_mm = np.linspace(a_m * 1e3, b_m * 1e3, nr)
    z_mm = np.linspace(0, L_m * 1e3, nz)

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=[
            f"Density (t={snap['t_us']:.1f} us)",
            f"Pressure (t={snap['t_us']:.1f} us)",
        ],
        horizontal_spacing=0.15,
    )

    fig.add_trace(go.Heatmap(
        z=rho, x=z_mm, y=r_mm, colorscale="Viridis", name="rho",
        colorbar=dict(title="kg/m<sup>3</sup>", x=0.42, len=0.9),
    ), row=1, col=1)

    fig.add_trace(go.Heatmap(
        z=P, x=z_mm, y=r_mm, colorscale="Inferno", name="P",
        colorbar=dict(title="Pa", x=1.0, len=0.9),
    ), row=1, col=2)

    fig.update_layout(
        height=400, template="plotly_dark",
        margin=dict(l=60, r=20, t=40, b=40),
    )
    fig.update_xaxes(title_text="z [mm]")
    fig.update_yaxes(title_text="r [mm]")
    return fig
