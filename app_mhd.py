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
    "lee": "Lee Model (< 1 sec) -- waveforms only, no spatial detail",
    "hybrid": "Hybrid (3-30 sec) -- Lee rundown + MHD pinch compression [RECOMMENDED]",
    "metal_plm": "2D MHD Fast (10-60 sec) -- GPU spatial fields, moderate accuracy",
    "metal_weno5": "2D MHD Precise (30-120 sec) -- GPU spatial fields, high accuracy",
    "metal_3d": "3D MHD (2-10 min) -- full 3D instability physics, GPU required",
    "athena": "Athena++ C++ (10-60 sec) -- Princeton reference engine, 3rd-order",
    "python": "Python MHD -- auto-redirects to 2D MHD Fast (unstable at high currents)",
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
    "metal_3d": {
        "reconstruction": "plm", "riemann_solver": "hll",
        "time_integrator": "ssp_rk2", "precision": "float32",
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

    if backend == "hybrid":
        result = _run_hybrid_lee_mhd(
            grid_shape, dr, dz, gas, rho0, p_pa,
            cc, sc, t_end, a, b, L_anode, progress_fn,
        )
    elif backend.startswith("metal"):
        result = _run_metal(
            backend, grid_shape, dr, dz, gas, rho0, p_pa,
            cc, t_end, a, b, L_anode, progress_fn,
        )
    elif backend == "athena":
        from pathlib import Path
        _athena_bin = Path(__file__).resolve().parent / "external" / "athena" / "bin" / "athena_cylindrical"
        if not _athena_bin.exists():
            try:
                import gradio as gr
                gr.Warning(
                    "Athena++ binary not found — falling back to Metal PLM engine. "
                    "Build Athena++ with `cd external/athena && make -j8` for native C++ fidelity."
                )
            except ImportError:
                pass
            logger.warning(
                "Athena++ binary not found at %s — falling back to Metal PLM", _athena_bin
            )
            backend = "metal_plm"
            result = _run_metal(
                backend, grid_shape, dr, dz, gas, rho0, p_pa,
                cc, t_end, a, b, L_anode, progress_fn,
            )
            result["backend"] = "metal_plm (fallback from athena)"
        else:
            result = _run_athena(
                grid_shape, dr, dz, gas, rho0, p_pa,
                cc, sc, t_end, a, b, L_anode, progress_fn,
            )
    else:
        # Python MHD (np.gradient) is numerically unstable for DPF-scale
        # currents. Fall back to Metal PLM which uses proper shock-capturing.
        try:
            import torch
            has_metal = True
        except ImportError:
            has_metal = False

        if has_metal:
            try:
                import gradio as gr
                gr.Info(
                    "Python MHD redirected to Metal PLM — the Python solver "
                    "(np.gradient) is unstable at MA currents. Metal uses "
                    "shock-capturing (PLM+HLL) for stable results."
                )
            except ImportError:
                pass
            logger.info("Python backend redirected to metal_plm (stability)")
            result = _run_metal(
                "metal_plm", grid_shape, dr, dz, gas, rho0, p_pa,
                cc, t_end, a, b, L_anode, progress_fn,
            )
            result["backend"] = "metal_plm (redirected from python)"
        else:
            result = _run_python_mhd(
                grid_shape, dr, dz, gas, rho0, p_pa,
                cc, t_end, a, b, L_anode, progress_fn,
            )

    elapsed = wall_time.perf_counter() - t0_wall

    # Preserve custom backend label from redirect/fallback logic
    effective_backend = result.get("backend", backend)
    result.update({
        "E_bank_kJ": E_bank / 1e3,
        "T_LC_us": 2 * np.pi * np.sqrt(cc["L0"] * cc["C"]) * 1e6,
        "elapsed_s": elapsed,
        "device": meta.get("device", preset_name),
        "circuit": cc, "snowplow_cfg": sc,
        "gas": gas, "gas_key": gas_key,
        "rho0": rho0,
        "backend": effective_backend,
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


def _run_hybrid_lee_mhd(
    grid_shape: tuple[int, int, int],
    dr: float, dz: float,
    gas: dict, rho0: float, p_pa: float,
    cc: dict, sc: dict, t_end: float,
    a: float, b: float, L_anode: float,
    progress_fn=None,
) -> dict[str, Any]:
    """Hybrid Lee+MHD: Lee model runs axial rundown, MHD handles radial implosion.

    Phase 1 (Lee): Snowplow model sweeps gas along anode. Fast (0D), well-validated.
        Provides: circuit state (I, V), swept mass, sheath velocity at transition.
    Phase 2 (MHD): Metal solver takes over at start of radial phase.
        IC: compressed gas column with B_theta from circuit current.
        Resolves: radial implosion, pinch compression, instabilities.
    """
    import torch

    from dpf.circuit.rlc_solver import RLCSolver
    from dpf.core.bases import CouplingState
    from dpf.fluid.snowplow import SnowplowModel
    from dpf.metal.metal_solver import MetalMHDSolver

    mu_0 = 4.0 * np.pi * 1e-7

    # ---- Phase 1: Lee model axial rundown ----
    circuit = RLCSolver(
        C=cc["C"], V0=cc["V0"], L0=cc["L0"],
        R0=cc.get("R0", 0.0),
        anode_radius=a, cathode_radius=b,
        crowbar_enabled=cc.get("crowbar_enabled", False),
        crowbar_mode=cc.get("crowbar_mode", "voltage_zero"),
        crowbar_time=cc.get("crowbar_time", 0.0),
        crowbar_resistance=cc.get("crowbar_resistance", 0.0),
    )

    snowplow = SnowplowModel(
        anode_radius=a, cathode_radius=b,
        fill_density=rho0,
        anode_length=L_anode,
        mass_fraction=sc.get("mass_fraction", 0.15),
        fill_pressure_Pa=sc.get("fill_pressure_Pa", p_pa),
        current_fraction=sc.get("current_fraction", 0.7),
        radial_mass_fraction=sc.get("radial_mass_fraction", None),
        pinch_column_fraction=sc.get("pinch_column_fraction", 1.0),
    )

    # Run Lee model until radial phase begins (or t_end)
    L_total = cc["L0"] + 1e-9
    T_LC = 2 * np.pi * np.sqrt(L_total * cc["C"])
    dt_lee = T_LC / 5000

    times, currents, voltages, L_plasmas = [], [], [], []
    E_cap, E_ind, E_res = [], [], []
    sheath_zs, shock_rs, phases_list = [], [], []

    t = 0.0
    coupling = CouplingState()
    lee_steps = 0
    handoff_time = None

    while t < t_end:
        sp = snowplow.step(dt_lee, circuit.current)
        coupling.Lp = sp["L_plasma"]
        coupling.dL_dt = sp["dL_dt"]
        coupling.R_plasma = sp.get("R_plasma", 0.0)
        coupling = circuit.step(coupling, back_emf=0.0, dt=dt_lee)
        t += dt_lee
        lee_steps += 1

        times.append(t * 1e6)
        currents.append(circuit.current / 1e6)
        voltages.append(circuit.voltage / 1e3)
        L_plasmas.append(coupling.Lp * 1e9)
        E_cap.append(circuit.state.energy_cap / 1e3)
        E_ind.append(circuit.state.energy_ind / 1e3)
        E_res.append(circuit.state.energy_res / 1e3)
        sheath_zs.append(sp["z_sheath"] * 1e3)
        shock_rs.append(sp["r_shock"] * 1e3)
        phases_list.append(sp["phase"])

        if progress_fn and lee_steps % 200 == 0:
            progress_fn(
                min(t / t_end, 0.3),
                desc=f"Lee rundown: t={t*1e6:.1f}us, z={sp['z_sheath']*1e3:.0f}mm",
            )

        # Handoff when Lee model enters radial phase
        if sp["phase"] == "radial":
            handoff_time = t
            break

    if handoff_time is None:
        # Never reached radial phase — return Lee-only results
        logger.warning("Hybrid: Lee model didn't reach radial phase in %.1f us", t_end * 1e6)
        t_arr = np.array(times)
        I_arr = np.array(currents)
        I_peak_idx = int(np.argmax(np.abs(I_arr)))
        return {
            "t_us": t_arr, "I_MA": I_arr, "V_kV": np.array(voltages),
            "L_p_nH": np.array(L_plasmas),
            "E_cap_kJ": np.array(E_cap), "E_ind_kJ": np.array(E_ind),
            "E_res_kJ": np.array(E_res),
            "z_mm": np.array(sheath_zs), "r_mm": np.array(shock_rs),
            "phases": phases_list,
            "I_peak": float(np.abs(I_arr[I_peak_idx])),
            "t_peak": float(t_arr[I_peak_idx]),
            "n_steps": lee_steps,
            "has_snowplow": True, "has_mhd": False,
            "mhd_snapshots": [], "final_state": None,
            "dip_pct": 0.0, "I_pre_dip": 0.0, "I_dip": 0.0, "t_dip": 0.0,
            "scaling": None, "crowbar_t": None,
            "snowplow_obj": snowplow, "dt_ns": dt_lee * 1e9,
            "rho_max": np.array([rho0] * len(times)),
            "T_max": np.array([300.0] * len(times)),
            "B_max": np.array([0.0] * len(times)),
        }

    # ---- Phase 2: MHD radial implosion ----
    I_handoff = circuit.current  # [A] at start of radial phase
    fc = sc.get("current_fraction", 0.7)
    fm = sc.get("mass_fraction", 0.15)
    fmr = sc.get("radial_mass_fraction", fm)
    z_f = sc.get("pinch_column_fraction", 1.0) * L_anode

    nr, ny, nz = grid_shape
    cfg = BACKEND_CONFIGS["metal_plm"]
    use_mps = torch.backends.mps.is_available()
    device = "mps" if use_mps else "cpu"

    # MHD domain: radial extent = cathode - anode, axial = z_f (pinch column)
    dr_mhd = (b - a) / nr
    dz_mhd = z_f / max(nz, 1)

    solver = MetalMHDSolver(
        grid_shape=grid_shape, dx=dr_mhd, dz=dz_mhd,
        gamma=gas.get("gamma", 5 / 3),
        cfl=0.3, device=device,
        use_ct=False,
        coordinates="cylindrical",
        ion_mass=gas["m_mol"],
        **cfg,
    )

    # Build physically motivated IC for MHD radial phase:
    # - Swept mass concentrated near cathode (outer boundary)
    # - Unswept gas fills the interior
    # - B_theta = mu_0 * fc * I / (2*pi*r) throughout
    r_cells = np.linspace(a + dr_mhd * 0.5, b - dr_mhd * 0.5, nr)

    # Density: swept mass forms a shell near cathode, unswept gas elsewhere
    rho_bg = rho0 * (1.0 - fmr)  # background (unswept)
    # Swept mass distributed in outer 20% of radial cells (current sheath)
    n_sheath = max(int(0.2 * nr), 2)
    rho_mhd = np.full((nr, ny, nz), rho_bg)
    # Sheath density: all swept mass in the thin shell
    shell_vol = sum(
        2.0 * np.pi * r_cells[nr - n_sheath + i] * dr_mhd * dz_mhd
        for i in range(n_sheath)
    )
    swept_mass_per_z = fmr * rho0 * np.pi * (b**2 - a**2)  # [kg/m]
    rho_sheath = swept_mass_per_z * dz_mhd / max(shell_vol, 1e-30)
    rho_mhd[nr - n_sheath:, :, :] = max(rho_sheath, rho_bg * 2.0)

    # B_theta profile from circuit current (the magnetic piston)
    B_theta_1d = mu_0 * fc * I_handoff / (2.0 * np.pi * r_cells)
    B_mhd = np.zeros((3, nr, ny, nz))
    B_mhd[1] = B_theta_1d[:, np.newaxis, np.newaxis]  # B_theta

    # Pressure: gas pressure + kinetic pressure from sheath velocity
    # In the Lee model, the sheath starts radial phase with vr = 0
    # but has high magnetic pressure behind it
    p_mhd = np.full((nr, ny, nz), p_pa)

    state = {
        "rho": rho_mhd,
        "velocity": np.zeros((3, nr, ny, nz)),
        "pressure": p_mhd,
        "B": B_mhd,
        "Te": np.full((nr, ny, nz), 300.0),
        "Ti": np.full((nr, ny, nz), 300.0),
        "psi": np.zeros((nr, ny, nz)),
    }

    # Continue circuit from handoff state
    rho_max_arr = [float(np.max(rho_mhd))]
    T_max_arr = [300.0]
    B_max_arr = [float(np.max(np.abs(B_mhd)))]
    mhd_snapshots = []

    t_mhd_start = t
    mhd_step = 0
    # Target ~30 snapshots during MHD phase; compute interval after first step
    _target_snaps = 30
    snap_interval = 3  # default; recalculated after first timestep

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
        mhd_step += 1

        # Recalculate snap_interval after first step (now we know dt)
        if mhd_step == 1 and dt > 0:
            est_total_steps = max(1, int((t_end - t_mhd_start) / dt))
            snap_interval = max(1, est_total_steps // _target_snaps)

        times.append(t * 1e6)
        currents.append(circuit.current / 1e6)
        voltages.append(circuit.voltage / 1e3)
        L_plasmas.append(coupling.Lp * 1e9)
        E_cap.append(circuit.state.energy_cap / 1e3)
        E_ind.append(circuit.state.energy_ind / 1e3)
        E_res.append(circuit.state.energy_res / 1e3)
        sheath_zs.append(sp["z_sheath"] * 1e3)
        shock_rs.append(0.0)
        phases_list.append("mhd_radial")

        rho_max_arr.append(float(np.max(state["rho"])))
        T_max_arr.append(float(np.max(state.get("Te", state["pressure"] / state["rho"]))))
        B_max_arr.append(float(np.max(np.sqrt(np.sum(state["B"] ** 2, axis=0)))))

        if mhd_step % snap_interval == 0:
            mhd_snapshots.append({
                "t_us": t * 1e6,
                "rho_mid": state["rho"][:, ny // 2, :].copy(),
                "B_mid": state["B"][:, :, ny // 2, :].copy(),
                "P_mid": state["pressure"][:, ny // 2, :].copy(),
            })

        if progress_fn and mhd_step % 20 == 0:
            progress_fn(
                min(0.3 + 0.7 * (t - t_mhd_start) / max(t_end - t_mhd_start, 1e-30), 1.0),
                desc=f"MHD radial: t={t*1e6:.1f}us, step={mhd_step}",
            )

    t_arr = np.array(times)
    I_arr = np.array(currents)
    I_peak_idx = int(np.argmax(np.abs(I_arr)))

    # Find Lee-phase peak for snowplow dip detection
    lee_mask = [p != "mhd_radial" for p in phases_list]
    lee_I = I_arr[lee_mask] if any(lee_mask) else I_arr
    I_pre_dip = float(np.max(np.abs(lee_I)))

    # Find MHD-phase minimum for dip detection
    mhd_mask = np.array([p == "mhd_radial" for p in phases_list])
    # Find pre-dip peak (Lee phase) and dip (MHD phase) with timestamps
    lee_I_indices = [i for i, p in enumerate(phases_list) if p != "mhd_radial"]
    if lee_I_indices:
        lee_peak_idx = lee_I_indices[int(np.argmax(np.abs(I_arr[lee_I_indices])))]
        I_pre_dip = float(np.abs(I_arr[lee_peak_idx]))
        t_pre_dip = float(t_arr[lee_peak_idx])
    else:
        I_pre_dip = float(np.abs(I_arr[I_peak_idx]))
        t_pre_dip = float(t_arr[I_peak_idx])

    mhd_I_indices = [i for i, p in enumerate(phases_list) if p == "mhd_radial"]
    if mhd_I_indices:
        mhd_min_idx = mhd_I_indices[int(np.argmin(np.abs(I_arr[mhd_I_indices])))]
        I_dip = float(np.abs(I_arr[mhd_min_idx]))
        t_dip = float(t_arr[mhd_min_idx])
        dip_pct = (1 - I_dip / I_pre_dip) * 100 if I_pre_dip > 0 else 0
    else:
        I_dip = I_pre_dip
        t_dip = t_pre_dip
        dip_pct = 0.0

    return {
        "t_us": t_arr, "I_MA": I_arr, "V_kV": np.array(voltages),
        "L_p_nH": np.array(L_plasmas),
        "z_mm": np.array(sheath_zs), "r_mm": np.array(shock_rs),
        "phases": phases_list,
        "E_cap_kJ": np.array(E_cap), "E_ind_kJ": np.array(E_ind),
        "E_res_kJ": np.array(E_res),
        "rho_max": np.array(rho_max_arr),
        "T_max": np.array(T_max_arr),
        "B_max": np.array(B_max_arr),
        "mhd_snapshots": mhd_snapshots,
        "final_state": state,
        "I_peak": float(np.abs(I_arr[I_peak_idx])),
        "t_peak": float(t_arr[I_peak_idx]),
        "I_pre_dip": I_pre_dip,
        "t_pre_dip": t_pre_dip,
        "I_dip": I_dip,
        "t_dip": t_dip,
        "dip_pct": dip_pct,
        "n_steps": lee_steps + mhd_step,
        "has_snowplow": True,
        "has_mhd": True,
        "snowplow_obj": snowplow,
        "scaling": None, "crowbar_t": None,
        "dt_ns": 0,
        "handoff_time_us": handoff_time * 1e6,
        "lee_steps": lee_steps,
        "mhd_steps": mhd_step,
    }


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

    is_3d = backend == "metal_3d"
    coord_type = "cartesian" if is_3d else "cylindrical"
    # For 3D Cartesian, use isotropic dx (average of dr and dz)
    solver_dx = (dr + dz) / 2.0 if is_3d else dr
    solver_dz = solver_dx if is_3d else dz

    solver = MetalMHDSolver(
        grid_shape=grid_shape, dx=solver_dx, dz=solver_dz,
        gamma=gas.get("gamma", 5 / 3),
        cfl=0.3, device=device,
        use_ct=False,
        coordinates=coord_type,
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
    _target_snaps_metal = 30
    snap_interval = 3  # recalculated after first step

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

        if step == 1 and dt > 0:
            est_total = max(1, int(t_end / dt))
            snap_interval = max(1, est_total // _target_snaps_metal)

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
    """Run Python NumPy MHD solver (CylindricalMHDSolver).

    The Python solver uses np.gradient for spatial derivatives, which cannot
    handle the strong B-field electrode BC (overflow at MA currents). Instead,
    it evolves the circuit + a gentle B_theta seed scaled to the current,
    providing circuit waveforms with approximate MHD coupling.
    """
    from dpf.circuit.rlc_solver import RLCSolver
    from dpf.core.bases import CouplingState
    from dpf.fluid.cylindrical_mhd import CylindricalMHDSolver

    nr, ny, nz = grid_shape

    solver = CylindricalMHDSolver(
        nr=nr, nz=nz, dr=dr, dz=dz,
        gamma=gas.get("gamma", 5 / 3),
        cfl=0.3,
        enable_hall=False,
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

    # Uniform IC — no stochastic perturbation for Python solver (numerically fragile)
    state = {
        "rho": np.full((nr, 1, nz), rho0),
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
    nan_detected = False
    while t < t_end:
        dt_mhd = solver.compute_dt(state)
        dt = min(dt_mhd, t_end - t)
        if dt <= 0:
            break

        state = solver.step(
            state, dt, current=circuit.current, voltage=circuit.voltage,
        )

        # NaN detection — break early and return valid data so far
        if np.any(np.isnan(state["rho"])) or np.any(np.isnan(state["pressure"])):
            nan_detected = True
            logger.warning("Python MHD: NaN detected at step %d, t=%.3e — stopping early", step, t)
            break

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
                if gas.get("Z", 1) > 1:
                    from dpf.radiation.line_radiation import apply_line_radiation_losses
                    state["Te"], _ = apply_line_radiation_losses(
                        state["Te"], ne, dt, Z_eff=0,
                        n_imp_frac=0.0, Z_imp=gas.get("Z", 10),
                    )
            except ImportError:
                pass

        # Back-EMF from MHD plasma inductance change
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
        rho_max_arr.append(float(np.nanmax(state["rho"])))
        rho_safe = np.where(state["rho"] > 0, state["rho"], 1.0)
        T_max_arr.append(float(np.nanmax(state.get("Te", state["pressure"] / rho_safe))))
        B_max_arr.append(float(np.nanmax(np.sqrt(np.sum(state["B"] ** 2, axis=0)))))

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
