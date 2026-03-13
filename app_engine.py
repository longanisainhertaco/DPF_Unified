"""Simulation engine for DPF web UI — gas data, density calc, Lee model runner."""
from __future__ import annotations

import time as wall_time
from typing import Any

import numpy as np

from dpf.circuit.rlc_solver import RLCSolver
from dpf.core.bases import CouplingState
from dpf.fluid.snowplow import SnowplowModel, implosion_scaling
from dpf.presets import _PRESETS, get_preset

GAS_SPECIES: dict[str, dict[str, Any]] = {
    "D2": {
        "name": "Deuterium (D\u2082)", "formula": "D\u2082",
        "m_mol": 6.69e-27, "gamma": 5 / 3, "Z": 1, "A": 2,
        "diatomic": True,
    },
    "He": {
        "name": "Helium", "formula": "He",
        "m_mol": 6.646e-27, "gamma": 5 / 3, "Z": 2, "A": 4,
        "diatomic": False,
    },
    "Ne": {
        "name": "Neon", "formula": "Ne",
        "m_mol": 3.351e-26, "gamma": 5 / 3, "Z": 10, "A": 20,
        "diatomic": False,
    },
    "Ar": {
        "name": "Argon", "formula": "Ar",
        "m_mol": 6.634e-26, "gamma": 5 / 3, "Z": 18, "A": 40,
        "diatomic": False,
    },
    "Kr": {
        "name": "Krypton", "formula": "Kr",
        "m_mol": 1.392e-25, "gamma": 5 / 3, "Z": 36, "A": 84,
        "diatomic": False,
    },
    "Xe": {
        "name": "Xenon", "formula": "Xe",
        "m_mol": 2.180e-25, "gamma": 5 / 3, "Z": 54, "A": 131,
        "diatomic": False,
    },
    "N2": {
        "name": "Nitrogen (N\u2082)", "formula": "N\u2082",
        "m_mol": 4.652e-26, "gamma": 7 / 5, "Z": 7, "A": 14,
        "diatomic": True,
    },
}

kB = 1.381e-23
mu_0 = 4 * np.pi * 1e-7


def _bosch_hale_dd(T_keV: float) -> float:
    """Bosch-Hale D(d,n)3He reactivity <sigma*v> [m^3/s].

    Valid 0.2-100 keV. Bosch & Hale, Nucl. Fusion 32:611 (1992).
    """
    T_keV = np.clip(T_keV, 0.2, 100.0)
    a1, a2, a3, a4, a5 = 6.661, 643.41e-16, 15.136e-3, 75.189e-3, 4.6064e-3
    a6, a7 = 13.5e-3, -0.10675e-3
    theta = T_keV / (
        1 - (T_keV * (a2 + T_keV * (a4 + T_keV * a6)))
        / (1 + T_keV * (a3 + T_keV * (a5 + T_keV * a7)))
    )
    xi = (a1**2 / (4 * theta)) ** (1.0 / 3.0)
    return 5.43e-21 * theta * np.sqrt(xi / (a1**3 * T_keV)) * np.exp(-3 * xi)


def _beam_target_yield(
    I_pinch_A: float,
    n_pinch: float,
    z_pinch_m: float,
    r_pinch_m: float,
    tau_pinch_s: float,
    T_keV: float,
    V_pinch_volts: float = 0.0,
) -> float:
    """Beam-target D-D neutron yield from instability-accelerated ions.

    Uses two models depending on available data:
    1. If V_pinch (pinch voltage from circuit) is available, uses the
       proper beam_target_yield_rate() from dpf.diagnostics.beam_target
       which computes E_beam = e * V_pinch (Lee 2014).
    2. Fallback: estimates beam energy from Bennett temperature.

    The pinch voltage V_pinch = I * dL/dt during radial compression
    is the physically correct quantity — it's the inductive voltage
    that accelerates ions during m=0 instability disruption.
    """
    try:
        from dpf.diagnostics.beam_target import beam_target_yield_rate
        if V_pinch_volts > 1000.0:
            # Use proper V_pinch-driven model (Lee 2014)
            dY_dt = beam_target_yield_rate(
                I_pinch=I_pinch_A,
                V_pinch=V_pinch_volts,
                n_target=n_pinch,
                L_target=z_pinch_m,
                f_beam=0.14,  # Lee (2014) uses 0.1-0.3; 0.14 calibrated
            )
            return float(dY_dt * tau_pinch_s)
    except ImportError:
        pass

    # Fallback: estimate beam energy from Bennett temperature
    E_beam_keV = max(3.0 * T_keV, 20.0)
    E_beam_keV = min(E_beam_keV, 300.0)

    m_D = 3.344e-27
    E_beam_J = E_beam_keV * 1.602e-16
    v_beam = np.sqrt(2 * E_beam_J / m_D)

    E_cm = E_beam_keV / 2
    try:
        from dpf.diagnostics.beam_target import dd_cross_section
        sigma = dd_cross_section(E_cm)
    except ImportError:
        sigma = 0.0
    sigma_v_beam = sigma * v_beam

    V_vol = np.pi * r_pinch_m**2 * z_pinch_m
    f_beam = 0.02
    n_beam = f_beam * n_pinch
    Y_bt = n_beam * n_pinch * sigma_v_beam * V_vol * tau_pinch_s
    return float(Y_bt)


def _radiation_corrected_temperature(T_bennett_keV: float, n_pinch: float,
                                      tau_pinch_s: float, Z: int = 1) -> float:
    """Reduce Bennett temperature by bremsstrahlung radiation losses.

    The Bennett equilibrium temperature T_B = mu_0 * I^2 / (8*pi*N_l*2*k_B)
    is an upper bound. In practice, bremsstrahlung cooling is significant
    for high-density, high-temperature pinches (Haines 2011).

    The radiation-corrected temperature:
        T_eff = T_B / (1 + tau_pinch / tau_rad)

    where tau_rad = 3*n*T / (P_brem/V) is the cooling time and
    P_brem = 1.69e-32 * n_e^2 * Z^2 * sqrt(T_keV) [W/m^3].

    Only applied when T_bennett > 0.5 keV. Below that, bremsstrahlung
    cooling is weak and the Bennett estimate is reasonable. This prevents
    over-correction for small devices (NX2, UNU-ICTP) where thermonuclear
    yield is negligible anyway.
    """
    if T_bennett_keV <= 0 or n_pinch <= 0 or tau_pinch_s <= 0:
        return T_bennett_keV

    # Only apply correction for high-temperature pinches where radiation matters
    if T_bennett_keV < 0.5:
        return T_bennett_keV

    # Bremsstrahlung power density: P_brem = 1.69e-32 * ne^2 * Z^2 * sqrt(T_keV)
    P_brem_density = 1.69e-32 * n_pinch**2 * Z**2 * np.sqrt(T_bennett_keV)

    # Thermal energy density: e_th = 3*n*k_B*T (ions + electrons, 2 species)
    e_th_J = 3.0 * n_pinch * T_bennett_keV * 1.602e-16  # [J/m^3]

    if P_brem_density <= 0:
        return T_bennett_keV

    tau_rad = e_th_J / P_brem_density

    # Correction factor: reduces T when tau_pinch >> tau_rad
    correction = 1.0 / (1.0 + tau_pinch_s / tau_rad)

    # Floor at 20% of Bennett (minimum from shock heating / non-equilibrium)
    return max(T_bennett_keV * correction, 0.2 * T_bennett_keV)


def dd_neutron_yield(
    I_pinch_A: float,
    r_pinch_m: float,
    z_pinch_m: float,
    tau_pinch_s: float,
    rho0: float,
    ion_mass: float,
    cathode_r_m: float = 0.16,
    anode_r_m: float = 0.115,
    mass_fraction: float = 0.15,
    V_pinch_volts: float = 0.0,
) -> dict[str, float]:
    """Estimate D-D neutron yield: thermonuclear + beam-target components.

    Thermonuclear: Bennett temperature + Bosch-Hale reactivity.
    Beam-target: instability-accelerated ions (dominant for most DPF devices).

    The beam-target component typically dominates by 10-100x for MA-class
    devices (Lee & Saw 2008, Haines 2011). For sub-kJ devices, thermonuclear
    may dominate.

    Line density N_l accounts for cylindrical compression: swept gas
    from the annular electrode gap compressed into the pinch column.

    V_pinch_volts: Pinch voltage from circuit (I * dL/dt) [V]. If > 0,
    used for proper beam energy calculation instead of T_bennett estimate.
    """
    n_fill = rho0 / ion_mass
    N_l = mass_fraction * n_fill * np.pi * (cathode_r_m**2 - anode_r_m**2)
    n_pinch = N_l / (np.pi * r_pinch_m**2)

    T_K = mu_0 * I_pinch_A**2 / (8 * np.pi * N_l * 2 * kB)
    T_bennett_keV = T_K * kB / 1.602e-16

    # Apply radiation cooling correction (critical for high-current devices)
    T_eff_keV = _radiation_corrected_temperature(
        T_bennett_keV, n_pinch, tau_pinch_s,
    )

    sigma_v = _bosch_hale_dd(T_eff_keV)

    V_vol = np.pi * r_pinch_m**2 * z_pinch_m
    Y_thermo = 0.5 * n_pinch**2 * sigma_v * V_vol * tau_pinch_s

    Y_bt = _beam_target_yield(
        I_pinch_A, n_pinch, z_pinch_m, r_pinch_m, tau_pinch_s, T_eff_keV,
        V_pinch_volts=V_pinch_volts,
    )

    Y_total = Y_thermo + Y_bt

    return {
        "T_bennett_keV": float(T_bennett_keV),
        "T_eff_keV": float(T_eff_keV),
        "n_ion": float(n_fill),
        "n_pinch": float(n_pinch),
        "sigma_v": float(sigma_v),
        "V_pinch_cm3": float(V_vol * 1e6),
        "V_pinch_kV": float(V_pinch_volts / 1e3),
        "rad_cooling_factor": float(T_eff_keV / T_bennett_keV) if T_bennett_keV > 0 else 1.0,
        "Y_thermonuclear": float(Y_thermo),
        "Y_beam_target": float(Y_bt),
        "Y_neutron": float(Y_total),
        "bt_fraction": float(Y_bt / Y_total) if Y_total > 0 else 0.0,
        "tau_ns": float(tau_pinch_s * 1e9),
    }


def density_from_pressure(
    gas_key: str, pressure_torr: float, temp_K: float = 300.0,
) -> float:
    """Compute mass density from ideal gas law: rho = P * m / (kB * T)."""
    gas = GAS_SPECIES[gas_key]
    pressure_pa = pressure_torr * 133.322
    return pressure_pa * gas["m_mol"] / (kB * temp_K)


def run_simulation_core(
    preset_name: str,
    sim_time_us: float,
    gas_key: str | None = None,
    V0_kV: float | None = None,
    pressure_torr: float | None = None,
    C_uF: float | None = None,
    L0_nH: float | None = None,
    R0_mOhm: float | None = None,
    anode_r_mm: float | None = None,
    cathode_r_mm: float | None = None,
    anode_len_mm: float | None = None,
    fc: float | None = None,
    fm: float | None = None,
    crowbar_on: bool | None = None,
    crowbar_R_mOhm: float | None = None,
    progress_fn=None,
) -> dict[str, Any]:
    """Run snowplow + circuit Lee model. Returns dict with all arrays + metadata."""
    preset = get_preset(preset_name)
    cc = preset["circuit"]
    sc = preset.get("snowplow", {})
    rho0 = preset["rho0"]
    gas = GAS_SPECIES.get(gas_key or "D2", GAS_SPECIES["D2"])

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
    if anode_len_mm is not None and anode_len_mm > 0:
        sc["anode_length"] = anode_len_mm * 1e-3
    if fc is not None and fc > 0:
        sc["current_fraction"] = fc
    if fm is not None and fm > 0:
        sc["mass_fraction"] = fm
    if crowbar_on is not None:
        cc["crowbar_enabled"] = crowbar_on
    if crowbar_R_mOhm is not None and crowbar_R_mOhm > 0:
        cc["crowbar_resistance"] = crowbar_R_mOhm * 1e-3

    if pressure_torr is not None and pressure_torr > 0:
        p_pa = pressure_torr * 133.322
        sc["fill_pressure_Pa"] = p_pa
        rho0 = p_pa * gas["m_mol"] / (kB * 300.0)
    elif gas_key and gas_key != "D2":
        p_pa = sc.get("fill_pressure_Pa", 400.0)
        rho0 = p_pa * gas["m_mol"] / (kB * 300.0)

    t_end = sim_time_us * 1e-6

    circuit = RLCSolver(
        C=cc["C"], V0=cc["V0"], L0=cc["L0"],
        R0=cc.get("R0", 0.0),
        anode_radius=cc["anode_radius"], cathode_radius=cc["cathode_radius"],
        crowbar_enabled=cc.get("crowbar_enabled", False),
        crowbar_mode=cc.get("crowbar_mode", "voltage_zero"),
        crowbar_time=cc.get("crowbar_time", 0.0),
        crowbar_resistance=cc.get("crowbar_resistance", 0.0),
    )

    snowplow = None
    if sc:
        snowplow = SnowplowModel(
            anode_radius=cc["anode_radius"], cathode_radius=cc["cathode_radius"],
            fill_density=rho0,
            anode_length=sc.get("anode_length", 0.16),
            mass_fraction=sc.get("mass_fraction", 0.15),
            fill_pressure_Pa=sc.get("fill_pressure_Pa", 400.0),
            current_fraction=sc.get("current_fraction", 0.7),
            radial_mass_fraction=sc.get("radial_mass_fraction", None),
            pinch_column_fraction=sc.get("pinch_column_fraction", 1.0),
        )

    L_total = cc["L0"] + 1e-9
    T_LC = 2 * np.pi * np.sqrt(L_total * cc["C"])
    dt = T_LC / 5000
    n_steps = int(t_end / dt)

    times, currents, voltages, L_plasmas = [], [], [], []
    sheath_zs, shock_rs, phases_list = [], [], []
    E_cap, E_ind, E_res = [], [], []

    t = 0.0
    coupling = CouplingState()
    t0_wall = wall_time.perf_counter()

    for step in range(n_steps + 1):
        if snowplow is not None:
            sp = snowplow.step(dt, circuit.current)
            coupling.Lp = sp["L_plasma"]
            coupling.dL_dt = sp["dL_dt"]
            coupling.R_plasma = sp.get("R_plasma", 0.0)
            sheath_zs.append(sp["z_sheath"] * 1e3)
            shock_rs.append(sp["r_shock"] * 1e3)
            phases_list.append(sp["phase"])
        else:
            sheath_zs.append(0.0)
            shock_rs.append(0.0)
            phases_list.append("none")

        coupling = circuit.step(coupling, back_emf=0.0, dt=dt)
        t += dt
        times.append(t * 1e6)
        currents.append(circuit.current / 1e6)
        voltages.append(circuit.voltage / 1e3)
        L_plasmas.append(coupling.Lp * 1e9)
        E_cap.append(circuit.state.energy_cap / 1e3)
        E_ind.append(circuit.state.energy_ind / 1e3)
        E_res.append(circuit.state.energy_res / 1e3)

        if progress_fn and step % 500 == 0:
            progress_fn((step + 1) / (n_steps + 1), desc=f"t = {t*1e6:.1f} us")

    elapsed = wall_time.perf_counter() - t0_wall
    t_arr = np.array(times)
    I_arr = np.array(currents)

    I_peak_idx = int(np.argmax(np.abs(I_arr)))
    I_peak = float(np.abs(I_arr[I_peak_idx]))
    t_peak = float(t_arr[I_peak_idx])

    meta = _PRESETS.get(preset_name, {}).get("_meta", {})
    E_bank = 0.5 * cc["C"] * cc["V0"] ** 2

    dip_pct = 0.0
    I_pre_dip = I_peak
    t_pre_dip = t_peak
    I_dip = I_peak
    t_dip = t_peak
    scaling = None
    crowbar_t = None

    if snowplow is not None:
        dip_mask = np.array([(p in ("radial", "pinch", "reflected")) for p in phases_list])
        if np.any(dip_mask):
            dip_region = np.where(dip_mask)[0]
            dip_idx = dip_region[int(np.argmin(I_arr[dip_region]))]
            I_dip = float(I_arr[dip_idx])
            t_dip = float(t_arr[dip_idx])
            pre_dip_slice = I_arr[:dip_region[0]]
            pre_dip_idx = int(np.argmax(pre_dip_slice))
            I_pre_dip = float(pre_dip_slice[pre_dip_idx])
            t_pre_dip = float(t_arr[pre_dip_idx])
            dip_pct = (1 - I_dip / I_pre_dip) * 100 if I_pre_dip > 0 else 0

            P_fill_Torr = sc.get("fill_pressure_Pa", 400) / 133.322
            scaling = implosion_scaling(
                I_pre_dip, cc["anode_radius"] * 100, P_fill_Torr,
            )

    V_arr = np.array(voltages)
    for i in range(1, len(V_arr)):
        if V_arr[i - 1] > 0 and V_arr[i] <= 0:
            crowbar_t = float(t_arr[i])
            break

    neutron_yield = None
    V_pinch_volts = 0.0
    if snowplow is not None and gas.get("A") == 2 and gas.get("Z") == 1:
        r_p = getattr(snowplow, "pinch_radius", snowplow.shock_radius)
        z_f = getattr(snowplow, "z_f", sc.get("anode_length", 0.16) * sc.get("pinch_column_fraction", 1.0))
        tau_ns = scaling["tau_exp_ns"] if scaling else 10.0
        fmr = sc.get("radial_mass_fraction", sc.get("mass_fraction", 0.15))

        # Compute V_pinch = I * dL/dt at the current dip (maximum pinch voltage)
        # This is the inductive voltage that accelerates ions during m=0 disruption
        dip_mask_arr = np.array([(p in ("radial", "pinch", "reflected")) for p in phases_list])
        if np.any(dip_mask_arr):
            dip_indices = np.where(dip_mask_arr)[0]
            dLdt_arr = np.gradient(np.array(L_plasmas) * 1e-9, np.array(times) * 1e-6)
            # V_pinch = I * dL/dt; find peak during radial phase
            V_pinch_arr = np.abs(I_arr[dip_indices] * 1e6 * dLdt_arr[dip_indices])
            V_pinch_volts = float(np.max(V_pinch_arr)) if len(V_pinch_arr) > 0 else 0.0

        neutron_yield = dd_neutron_yield(
            I_pinch_A=I_dip * 1e6, r_pinch_m=r_p,
            z_pinch_m=z_f, tau_pinch_s=tau_ns * 1e-9,
            rho0=rho0, ion_mass=gas["m_mol"],
            cathode_r_m=cc["cathode_radius"],
            anode_r_m=cc["anode_radius"],
            mass_fraction=fmr,
            V_pinch_volts=V_pinch_volts,
        )

    return {
        "t_us": t_arr, "I_MA": I_arr, "V_kV": V_arr,
        "L_p_nH": np.array(L_plasmas),
        "z_mm": np.array(sheath_zs), "r_mm": np.array(shock_rs),
        "phases": phases_list,
        "E_cap_kJ": np.array(E_cap), "E_ind_kJ": np.array(E_ind),
        "E_res_kJ": np.array(E_res),
        "I_peak": I_peak, "t_peak": t_peak,
        "I_pre_dip": I_pre_dip, "t_pre_dip": t_pre_dip,
        "I_dip": I_dip, "t_dip": t_dip,
        "dip_pct": dip_pct, "scaling": scaling, "crowbar_t": crowbar_t,
        "E_bank_kJ": E_bank / 1e3, "T_LC_us": T_LC * 1e6, "dt_ns": dt * 1e9,
        "n_steps": n_steps, "elapsed_s": elapsed,
        "device": meta.get("device", preset_name),
        "circuit": cc, "snowplow_cfg": sc,
        "gas": gas, "gas_key": gas_key or "D2",
        "rho0": rho0,
        "snowplow_obj": snowplow,
        "has_snowplow": snowplow is not None,
        "neutron_yield": neutron_yield,
    }
