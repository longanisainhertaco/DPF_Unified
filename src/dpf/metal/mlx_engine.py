"""Pure-MLX DPF engine: circuit + snowplow + MHD with zero CPU sync.

Chains MLXCircuitSolver + MLXSnowplow + MLXMHDSolver into a single
discharge simulation. In MHD mode, the MHD solver's density-weighted
plasma inductance feeds back into the circuit ODE, replacing the
snowplow's analytic Lp once the MHD fields are resolved.

Compatible with mx.grad for differentiable calibration.

Usage:
    from dpf.metal.mlx_engine import run_mlx_discharge
    result = run_mlx_discharge(preset_name="pf1000", max_steps=5000)
"""
from __future__ import annotations

import math
import time
from typing import Any

import numpy as np

from dpf.metal.mlx_circuit import MLXCircuitSolver
from dpf.metal.mlx_snowplow import MLXSnowplow
from dpf.presets import get_preset


def _blend_lp(
    Lp_sp: float,
    dLp_dt_sp: float,
    Lp_mhd: float,
    dLp_dt_mhd: float | None,
    alpha: float,
    prev_Lp: float,
) -> tuple[float, float]:
    """Blend snowplow and MHD inductance with jump clamping.

    Parameters
    ----------
    Lp_sp, dLp_dt_sp : float
        Snowplow inductance [H] and its time derivative [H/s].
    Lp_mhd : float
        MHD density-weighted inductance [H].
    dLp_dt_mhd : float or None
        MHD dL/dt [H/s]. None if not yet computed.
    alpha : float
        Blend weight in [0, 1]. 0 = pure snowplow, 1 = pure MHD.
    prev_Lp : float
        Previous timestep blended Lp [H], for jump clamping.

    Returns
    -------
    tuple[float, float]
        (blended Lp, blended dLp_dt)
    """
    dLp_dt_m = dLp_dt_mhd if dLp_dt_mhd is not None else dLp_dt_sp
    Lp = alpha * Lp_mhd + (1.0 - alpha) * Lp_sp
    dLp_dt = alpha * dLp_dt_m + (1.0 - alpha) * dLp_dt_sp

    # Clamp: no >20% Lp jump per step (matches SimulationEngine)
    if prev_Lp > 0:
        ratio = Lp / prev_Lp
        if ratio > 1.2:
            Lp = 1.2 * prev_Lp
        elif ratio < 0.8:
            Lp = 0.8 * prev_Lp
    return Lp, dLp_dt


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
              In MHD mode, the solver's density-weighted Lp feeds back
              into the circuit once the fields are resolved.
        grid_shape: MHD grid resolution (only used in "mhd" mode).

    Returns dict with I_peak_MA, t_peak_us, n_steps, elapsed_s, and
    time-series arrays (I_MA, t_us, Lp_nH, Lp_mhd_nH, phases).
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

    # MHD t_peak is ~30% later than snowplow due to numerical diffusion.
    # 15 us gives enough margin for the current to peak and decay.
    sim_time = preset.get("sim_time", 10e-6)
    if mode == "mhd":
        sim_time = max(sim_time, 12e-6)  # MHD t_peak is later than snowplow

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
        from dpf.metal.mlx_solver import MLXMHDSolver

        r_anode = cc["anode_radius"]
        r_cathode = cc["cathode_radius"]
        anode_length = sp_cfg.get("anode_length", 0.16)
        nr, ny, nz = grid_shape
        # Grid spans FULL radial extent: axis (r=0) to cathode (r=r_cathode).
        # During axial rundown, plasma is between anode and cathode (r_a < r < r_c).
        # During radial phase, the sheath moves inward from r_anode toward r=0.
        # The pinch forms at r ~ 0. The old grid (r_anode to r_cathode) could
        # not resolve the radial implosion or pinch — the entire region where
        # L_p changes was outside the computational domain.
        #
        # Sun et al. (2025), Acta Phys. Sin. 74:115201, Fig. 2:
        #   Full domain 0 to r_cathode, anode as internal boundary.
        # Ou Haibin et al. (2024), Fig. 1:
        #   Full domain including axis, anode excluded as solid body.
        dr_mhd = r_cathode / nr  # full radial extent
        dz_mhd = anode_length / nz

        mhd_solver = MLXMHDSolver(
            grid_shape=grid_shape, dx=dr_mhd, dz=dz_mhd,
            riemann_solver="hlls", reconstruction="plm",
            time_integrator="ssp_rk2", coordinates="cylindrical",
            r_inner=0.0,  # domain starts at axis, not at anode surface
            cathode_radius=r_cathode,
            ion_mass=_m_D2 / 2.0,
            # Electrode BC computes B_theta = mu0*I/(2*pi*r) in SI Tesla.
            # The MLX solver operates in Heaviside-Lorentz units (mu0=1).
            # Without this flag, B is 1/sqrt(mu0) ~ 1000x too weak,
            # producing negligible J×B compression.
            convert_b_si_to_hl=True,
            # Spitzer resistivity with implicit Thomas solver.
            # Stone & Norman, ApJS 80:753 (1992): operator-split implicit diffusion.
            # The Thomas tridiagonal solver is unconditionally stable at any eta.
            # The RKL2 STS path causes NaN in vacuum cells — disabled via
            # use_rkl2_transport=False.
            # Spitzer: eta = 5.2e-5 * Z * ln(Lambda) / Te_eV^1.5 [Ohm*m]
            # Johnson et al. (2024): Spitzer confirmed valid for DPF conditions.
            resistivity_model="spitzer",
            use_rkl2_transport=False,  # Thomas solver, not STS (avoids NaN)
        )
        # Initial state: fill gas between electrodes (r_anode < r < r_cathode).
        # Inside the anode (r < r_anode): vacuum/low density (solid body).
        # Outside this range: fill gas at fill pressure and temperature.
        r_cells = (np.arange(nr) + 0.5) * dr_mhd  # cell centers from 0 to r_cathode
        in_gap = (r_cells >= r_anode) & (r_cells <= r_cathode)
        rho_init = np.full((nr, ny, nz), rho0 * 1e-4, dtype=np.float32)  # vacuum inside anode
        rho_init[in_gap, :, :] = rho0  # fill gas in electrode gap
        p_init = np.full((nr, ny, nz), _p_Pa * 1e-4, dtype=np.float32)
        p_init[in_gap, :, :] = _p_Pa

        # Sheath seed at inlet (z=0): thin dense layer per Sun et al. (2025)
        # Section 3.2: "at the inlet, a plasma thin layer with average density
        # ~4*n0 and temperature ~2 eV is initialized."
        # This pre-formed sheath carries the initial current and B_theta.
        # J×B accelerates it axially — B propagates WITH the dense layer.
        # Without this seed, B sits at the inlet and doesn't propagate.
        _n_sheath_cells = max(2, nz // 16)  # ~4-8 cells at z=0
        _kB = 1.380649e-23
        _Te_sheath = 2.0 * 1.602e-19 / _kB  # 2 eV in Kelvin = ~23,200 K
        rho_init[in_gap, :, :_n_sheath_cells] = 4.0 * rho0  # 4x fill density
        p_init[in_gap, :, :_n_sheath_cells] = (
            4.0 * rho0 / (_m_D2 / 2.0) * _kB * _Te_sheath * 2.0  # (1+Z)*n*kB*T
        )

        mhd_state = {
            "rho": rho_init,
            "velocity": np.zeros((3, nr, ny, nz), dtype=np.float32),
            "pressure": p_init,
            "B": np.zeros((3, nr, ny, nz), dtype=np.float32),
            "Te": np.full((nr, ny, nz), 300.0, dtype=np.float32),
            "Ti": np.full((nr, ny, nz), 300.0, dtype=np.float32),
        }

    # Timestep from LC period
    L_total = cc["L0"] + 1e-9
    T_LC = 2.0 * math.pi * math.sqrt(L_total * cc["C"])
    dt = T_LC / 5000.0
    n_steps_max = min(max_steps, int(sim_time / dt) + 1)

    # Short-circuit current for divergence guard
    I_sc = _V0 / max(math.sqrt(cc["L0"] / cc["C"]), 1e-30)
    I_diverge = 10.0 * I_sc

    # Time series storage
    times: list[float] = []
    currents: list[float] = []
    voltages: list[float] = []
    Lp_list: list[float] = []
    Lp_mhd_list: list[float] = []
    _phi_history: list[tuple[float, float]] = []  # (time, Phi_SI) for voltage-flux BDF2
    phases: list[str] = []

    t = 0.0
    t0_wall = time.perf_counter()
    I_peak = 0.0
    t_peak = 0.0

    # MHD-circuit coupling state
    blend_alpha = 0.0
    blend_active = False
    prev_Lp_blend = 0.0
    # Phases where MHD Lp can be trusted (includes rundown with density gate)
    _MHD_TRUST_PHASES = {"rundown", "radial", "radial_reflected", "pinch", "column"}

    for _step in range(n_steps_max):
        # Snowplow step (always runs — provides phase detection)
        sp_result = snowplow.step(
            dt, circuit.current,
            voltage=circuit.voltage, R0=cc["R0"], L0=cc["L0"],
        )
        Lp_sp = sp_result["L_plasma"]
        dLp_dt_sp = sp_result["dL_dt"]
        R_plasma = sp_result.get("R_plasma", 0.0)
        phase = sp_result["phase"]

        # MHD step + coupling feedback
        Lp_mhd_val = 0.0
        U_PF = 0.0  # default: no MHD back-EMF
        if mhd_solver is not None and mhd_state is not None:
            mhd_dt = mhd_solver._compute_dt(mhd_state)
            mhd_dt = min(mhd_dt, dt)
            # Z-dependent electrode BC from MHD density field.
            # B_theta only injected where the sheath has compressed the gas
            # (rho > 1.5 * rho_fill). Ahead of the sheath, neutral gas
            # cannot carry current. Ou Haibin et al. (2024), Fig. 1.
            # This produces the progressive L_p growth that the snowplow
            # captures analytically — without it, B fills the entire domain
            # instantly and L_p is constant (no dL/dt, no back-EMF timing).
            rho_mhd = np.asarray(mhd_state["rho"]).squeeze()
            rho_col_max = np.max(rho_mhd, axis=0)  # max density along r, per z
            compressed_z = rho_col_max > 1.5 * rho0
            if np.any(compressed_z):
                _z_frac_mhd = float(np.max(np.where(compressed_z)[0]) + 1) / rho_mhd.shape[1]
            else:
                _z_frac_mhd = max(2.0 / rho_mhd.shape[1], 0.03)
            mhd_state = mhd_solver.step(
                mhd_state, mhd_dt,
                current=circuit.current, voltage=circuit.voltage,
                apply_electrode_bc=True,
                z_sheath_frac=_z_frac_mhd,
            )

            # Voltage-flux coupling: U_PF = dPhi/dt at inlet boundary.
            # Replaces the density-weighted Lee formula (which systematically
            # miscalculates Lp because density centroid ≠ current centroid).
            #
            # Sun et al. (2025), Acta Physica Sinica 74:115201, Eq. (15)-(17).
            # Beresnyak et al. (2018), IEEE TPS 46:3881 (NRL Athena DPF).
            # Auluck (2021), Phys. Plasmas 28:030703: proves Lee formula is
            #   fundamentally incomplete for moving plasma boundaries.
            from dpf.metal.mlx_coupling import compute_upf_voltage_flux
            coupling, U_PF = compute_upf_voltage_flux(
                mhd_solver._U, mhd_solver._grid,
                r_inner=r_anode, cathode_radius=r_cathode,
                phi_history=_phi_history,
                sim_time=t, current=circuit.current, voltage=circuit.voltage,
            )
            Lp_mhd_val = coupling.Lp  # diagnostic only (Phi/I)
            dLp_dt_mhd = coupling.dL_dt

        # Circuit step
        if mhd_solver is not None and mhd_state is not None:
            # MHD drives circuit via U_PF = dPhi/dt (voltage-flux coupling).
            # U_PF replaces I*dLp/dt entirely — set Lp=0, dLp_dt=0 to avoid
            # double-counting. The circuit equation becomes:
            #   L0*dI/dt = V_cap - R*I - U_PF
            # Sun et al. (2025) Eq. (15): L0 is external only, U_PF captures all DPF flux.
            circuit.step(
                Lp=0.0, dLp_dt=0.0, R_plasma=R_plasma,
                back_emf=U_PF, dt=dt,
            )
        else:
            circuit.step(Lp=Lp_sp, dLp_dt=dLp_dt_sp, R_plasma=R_plasma, back_emf=0.0, dt=dt)
        t += dt

        # Record
        I_MA = circuit.current / 1e6
        times.append(t * 1e6)
        currents.append(I_MA)
        voltages.append(circuit.voltage / 1e3)
        Lp_list.append(Lp_sp * 1e9)
        Lp_mhd_list.append(Lp_mhd_val * 1e9)
        phases.append(phase)

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
        "Lp_mhd_nH": Lp_mhd_list,
        "phases": phases,
        "blend_alpha": blend_alpha,
    }
