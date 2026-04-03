"""Unified physics visualization data pipeline.

Extracts all physics layers from simulation results for the Babylon.js
renderer. Each layer is a self-contained data structure that the JS
renderer can toggle independently.

Layers:
1. geometry    — electrode dimensions (always on)
2. sheath      — current sheath position + phase timeline
3. density     — rho(r,z) normalized heatmap from MHD state
4. temperature — Te(r,z) normalized heatmap
5. bfield      — |B|(r,z) + poloidal field line seed points
6. pinch       — pinch column metrics (radius, intensity, position)
7. beam        — beam ion trajectories from BeamTracker
8. instability — m=0 sausage perturbation amplitude
9. radiation   — radiation cooling power density
10. yield      — neutron yield rate spatial distribution
"""
from __future__ import annotations

import base64
from typing import Any

import numpy as np


def _b64(arr: np.ndarray) -> str:
    return base64.b64encode(arr.astype(np.float32).tobytes()).decode("ascii")


def _norm(x: np.ndarray) -> np.ndarray:
    lo, hi = float(x.min()), float(x.max())
    return (x - lo) / max(hi - lo, 1e-30)


def _midplane(x: np.ndarray) -> np.ndarray:
    """Extract midplane slice from 3D array."""
    if x.ndim == 3:
        return x[:, x.shape[1] // 2, :]
    return x


def _density_isosurface(
    rho_mid: np.ndarray, a_mm: float, b_mm: float, threshold_frac: float = 0.3
) -> list[float]:
    """Extract r(z) isosurface contour from rho(r,z) at a density threshold.

    For each z-column, finds the outermost radius where rho exceeds
    threshold_frac * rho_max_in_column. Returns r in mm for each z-cell.
    This gives the sheath/pinch surface shape from MHD data.

    Physics: the density isosurface traces the current sheath boundary.
    Lee 2014 p.324-325: r_s (shock front) is where density jumps from ambient.
    Ch.13 Eq. 3.8 (Bennett): n(r) = n0/(1+n0*b*r^2)^2 — peaked on axis.
    """
    nr, nz = rho_mid.shape
    r_grid = np.linspace(a_mm, b_mm, nr)
    r_iso = []
    for iz in range(nz):
        col = rho_mid[:, iz]
        col_max = col.max()
        if col_max < 1e-30:
            r_iso.append(float(b_mm))
            continue
        thresh = threshold_frac * col_max
        # Find outermost radius above threshold (scan from outside inward)
        idx_above = np.where(col >= thresh)[0]
        if len(idx_above) == 0:
            r_iso.append(float(b_mm))
        else:
            r_iso.append(float(r_grid[idx_above[-1]]))
    return r_iso


def extract_all_layers(d: dict[str, Any]) -> dict[str, Any]:
    """Extract all visualization layers from simulation result.

    Returns a dict ready for JSON serialization to the Babylon renderer.
    """
    cc = d.get("circuit", {})
    a = cc.get("anode_radius", 0.01)
    b = cc.get("cathode_radius", 0.03)
    L = d.get("snowplow_cfg", {}).get("anode_length", 0.16)

    # Layer 1: Geometry (always present)
    fill_p = d.get("snowplow_cfg", {}).get("fill_pressure_Pa", 400.0)
    n_rods = cc.get("n_cathode_rods", 0)  # 0 = let renderer use its default (8)
    geometry = {
        "anode_radius": a * 1e3,
        "cathode_radius": b * 1e3,
        "anode_length": L * 1e3,
        "fill_pressure_Pa": fill_p,
        "n_cathode_rods": n_rods,
    }

    # Layer 2: Sheath timeline
    t_us = np.array(d.get("t_us", [0]))
    z_mm = np.array(d.get("z_mm", [0]))
    r_mm = np.array(d.get("r_mm", [0]))
    # Piston radius (magnetic piston) — distinct from shock radius during radial phase
    # Lee 2014 p.324 Fig.4: r_s (shock) and r_p (piston) are separate trajectories
    r_p_mm = np.array(d.get("r_p_mm", r_mm))  # fallback to shock radius if not available
    I_MA = np.array(d.get("I_MA", [0]))
    phases = d.get("phases", ["none"] * len(t_us))

    n = len(t_us)
    step = max(1, n // 60)
    idx = list(range(0, n, step))
    if idx[-1] != n - 1:
        idx.append(n - 1)

    frames = [{"t": float(t_us[i]), "z": float(z_mm[i]),
               "r": float(r_mm[i]), "r_p": float(r_p_mm[i]),
               "I": float(I_MA[i]),
               "phase": phases[i]} for i in idx]

    # Re-strike detection: Damideh 2025 p.1: "re-strikes which divert current away"
    # The two-step radial model marks re-strike at radial_transition_time
    sp_cfg = d.get("snowplow_cfg", {})
    restrike_t_us = None
    if sp_cfg.get("radial_transition_time") and sp_cfg.get("radial_current_fraction_2"):
        restrike_t_us = float(sp_cfg["radial_transition_time"] * 1e6)

    sheath = {
        "frames": frames,
        "n_frames": len(frames),
        "I_peak": float(np.max(np.abs(I_MA))),
        "restrike_t_us": restrike_t_us,  # None if no re-strike in this device
    }

    # Layers 3-5: MHD field data (only if final_state exists)
    final = d.get("final_state")
    mhd_snapshots = d.get("mhd_snapshots", [])
    density = None
    temperature = None
    bfield = None
    radiation_layer = None
    yield_layer = None
    beta_layer = None
    j_layer = None
    ohmic_layer = None
    Ti_layer = None

    if final is not None:
        rho = final["rho"]
        rho_mid = _midplane(rho)
        rho_norm = _norm(rho_mid)
        shape = list(rho_mid.shape)

        density = {
            "data": _b64(rho_norm),
            "shape": shape,
            "max_val": float(rho.max()),
            "min_val": float(rho.min()),
            "compression_ratio": float(rho.max() / max(rho.min(), 1e-30)),
        }

        Te = final.get("Te")
        if Te is not None:
            Te_mid = _midplane(Te)
            Te_norm = _norm(Te_mid)
            Te_eV = Te_mid * 1.380649e-23 / 1.602e-19
            temperature = {
                "data": _b64(Te_norm),
                "shape": shape,
                "max_eV": float(Te_eV.max()),
                "min_eV": float(Te_eV.min()),
            }

            # Ion temperature — separate from T_e for two-temperature physics
            # Ch.13 Eq.2.13 p.342: p = nk(T_e + T_i) — they may differ
            # Chen Eq.5.76 p.171: resistivity depends on T_e only
            Ti = final.get("Ti")
            Ti_layer = None
            if Ti is not None:
                Ti_mid = _midplane(Ti)
                Ti_eV = Ti_mid * 1.380649e-23 / 1.602e-19
                Ti_layer = {
                    "data": _b64(_norm(Ti_mid)),
                    "shape": shape,
                    "max_eV": float(Ti_eV.max()),
                    "min_eV": float(Ti_eV.min()),
                }

            # Layer 9: Radiation cooling (P_rad ~ ne^2 * sqrt(Te))
            ion_mass = d.get("gas", {}).get("m_mol", 3.34e-27)
            ne = rho / ion_mass
            ne_mid = _midplane(ne)
            Te_eV_mid = _midplane(Te) * 1.380649e-23 / 1.602e-19
            P_rad = 1.69e-32 * ne_mid**2 * np.sqrt(np.maximum(Te_eV_mid, 0.1))
            P_rad_norm = _norm(P_rad)
            radiation_layer = {
                "data": _b64(P_rad_norm),
                "shape": shape,
                "max_W_m3": float(P_rad.max()),
            }

        B = final.get("B")
        if B is not None:
            B_mag = np.sqrt(np.sum(B**2, axis=0))
            B_mid = _midplane(B_mag)
            B_norm = _norm(B_mid)

            # Field line seed points (evenly spaced in r)
            nr, nz = B_mid.shape
            n_seeds = min(12, nr // 2)
            seed_r = np.linspace(0.2, 0.8, n_seeds)  # normalized r positions
            seed_z = [0.5] * n_seeds  # start at midplane z

            # Poloidal B components for field line tracing
            Br_mid = _midplane(B[0]) if B.shape[0] > 0 else np.zeros_like(B_mid)
            Bz_mid = _midplane(B[2]) if B.shape[0] > 2 else np.zeros_like(B_mid)

            bfield = {
                "data": _b64(B_norm),
                "Br": _b64(Br_mid.astype(np.float32)),
                "Bz": _b64(Bz_mid.astype(np.float32)),
                "shape": shape,
                "max_T": float(B_mag.max()),
                "seed_r": seed_r.tolist(),
                "seed_z": seed_z,
            }

            # Layer 10: Neutron yield spatial distribution
            # Y ~ n^2 * <sigma*v>(T) ~ n^2 * T^4 (rough approximation)
            if Te is not None and d.get("gas", {}).get("A") == 2:
                ne_mid2 = ne_mid**2
                Te_keV = Te_eV_mid / 1e3
                # Rough D-D reactivity scaling: <sv> ~ T^4 below 20 keV
                yield_rate = ne_mid2 * np.maximum(Te_keV, 0)**4
                yield_norm = _norm(yield_rate)
                yield_layer = {
                    "data": _b64(yield_norm),
                    "shape": shape,
                    "max_rate": float(yield_rate.max()),
                }

    # Encode mhd_snapshots as per-field time-series frames.
    # Each snapshot is {t_us, rho_mid, B_mid, P_mid, vel_mid} from the solver.
    # Normalisation is per-field-across-all-snaps so colours stay consistent.
    # Cap at 30 frames for animation smoothness (more than 30 doesn't improve visual quality).
    # No payload size limit — renderer HTML is served as a file, not srcdoc.
    vel_layer: dict[str, Any] | None = None
    if mhd_snapshots and len(mhd_snapshots) > 30:
        step = max(1, len(mhd_snapshots) // 30)
        mhd_snapshots = mhd_snapshots[::step][:30]
    if mhd_snapshots and density is not None:
        snap_shape = list(np.asarray(mhd_snapshots[0]["rho_mid"]).shape)

        # --- density frames ---
        rho_arrays = [np.asarray(s["rho_mid"], dtype=np.float32) for s in mhd_snapshots]
        rho_global_lo = float(min(a.min() for a in rho_arrays))
        rho_global_hi = float(max(a.max() for a in rho_arrays))
        rho_scale = max(rho_global_hi - rho_global_lo, 1e-30)
        a_mm = a * 1e3
        b_mm = b * 1e3
        density["frames"] = []
        density["isosurface_frames"] = []  # r(z) contour per frame for 3D geometry
        for i, s in enumerate(mhd_snapshots):
            density["frames"].append({
                "t_us": float(s["t_us"]),
                "data": _b64((rho_arrays[i] - rho_global_lo) / rho_scale),
            })
            # Extract density isosurface: r(z) at 30% of column-peak density
            # This traces the sheath/pinch boundary from MHD data
            density["isosurface_frames"].append(
                _density_isosurface(rho_arrays[i], a_mm, b_mm, threshold_frac=0.3)
            )
        density["frames_shape"] = snap_shape

        # --- temperature frames (from P_mid via ideal-gas: T ~ P/rho) ---
        if temperature is not None and "P_mid" in mhd_snapshots[0]:
            P_arrays = [np.asarray(s["P_mid"], dtype=np.float32) for s in mhd_snapshots]
            # T_norm ~ P/rho (relative, dimensionless for colouring)
            T_arrays = [
                P_arrays[i] / np.maximum(rho_arrays[i], 1e-30)
                for i in range(len(mhd_snapshots))
            ]
            T_global_lo = float(min(a.min() for a in T_arrays))
            T_global_hi = float(max(a.max() for a in T_arrays))
            T_scale = max(T_global_hi - T_global_lo, 1e-30)
            temperature["frames"] = [
                {
                    "t_us": float(s["t_us"]),
                    "data": _b64((T_arrays[i] - T_global_lo) / T_scale),
                }
                for i, s in enumerate(mhd_snapshots)
            ]
            temperature["frames_shape"] = snap_shape

        # --- bfield frames (magnitude + Br/Bz/Bt components for field line tracing) ---
        if bfield is not None and "B_mid" in mhd_snapshots[0]:
            B_arrays = [np.asarray(s["B_mid"], dtype=np.float32) for s in mhd_snapshots]
            B_global_lo = float(min(a.min() for a in B_arrays))
            B_global_hi = float(max(a.max() for a in B_arrays))
            B_scale = max(B_global_hi - B_global_lo, 1e-30)
            bfield["frames"] = []
            for i, s in enumerate(mhd_snapshots):
                frame_entry: dict[str, Any] = {
                    "t_us": float(s["t_us"]),
                    "data": _b64((B_arrays[i] - B_global_lo) / B_scale),
                }
                b_full = np.asarray(s["B_mid"], dtype=np.float32)
                if b_full.ndim == 3 and b_full.shape[0] >= 3:
                    frame_entry["Br"] = _b64(b_full[0])
                    frame_entry["Bz"] = _b64(b_full[2])
                    frame_entry["Bt"] = _b64(b_full[1])
                bfield["frames"].append(frame_entry)
            bfield["frames_shape"] = snap_shape

        # --- current density J_z frames from curl(B) ---
        # Ch.13 Eq.2.17 p.343: (1/r) d/dr [r B_theta] = mu_0 J_z
        # Ch.13 Eq.2.18 p.343: J_z = (1/mu_0)(dB_theta/dr + B_theta/r)
        j_layer: dict[str, Any] | None = None
        if bfield is not None and "B_mid" in mhd_snapshots[0]:
            mu_0 = 4e-7 * np.pi
            nr_s = snap_shape[0]
            r_grid = np.linspace(a, b, nr_s)  # radial grid in meters
            dr = (b - a) / max(nr_s - 1, 1)
            j_frames = []
            for _ji, s in enumerate(mhd_snapshots):
                b_full = np.asarray(s["B_mid"], dtype=np.float32)
                if b_full.ndim == 3 and b_full.shape[0] >= 2:
                    Bt = b_full[1]  # B_theta component [nr, nz]
                else:
                    continue
                rBt = r_grid[:, None] * Bt
                d_rBt_dr = np.gradient(rBt, dr, axis=0)
                r_safe = np.maximum(r_grid[:, None], 1e-6)
                Jz = d_rBt_dr / (mu_0 * r_safe)
                Jz_norm = _norm(np.abs(Jz))
                j_frames.append({
                    "t_us": float(s["t_us"]),
                    "data": _b64(Jz_norm.astype(np.float32)),
                })
            if j_frames:
                j_layer = {
                    "frames": j_frames,
                    "frames_shape": snap_shape,
                    "description": "Current density |J_z| from curl(B). Ch.13 Eq.2.17-2.18.",
                }

            # --- Ohmic heating: P_ohm = J^2 * eta (Chen Eq.5.75 p.170) ---
            # Spitzer resistivity: eta = 5.2e-5 * Z * ln_Lambda / T_eV^1.5
            # (Chen Eq.5.76 p.171, verified from PDF this session)
            ohmic_layer: dict[str, Any] | None = None
            if "P_mid" in mhd_snapshots[0] and temperature is not None:
                ohmic_frames = []
                k_B = 1.380649e-23
                e_charge = 1.602e-19
                ln_Lambda = 10.0  # Chen p.169: "for most purposes ln Lambda = 10"
                for i, s in enumerate(mhd_snapshots):
                    b_full_i = np.asarray(s["B_mid"], dtype=np.float32)
                    if b_full_i.ndim != 3 or b_full_i.shape[0] < 2:
                        continue
                    Bt_i = b_full_i[1]
                    rBt_i = r_grid[:, None] * Bt_i
                    d_rBt_dr_i = np.gradient(rBt_i, dr, axis=0)
                    r_safe_i = np.maximum(r_grid[:, None], 1e-6)
                    Jz_i = d_rBt_dr_i / (mu_0 * r_safe_i)  # A/m^2
                    # Temperature from P/rho (total T ~ T_e + T_i, so T_e ~ T/2)
                    P_i = np.asarray(s["P_mid"], dtype=np.float32)
                    rho_i = rho_arrays[i]
                    ion_mass = d.get("gas", {}).get("m_mol", 3.34e-27)
                    T_K = P_i * ion_mass / (np.maximum(rho_i, 1e-30) * k_B)
                    T_eV = np.maximum(T_K * k_B / e_charge * 0.5, 0.1)  # T_e ~ T_total/2
                    # Spitzer: eta = 5.2e-5 * Z * ln_Lambda / T_eV^1.5
                    eta = 5.2e-5 * 1.0 * ln_Lambda / np.maximum(T_eV, 0.1) ** 1.5
                    P_ohm = Jz_i ** 2 * eta  # W/m^3
                    ohmic_frames.append({
                        "t_us": float(s["t_us"]),
                        "data": _b64(_norm(P_ohm).astype(np.float32)),
                    })
                if ohmic_frames:
                    ohmic_layer = {
                        "frames": ohmic_frames,
                        "frames_shape": snap_shape,
                    }

        # --- plasma beta frames: beta = 2*mu_0*P / B^2 (Chen Eq.6.8, p.191) ---
        # beta < 1: magnetically dominated. beta > 1: pressure dominated.
        # Chen Eq.6.7, p.191: p + B^2/(2*mu_0) = constant in equilibrium
        beta_layer: dict[str, Any] | None = None
        if "P_mid" in mhd_snapshots[0] and "B_mid" in mhd_snapshots[0]:
            mu_0 = 4e-7 * np.pi
            P_arrays = [np.asarray(s["P_mid"], dtype=np.float32) for s in mhd_snapshots]
            beta_frames = []
            for i, s in enumerate(mhd_snapshots):
                B_mag_i = B_arrays[i] if B_arrays[i].ndim == 2 else np.sqrt(
                    np.sum(np.asarray(s["B_mid"], dtype=np.float32) ** 2, axis=0))
                B_sq = np.maximum(B_mag_i ** 2, 1e-30)
                beta_i = 2.0 * mu_0 * P_arrays[i] / B_sq
                # Normalize: log10(beta) mapped to [0,1] with beta=1 at 0.5
                # Range: beta=0.01 (mag dominated) to beta=100 (pressure dominated)
                log_beta = np.clip(np.log10(np.maximum(beta_i, 1e-3)), -2, 2)
                beta_norm = (log_beta + 2) / 4.0  # maps [-2,2] to [0,1]
                beta_frames.append({
                    "t_us": float(s["t_us"]),
                    "data": _b64(beta_norm.astype(np.float32)),
                })
            beta_layer = {
                "frames": beta_frames,
                "frames_shape": snap_shape,
                "description": "Plasma beta = 2*mu_0*P/B^2 (Chen Eq.6.8). <1 = magnetic, >1 = pressure",
            }

        # --- velocity frames (vr, vz components for particle direction + isosurface) ---
        if "vel_mid" in mhd_snapshots[0]:
            vel_arrays = [np.asarray(s["vel_mid"], dtype=np.float32) for s in mhd_snapshots]
            vel_layer: dict[str, Any] = {"frames": [], "frames_shape": snap_shape}
            for i, s in enumerate(mhd_snapshots):
                v = vel_arrays[i]
                v_entry: dict[str, Any] = {"t_us": float(s["t_us"])}
                if v.ndim == 3 and v.shape[0] >= 3:
                    v_entry["vr"] = _b64(v[0])
                    v_entry["vz"] = _b64(v[2])
                    vmag = np.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2)
                    v_entry["vmag"] = _b64(_norm(vmag))
                elif v.ndim == 2:
                    v_entry["vmag"] = _b64(_norm(np.abs(v)))
                vel_layer["frames"].append(v_entry)
        else:
            vel_layer = None

    # Layer 6: Pinch metrics
    pinch = None
    sp = d.get("snowplow_obj")
    if sp and hasattr(sp, "pinch_radius"):
        pinch = {
            "radius_mm": float(sp.pinch_radius * 1e3) if sp.pinch_radius else 0,
            "position_mm": float(L * 1e3 * 0.85),
        }
    elif d.get("has_mhd") and final is not None:
        rho_max_r = np.argmax(rho_mid[:, rho_mid.shape[1] // 2])
        pinch = {
            "radius_mm": float(rho_max_r * (b - a) / rho_mid.shape[0] * 1e3),
            "position_mm": float(L * 1e3 * 0.85),
        }

    # Layer 7: Beam ion data
    beam = None
    bt = d.get("beam_tracker")
    if bt:
        beam = {
            "n_particles": bt["n_particles"],
            "mean_energy_keV": bt["mean_energy_keV"],
            "max_energy_keV": bt["max_energy_keV"],
        }

    # Layer 8: m=0 instability
    instability = None
    inst = d.get("instability")
    if inst:
        tau_ns = inst.get("tau_m0_ns", 0)
        n_efolds = float(d.get("t_peak", 0) * 1e3 / max(tau_ns, 1))
        instability = {
            "tau_m0_ns": tau_ns,
            "n_efolds": n_efolds,
            "amplitude": min(1.0, float(np.expm1(min(n_efolds, 50)))),
        }

    # Energy partition arrays (for energy bar visualization)
    e_cap_raw = d.get("E_cap_kJ", [])
    e_ind_raw = d.get("E_ind_kJ", [])
    e_res_raw = d.get("E_res_kJ", [])
    E_cap_kJ = [float(v) for v in np.asarray(e_cap_raw)[::step]] if len(e_cap_raw) else []
    E_ind_kJ = [float(v) for v in np.asarray(e_ind_raw)[::step]] if len(e_ind_raw) else []
    E_res_kJ = [float(v) for v in np.asarray(e_res_raw)[::step]] if len(e_res_raw) else []

    return {
        "geometry": geometry,
        "sheath": sheath,
        "density": density,
        "temperature": temperature,
        "bfield": bfield,
        "velocity": vel_layer,
        "beta": beta_layer,
        "current_density": j_layer,
        "ohmic_heating": ohmic_layer,
        "ion_temperature": Ti_layer,
        "pinch": pinch,
        "beam": beam,
        "instability": instability,
        "radiation": radiation_layer,
        "yield_map": yield_layer,
        "E_cap_kJ": E_cap_kJ,
        "E_ind_kJ": E_ind_kJ,
        "E_res_kJ": E_res_kJ,
        "device": d.get("device", "DPF"),
        "backend": d.get("backend", "lee"),
        "has_mhd": d.get("has_mhd", False),
    }
