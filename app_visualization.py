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


def extract_all_layers(d: dict[str, Any]) -> dict[str, Any]:
    """Extract all visualization layers from simulation result.

    Returns a dict ready for JSON serialization to the Babylon renderer.
    """
    cc = d.get("circuit", {})
    a = cc.get("anode_radius", 0.01)
    b = cc.get("cathode_radius", 0.03)
    L = d.get("snowplow_cfg", {}).get("anode_length", 0.16)

    # Layer 1: Geometry (always present)
    geometry = {
        "anode_radius": a * 1e3,
        "cathode_radius": b * 1e3,
        "anode_length": L * 1e3,
    }

    # Layer 2: Sheath timeline
    t_us = np.array(d.get("t_us", [0]))
    z_mm = np.array(d.get("z_mm", [0]))
    r_mm = np.array(d.get("r_mm", [0]))
    I_MA = np.array(d.get("I_MA", [0]))
    phases = d.get("phases", ["none"] * len(t_us))

    n = len(t_us)
    step = max(1, n // 60)
    idx = list(range(0, n, step))
    if idx[-1] != n - 1:
        idx.append(n - 1)

    frames = [{"t": float(t_us[i]), "z": float(z_mm[i]),
               "r": float(r_mm[i]), "I": float(I_MA[i]),
               "phase": phases[i]} for i in idx]

    sheath = {
        "frames": frames,
        "n_frames": len(frames),
        "I_peak": float(np.max(np.abs(I_MA))),
    }

    # Layers 3-5: MHD field data (only if final_state exists)
    final = d.get("final_state")
    mhd_snapshots = d.get("mhd_snapshots", [])
    density = None
    temperature = None
    bfield = None
    radiation_layer = None
    yield_layer = None

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
    # Limit to 30 snapshots max to keep Babylon iframe payload under ~500KB.
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
        density["frames"] = [
            {
                "t_us": float(s["t_us"]),
                "data": _b64((rho_arrays[i] - rho_global_lo) / rho_scale),
            }
            for i, s in enumerate(mhd_snapshots)
        ]
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

        # --- bfield frames (magnitude + Br/Bz components for field line tracing) ---
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

    return {
        "geometry": geometry,
        "sheath": sheath,
        "density": density,
        "temperature": temperature,
        "bfield": bfield,
        "velocity": vel_layer,
        "pinch": pinch,
        "beam": beam,
        "instability": instability,
        "radiation": radiation_layer,
        "yield_map": yield_layer,
        "device": d.get("device", "DPF"),
        "backend": d.get("backend", "lee"),
        "has_mhd": d.get("has_mhd", False),
    }
