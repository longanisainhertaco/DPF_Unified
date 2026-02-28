"""Named configuration presets for well-known DPF devices.

Each preset is a dictionary that can be unpacked into SimulationConfig(**preset).
Presets provide physically meaningful starting points for:
- Tutorial / quick-start (small grid, fast)
- PF-1000 (IPPLM Warsaw, 1 MJ)
- NX2 (NIE Singapore, 3 kJ)
- LLNL-DPF (Livermore, 4 kJ)
- MJOLNIR (LLNL, 2 MJ)

Usage:
    from dpf.presets import get_preset, list_presets
    config = get_preset("tutorial")
"""

from __future__ import annotations

from typing import Any

_PRESETS: dict[str, dict[str, Any]] = {
    "tutorial": {
        "_meta": {
            "description": "Minimal 8^3 Cartesian grid for quick tests and tutorials",
            "device": "Generic",
            "geometry": "cartesian",
        },
        "grid_shape": [8, 8, 8],
        "dx": 1e-3,
        "sim_time": 1e-7,
        "dt_init": 1e-10,
        "rho0": 1e-4,
        "T0": 300.0,
        "circuit": {
            "C": 1e-6,
            "V0": 1e3,
            "L0": 1e-7,
            "R0": 0.01,
            "anode_radius": 0.005,
            "cathode_radius": 0.01,
        },
    },
    "pf1000": {
        "_meta": {
            "description": "PF-1000 (IPPLM Warsaw) — 1 MJ deuterium DPF",
            "device": "PF-1000",
            "geometry": "cylindrical",
        },
        "grid_shape": [240, 1, 800],
        "dx": 7.5e-4,
        "sim_time": 5e-6,
        "dt_init": 1e-10,
        "rho0": 7.53e-4,  # 3.5 Torr D2 at 300K: n*m_D2 (molecular mass)
        "T0": 300.0,
        "anomalous_alpha": 0.05,
        "anomalous_threshold_model": "lhdi",
        # Circuit: Scholz et al., Nukleonika 51(1):79-84, 2006, Table 1
        # L0, R0 from short-circuit discharge calibration (includes AC effects)
        "circuit": {
            "C": 1.332e-3,     # 1.332 mF (Scholz 2006)
            "V0": 27e3,        # 27 kV charging voltage
            "L0": 33.5e-9,     # 33.5 nH external inductance
            "R0": 2.3e-3,      # 2.3 mOhm external resistance
            "anode_radius": 0.115,   # 115 mm (Scholz 2006)
            "cathode_radius": 0.16,  # 160 mm effective (Lee & Saw 2014)
            "crowbar_enabled": True,
            "crowbar_mode": "voltage_zero",
        },
        "geometry": {"type": "cylindrical"},
        "boundary": {"electrode_bc": True},
        "radiation": {"bremsstrahlung_enabled": True, "fld_enabled": True},
        "sheath": {"enabled": True, "boundary": "z_high"},
        "snowplow": {
            "anode_length": 0.6,  # Scholz (2006) Table 1: 600 mm
            "current_fraction": 0.816,  # Post-D2 calibration (Phase AC)
            "mass_fraction": 0.142,  # Post-D2 calibration (Phase AC)
            "radial_mass_fraction": 0.1,  # Lee & Saw (2014): f_mr ~ 0.07-0.12
            "pinch_column_fraction": 0.14,  # Lee & Saw (2014): z_f ~ 84 mm of 600 mm
        },
    },
    "nx2": {
        "_meta": {
            "description": "NX2 (NIE Singapore) — 3 kJ fast miniature DPF",
            "device": "NX2",
            "geometry": "cylindrical",
        },
        "grid_shape": [192, 1, 384],
        "dx": 2.5e-4,
        "sim_time": 1e-6,
        "dt_init": 1e-11,
        "rho0": 8e-5,
        "T0": 300.0,
        "anomalous_alpha": 0.03,
        "anomalous_threshold_model": "lhdi",
        "circuit": {
            "C": 28e-6,
            "V0": 11.5e3,         # 11.5 kV operating voltage (Lee & Saw 2008, Table 1)
            "L0": 20e-9,
            "R0": 5e-3,
            "anode_radius": 0.019,
            "cathode_radius": 0.041,
            "crowbar_enabled": True,
            "crowbar_mode": "voltage_zero",
        },
        "geometry": {"type": "cylindrical"},
        "boundary": {"electrode_bc": True},
        "radiation": {"bremsstrahlung_enabled": True},
        "snowplow": {
            "anode_length": 0.05,
            "fill_pressure_Pa": 133.0,  # ~1 Torr D2
            "current_fraction": 0.7,  # Lee & Saw (2008) Table 1
            "mass_fraction": 0.1,  # Lee & Saw (2008) Table 1
            "radial_mass_fraction": 0.12,  # Lee & Saw (2008): f_mr < f_m for NX2
            "pinch_column_fraction": 0.5,  # Small device: larger fraction focuses
        },
    },
    "llnl_dpf": {
        "_meta": {
            "description": "LLNL compact DPF — 4 kJ diagnostic device",
            "device": "LLNL-DPF",
            "geometry": "cylindrical",
            "reference": "Deutsch & Kies, Plasma Phys. Control. Fusion 30:263 (1988)",
        },
        "grid_shape": [64, 1, 128],
        "dx": 3e-4,
        "sim_time": 2e-6,
        "dt_init": 1e-11,
        "rho0": 1e-4,
        "T0": 300.0,
        "anomalous_alpha": 0.05,
        "anomalous_threshold_model": "lhdi",
        "circuit": {
            "C": 16e-6,
            "V0": 22e3,
            "L0": 50e-9,
            "R0": 8e-3,
            "anode_radius": 0.008,
            "cathode_radius": 0.015,
        },
        "geometry": {"type": "cylindrical"},
        "boundary": {"electrode_bc": True},
        "radiation": {"bremsstrahlung_enabled": True},
        "snowplow": {
            "anode_length": 0.08,
            "current_fraction": 0.7,  # Typical lab-scale DPF
            "mass_fraction": 0.15,  # Typical lab-scale DPF
            "pinch_column_fraction": 0.4,  # Lab-scale: moderate fraction
        },
    },
    "mjolnir": {
        "_meta": {
            "description": "MJOLNIR (LLNL) — 2 MJ MA-class deuterium DPF",
            "device": "MJOLNIR",
            "geometry": "cylindrical",
            "reference": "Goyon et al., Phys. Plasmas 32:033105 (2025)",
        },
        "grid_shape": [128, 1, 512],
        "dx": 1e-3,
        "sim_time": 8e-6,
        "dt_init": 1e-10,
        "rho0": 6e-4,  # ~7 Torr D2 fill
        "T0": 300.0,
        "anomalous_alpha": 0.05,
        "anomalous_threshold_model": "lhdi",
        # Circuit: Goyon et al. (2025) — 2 MJ stored at 100 kV design,
        # typical operation at 60 kV (0.75 MJ), 2.8 MA peak.
        # C = 2*E/V^2 = 2*2e6/100e3^2 = 0.4 mF (at design voltage)
        # L0, R0 estimated from quarter-period and peak current scaling.
        "circuit": {
            "C": 4e-4,             # 0.4 mF (2 MJ at 100 kV)
            "V0": 60e3,            # 60 kV typical operation
            "L0": 15e-9,           # ~15 nH external (MA-class low-inductance)
            "R0": 1e-3,            # ~1 mOhm (MA-class low-resistance)
            "anode_radius": 0.1143,  # 228.6 mm diameter / 2 (Goyon 2025)
            "cathode_radius": 0.16,  # estimated outer radius
            "crowbar_enabled": True,
            "crowbar_mode": "voltage_zero",
        },
        "geometry": {"type": "cylindrical"},
        "boundary": {"electrode_bc": True},
        "radiation": {"bremsstrahlung_enabled": True, "fld_enabled": True},
        "sheath": {"enabled": True, "boundary": "z_high"},
        # R_imp = 2.5 cm, anode length estimated from 15-degree taper geometry
        "snowplow": {
            "anode_length": 0.5,
            "current_fraction": 0.7,  # MA-class: similar to PF-1000
            "mass_fraction": 0.1,  # MA-class: similar to PF-1000
            "pinch_column_fraction": 0.14,  # MA-class geometry: ~14% per Lee & Saw
        },
    },
    "cartesian_demo": {
        "_meta": {
            "description": "32^3 Cartesian demo — all physics enabled",
            "device": "Generic",
            "geometry": "cartesian",
        },
        "grid_shape": [32, 32, 32],
        "dx": 5e-4,
        "sim_time": 5e-7,
        "dt_init": 1e-10,
        "rho0": 1e-4,
        "T0": 300.0,
        "circuit": {
            "C": 5e-6,
            "V0": 5e3,
            "L0": 5e-8,
            "R0": 0.01,
            "anode_radius": 0.005,
            "cathode_radius": 0.01,
        },
        "radiation": {"bremsstrahlung_enabled": True},
    },
    "phase_p_fidelity": {
        "_meta": {
            "description": "Phase P maximum fidelity: WENO5-Z + HLLD + SSP-RK3 + float64 (8.9/10)",
            "device": "Generic",
            "geometry": "cartesian",
        },
        "grid_shape": [32, 32, 32],
        "dx": 5e-4,
        "sim_time": 1e-7,
        "dt_init": 1e-10,
        "rho0": 1e-4,
        "T0": 300.0,
        "circuit": {
            "C": 5e-6,
            "V0": 5e3,
            "L0": 5e-8,
            "R0": 0.01,
            "anode_radius": 0.005,
            "cathode_radius": 0.01,
        },
        "fluid": {
            "backend": "metal",
            "reconstruction": "weno5",
            "riemann_solver": "hlld",
            "time_integrator": "ssp_rk3",
            "precision": "float64",
        },
        "radiation": {"bremsstrahlung_enabled": True},
    },
}


def list_presets() -> list[dict[str, str]]:
    """Return summary info for all available presets.

    Returns:
        List of dicts with keys: name, description, device, geometry, grid_shape.
    """
    result = []
    for name, preset in _PRESETS.items():
        meta = preset.get("_meta", {})
        result.append({
            "name": name,
            "description": meta.get("description", ""),
            "device": meta.get("device", ""),
            "geometry": meta.get("geometry", "cartesian"),
            "grid_shape": preset.get("grid_shape", []),
        })
    return result


def get_preset(name: str) -> dict[str, Any]:
    """Return a preset config dict (without _meta) suitable for SimulationConfig.

    Args:
        name: Preset name.

    Returns:
        Config dict ready for ``SimulationConfig(**preset)``.

    Raises:
        KeyError: If the preset name is not found.
    """
    if name not in _PRESETS:
        available = ", ".join(_PRESETS.keys())
        raise KeyError(f"Unknown preset '{name}'. Available: {available}")
    preset = dict(_PRESETS[name])
    preset.pop("_meta", None)
    return preset


def get_preset_names() -> list[str]:
    """Return list of all preset names."""
    return list(_PRESETS.keys())
