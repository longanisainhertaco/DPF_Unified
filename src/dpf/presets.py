"""Named configuration presets for well-known DPF devices.

Each preset is a dictionary that can be unpacked into SimulationConfig(**preset).
Presets provide physically meaningful starting points for:
- Tutorial / quick-start (small grid, fast)
- PF-1000 (IPPLM Warsaw, 1 MJ)
- NX2 (NIE Singapore, 3 kJ)
- LLNL-DPF (Livermore, 100 kJ)

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
        "grid_shape": [128, 1, 256],
        "dx": 7.5e-4,
        "sim_time": 5e-6,
        "dt_init": 1e-10,
        "rho0": 4e-4,
        "T0": 300.0,
        "anomalous_alpha": 0.05,
        "circuit": {
            "C": 1.332e-3,
            "V0": 27e3,
            "L0": 33.5e-9,
            "R0": 2.3e-3,
            "anode_radius": 0.0575,
            "cathode_radius": 0.08,
        },
        "geometry": {"type": "cylindrical"},
        "radiation": {"bremsstrahlung_enabled": True, "fld_enabled": True},
        "sheath": {"enabled": True, "boundary": "z_high"},
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
        "circuit": {
            "C": 28e-6,
            "V0": 14e3,
            "L0": 20e-9,
            "R0": 5e-3,
            "anode_radius": 0.019,
            "cathode_radius": 0.041,
        },
        "geometry": {"type": "cylindrical"},
        "radiation": {"bremsstrahlung_enabled": True},
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
        "circuit": {
            "C": 16e-6,
            "V0": 22e3,
            "L0": 50e-9,
            "R0": 8e-3,
            "anode_radius": 0.008,
            "cathode_radius": 0.015,
        },
        "geometry": {"type": "cylindrical"},
        "radiation": {"bremsstrahlung_enabled": True},
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
            "backend": "python",
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
