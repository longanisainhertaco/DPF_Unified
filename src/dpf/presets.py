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

import copy
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
            "crowbar_mode": "fixed_time",
            "crowbar_time": 10.5e-6,  # Quarter period of loaded circuit (Scholz 2006: ~10 us)
            "crowbar_resistance": 1.5e-3,  # 1.5 mOhm spark gap (PhD Debate #30)
        },
        "geometry": {"type": "cylindrical"},
        "boundary": {"electrode_bc": True},
        "radiation": {"bremsstrahlung_enabled": True, "fld_enabled": True},
        "sheath": {"enabled": True, "boundary": "z_high"},
        "snowplow": {
            "anode_length": 0.6,  # Scholz (2006) Table 1: 600 mm
            "current_fraction": 0.800,  # Phase AR: correct bounds (0.6,0.8) + crowbar R=1.5mOhm
            "mass_fraction": 0.094,  # Phase AR: recalibrated (was 0.142 with incorrect bounds)
            "radial_mass_fraction": 0.1,  # Lee & Saw (2014): f_mr ~ 0.07-0.12
            "pinch_column_fraction": 0.14,  # Lee & Saw (2014): z_f ~ 84 mm of 600 mm
        },
    },
    "nx2": {
        "_meta": {
            "description": "NX2 (NIE Singapore) — 1.85 kJ fast miniature DPF",
            "device": "NX2",
            "geometry": "cylindrical",
            "reference": "Lee & Saw, J. Fusion Energy 27:292 (2008); RADPF Module 1",
        },
        "grid_shape": [192, 1, 384],
        "dx": 2.5e-4,
        "sim_time": 1e-6,
        "dt_init": 1e-11,
        "rho0": 6.46e-4,  # 3 Torr D2 at 300K: P/(kB*T) * m_D2
        "T0": 300.0,
        "anomalous_alpha": 0.03,
        "anomalous_threshold_model": "lhdi",
        # Circuit: Lee & Saw (2008), RADPF Module 1 (plasmafocus.net)
        # r0 = 2.3 mOhm from RADPF preset (RESF=0.1, L0=20 nH, C0=28 uF)
        "circuit": {
            "C": 28e-6,
            "V0": 11e3,           # 11 kV operating voltage (Lee & Saw 2008)
            "L0": 20e-9,          # 20 nH (RADPF Module 1)
            "R0": 2.3e-3,         # 2.3 mOhm (RADPF; actual RESF=0.086)
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
            "fill_pressure_Pa": 400.0,  # 3 Torr D2 = 400 Pa
            "current_fraction": 0.7,  # Lee & Saw (2008); Lee et al. (2009)
            "mass_fraction": 0.1,  # Lee et al., J. Appl. Phys. 106 (2009)
            "radial_mass_fraction": 0.12,  # Lee et al. (2009): fmr=0.12
            "pinch_column_fraction": 0.5,  # Small device: larger fraction focuses
        },
    },
    "unu_ictp": {
        "_meta": {
            "description": "UNU-ICTP PFF — 3 kJ deuterium DPF (Lee et al. 1988)",
            "device": "UNU-ICTP",
            "geometry": "cylindrical",
            "reference": "Lee et al., Am. J. Phys. 56:62 (1988); Lee (2014) Review",
        },
        "grid_shape": [64, 1, 256],
        "dx": 3e-4,
        "sim_time": 5e-6,
        "dt_init": 1e-11,
        "rho0": 6.46e-4,  # 3 Torr D2 at 300K: P/(kB*T) * m_D2
        "T0": 300.0,
        "anomalous_alpha": 0.03,
        "anomalous_threshold_model": "lhdi",
        # Circuit: Lee et al. (1988), Lee (2014) Review
        # RESF = r0/sqrt(L0/C0) = 12e-3/sqrt(110e-9/30e-6) = 0.198
        "circuit": {
            "C": 30e-6,           # 30 uF
            "V0": 14e3,           # 14 kV
            "L0": 110e-9,         # 110 nH
            "R0": 12e-3,          # 12 mOhm (RESF~0.2)
            "anode_radius": 0.0095,
            "cathode_radius": 0.032,
            "crowbar_enabled": False,  # No crowbar in UNU-ICTP PFF (simple capacitor bank)
        },
        "geometry": {"type": "cylindrical"},
        "boundary": {"electrode_bc": True},
        "radiation": {"bremsstrahlung_enabled": True},
        "snowplow": {
            "anode_length": 0.16,        # 160 mm
            "fill_pressure_Pa": 400.0,   # 3 Torr D2 = 400 Pa
            "current_fraction": 0.7,     # Lee & Saw (2009, 2010)
            "mass_fraction": 0.05,       # Lee & Saw (2009): fm=0.05
            "radial_mass_fraction": 0.2,  # Lee & Saw (2009): fmr=0.2
            "pinch_column_fraction": 0.06,  # ~1 cm pinch of 16 cm anode
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
            "description": "MJOLNIR (LLNL) — 2 MJ MA-class deuterium DPF at 60 kV",
            "device": "MJOLNIR",
            "geometry": "cylindrical",
            "reference": (
                "Schmidt et al., IEEE TPS (2021); "
                "Goyon et al., Phys. Plasmas 32:033105 (2025)"
            ),
        },
        "grid_shape": [128, 1, 256],
        "dx": 1e-3,
        "sim_time": 12e-6,
        "dt_init": 1e-10,
        "rho0": 6e-4,  # ~7 Torr D2 fill
        "T0": 300.0,
        "anomalous_alpha": 0.05,
        "anomalous_threshold_model": "lhdi",
        # Circuit: ATLAS-heritage Marx, 24 modules, 2 x 34 uF each
        # C_erected = 24 x 17 uF = 408 uF. Design V=100 kV (2 MJ).
        # Typical operation: 60 kV (734 kJ), peak current 2.8 MA.
        # L0 = 80 nH estimated from I_peak/I_sc ~ 0.65 loading factor.
        # 84 flexible transmission line cables from Marx pit to collector.
        # R0 = 1.4 mOhm from RESF ~ 0.1.
        "circuit": {
            "C": 408e-6,           # 408 uF (24 modules x 17 uF erected)
            "V0": 60e3,            # 60 kV typical operation
            "L0": 80e-9,           # ~80 nH (estimated from loading factor)
            "R0": 1.4e-3,          # ~1.4 mOhm (RESF ~ 0.1)
            "anode_radius": 0.1143,  # 114.3 mm (Goyon et al., Phys. Plasmas 32:033105, 2025)
            "cathode_radius": 0.157,  # ~157 mm (A-K gap + anode)
            "crowbar_enabled": True,
            "crowbar_mode": "voltage_zero",
            "crowbar_resistance": 1.5e-3,  # estimated spark gap
        },
        "geometry": {"type": "cylindrical"},
        "boundary": {"electrode_bc": True},
        "radiation": {"bremsstrahlung_enabled": True, "fld_enabled": True},
        "sheath": {"enabled": True, "boundary": "z_high"},
        # Anode effective length 18.3-22.1 cm (Petrov 2022)
        "snowplow": {
            "anode_length": 0.20,  # 200 mm (midpoint of Petrov 2022 range)
            "current_fraction": 0.55,  # Fitted: 3.6% I_peak, 2.3% t_peak vs Schmidt (2021)
            "mass_fraction": 0.45,    # Fitted: heavy mass loading for MA-class (ATLAS-heritage)
            "radial_mass_fraction": 0.1,
            "pinch_column_fraction": 0.14,  # MA-class geometry: ~14% per Lee & Saw
        },
    },
    "faeton": {
        "_meta": {
            "description": "FAETON-I (Fuse Energy) — 125 kJ, 100 kV, ~1 MA DPF",
            "device": "FAETON-I",
            "geometry": "cylindrical",
            "reference": "Damideh et al., Sci. Rep. 15:23048 (2025)",
        },
        "grid_shape": [64, 1, 192],
        "dx": 1.5e-3,
        "sim_time": 8e-6,
        "dt_init": 1e-10,
        "rho0": 1.29e-3,  # 12 Torr D2 at 300K: P/(kB*T) * m_D2
        "T0": 300.0,
        "anomalous_alpha": 0.03,
        "anomalous_threshold_model": "lhdi",
        # Circuit: 5 x 5 uF capacitors = 25 uF, 100 kV direct-charge
        # L0 = 220 nH static (Damideh 2025), but effective dynamic L ~40 nH
        #       (back-calculated from published t_peak = 1.2 us)
        # R0 = 35 mOhm (fitted to published I_peak = 1.1 MA)
        # No crowbar switch
        "circuit": {
            "C": 25e-6,            # 25 uF (5 x 5 uF)
            "V0": 100e3,           # 100 kV direct-charge
            "L0": 40e-9,           # 40 nH effective dynamic inductance
            "R0": 35e-3,           # 35 mOhm (includes Marx switch losses)
            "anode_radius": 0.05,  # 50 mm (Damideh 2025)
            "cathode_radius": 0.10,  # ~100 mm (estimated, 5 cm A-K gap)
            "crowbar_enabled": False,
        },
        "geometry": {"type": "cylindrical"},
        "boundary": {"electrode_bc": True},
        "radiation": {"bremsstrahlung_enabled": True},
        "snowplow": {
            "anode_length": 0.17,      # 170 mm (Damideh 2025)
            "fill_pressure_Pa": 1600.0,  # 12 Torr D2 = 1600 Pa
            "current_fraction": 0.7,   # Starting estimate (Sing Lee is co-author)
            "mass_fraction": 0.1,      # Starting estimate
            "radial_mass_fraction": 0.1,
            "pinch_column_fraction": 0.14,
        },
    },
    "poseidon": {
        "_meta": {
            "description": "POSEIDON (IPF Stuttgart) — 480 kJ MA-class deuterium DPF",
            "device": "POSEIDON",
            "geometry": "cylindrical",
            "reference": "Herold et al., Nucl. Fusion 29:33 (1989); Lee & Saw (2014)",
        },
        "grid_shape": [140, 1, 480],
        "dx": 1e-3,
        "sim_time": 8e-6,
        "dt_init": 1e-10,
        "rho0": 7.53e-4,  # 3.5 Torr D2 at 300K
        "T0": 300.0,
        "anomalous_alpha": 0.05,
        "anomalous_threshold_model": "lhdi",
        "circuit": {
            "C": 450e-6,           # 450 uF
            "V0": 40e3,            # 40 kV typical
            "L0": 20e-9,           # 20 nH (MA-class low-inductance)
            "R0": 2e-3,            # ~2 mOhm
            "anode_radius": 0.104,
            "cathode_radius": 0.135,
            "crowbar_enabled": True,
            "crowbar_mode": "voltage_zero",
        },
        "geometry": {"type": "cylindrical"},
        "boundary": {"electrode_bc": True},
        "radiation": {"bremsstrahlung_enabled": True, "fld_enabled": True},
        "sheath": {"enabled": True, "boundary": "z_high"},
        "snowplow": {
            "anode_length": 0.47,
            "current_fraction": 0.75,  # Fitted: 1.3% I_peak, 0.14 Yn_log vs Herold (1989)
            "mass_fraction": 0.05,     # Low fm for narrow A-K gap (b/a=1.30)
            "radial_mass_fraction": 0.1,
            "pinch_column_fraction": 0.14,
        },
    },
    "poseidon_60kv": {
        "_meta": {
            "description": "POSEIDON (IPF Stuttgart) — 280.8 kJ at 60 kV, IPFS digitized I(t)",
            "device": "POSEIDON-60kV",
            "geometry": "cylindrical",
            "reference": "IPFS (plasmafocus.net); Herold et al., Nucl. Fusion 29:33 (1989)",
        },
        "grid_shape": [96, 1, 300],
        "dx": 1e-3,
        "sim_time": 6e-6,
        "dt_init": 1e-10,
        "rho0": 8.18e-4,  # 3.8 Torr D2 at 300K
        "T0": 300.0,
        "anomalous_alpha": 0.05,
        "anomalous_threshold_model": "lhdi",
        "circuit": {
            "C": 156e-6,           # 156 uF
            "V0": 60e3,            # 60 kV
            "L0": 17.7e-9,         # 17.7 nH (Lee model fitted)
            "R0": 1.7e-3,          # 1.7 mOhm
            "anode_radius": 0.0655,
            "cathode_radius": 0.095,
            "crowbar_enabled": True,
            "crowbar_mode": "voltage_zero",
            "crowbar_resistance": 1.5e-3,
        },
        "geometry": {"type": "cylindrical"},
        "boundary": {"electrode_bc": True},
        "radiation": {"bremsstrahlung_enabled": True, "fld_enabled": True},
        "sheath": {"enabled": True, "boundary": "z_high"},
        "snowplow": {
            "anode_length": 0.30,
            "current_fraction": 0.595,   # IPFS Lee model fit (plasmafocus.net)
            "mass_fraction": 0.275,      # IPFS Lee model fit
            "radial_mass_fraction": 0.45,  # IPFS Lee model fit (fmr)
            "pinch_column_fraction": 0.14,
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
    preset = copy.deepcopy(_PRESETS[name])
    preset.pop("_meta", None)
    return preset


def get_preset_names() -> list[str]:
    """Return list of all preset names."""
    return list(_PRESETS.keys())
