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
            "device": "Tutorial Device (UNU-ICTP based)",
            "description": "Small training device. Try changing voltage (15-25 kV) and fill pressure (1-8 Torr) to see how they affect the pinch.",
            "geometry": "cylindrical",
            "topology": "mather",
            "reference": "Lee et al., Am. J. Phys. 56:62 (1988)",
            "learning_notes": [
                "1. Run with defaults first — watch the current rise, peak, then dip (the dip = radial implosion).",
                "2. Increase V0 from 15 to 25 kV — I_peak rises because E = 0.5*C*V^2 stores more energy.",
                "3. Increase fill pressure from 3 to 8 Torr — the sheath slows down, pinch happens later.",
                "4. Try fm=0.05 vs fm=0.20 — lower fm means less mass swept, faster implosion, hotter pinch.",
                "5. Run a parameter sweep on fm to see how neutron yield peaks at an optimal mass fraction.",
            ],
        },
        "grid_shape": [64, 1, 256],
        "dx": 3e-4,
        "sim_time": 5e-6,
        "dt_init": 1e-11,
        "rho0": 6.46e-4,
        "T0": 300.0,
        "anomalous_alpha": 0.03,
        "anomalous_threshold_model": "lhdi",
        "circuit": {
            "C": 30e-6,           # 30 uF
            "V0": 15e3,           # 15 kV
            "L0": 110e-9,         # 110 nH
            "R0": 12e-3,          # 12 mOhm
            "anode_radius": 0.0095,
            "cathode_radius": 0.032,
            "crowbar_enabled": False,
        },
        "geometry": {"type": "cylindrical"},
        "boundary": {"electrode_bc": True},
        "radiation": {"bremsstrahlung_enabled": True},
        "snowplow": {
            "anode_length": 0.16,        # 160 mm
            "fill_pressure_Pa": 400,     # 3 Torr D2 = 400 Pa
            "current_fraction": 0.7,
            "mass_fraction": 0.15,
            "pinch_column_fraction": 1.0,
        },
    },
    "pf1000": {
        "_meta": {
            "description": "PF-1000 (IPPLM Warsaw) — 1 MJ deuterium DPF",
            "device": "PF-1000",
            "geometry": "cylindrical",
            "topology": "mather",
        },
        "grid_shape": [240, 1, 800],
        "dx": 7.5e-4,
        "sim_time": 10e-6,  # 10 us: covers peak (5.8us), radial, pinch, post-pinch
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
            "L0": 33.5e-9,     # 33.5 nH external inductance (RADPF default)
            "R0": 2.3e-3,      # 2.3 mOhm bare-bank short-circuit (Scholz 2006 Table 1) — plasma R enters via sheath, not bank
            "anode_radius": 0.115,   # 115 mm (Scholz 2006)
            "cathode_radius": 0.16,  # 160 mm effective (Lee & Saw 2014)
            "n_cathode_rods": 12,  # UNVERIFIED: cited as 12 (Gribkov 2007) but PDF not read in session
            "crowbar_enabled": True,
            "crowbar_mode": "fixed_time",
            "crowbar_time": 10.5e-6,  # Quarter period of loaded circuit (Scholz 2006: ~10 us)
            "crowbar_resistance": 1.5e-3,  # 1.5 mOhm spark gap (PhD Debate #30)
            "crowbar_inductance": 20e-9,  # 20 nH ignitron arc channel inductance
        },
        "geometry": {"type": "cylindrical"},
        "boundary": {"electrode_bc": True},
        "radiation": {
            "bremsstrahlung_enabled": True,
            "line_radiation_enabled": True,  # Cu impurity line radiation (Six Sigma: activate before calibration)
            "impurity_Z": 29,       # Copper from electrode sputtering
            "impurity_fraction": 0.01,  # EMPIRICAL: 1% Cu impurity
            "fld_enabled": True,
        },
        "sheath": {"enabled": True, "boundary": "z_high"},
        "snowplow": {
            "anode_length": 0.6,  # Scholz (2006) Table 1: 600 mm
            "fill_pressure_Pa": 466.0,  # 3.5 Torr D2 (Scholz 2006) — was defaulting to 400 Pa
            "current_fraction": 0.7,  # RADPF default: fc=0.7
            "mass_fraction": 0.13,  # RADPF default: fm=0.13 (was 0.08 — not a published value)
            "radial_mass_fraction": 0.35,  # RADPF default: fmr=0.35
            "radial_current_fraction_2": 0.45,  # EMPIRICAL: two-step radial (Damideh 2025 method)
            "radial_transition_time": 5.5e-6,  # EMPIRICAL: re-strike onset during radial phase
            "pinch_column_fraction": 0.14,  # Lee & Saw (2014): z_f ~ 84 mm of 600 mm
        },
    },
    "pf1000_akel": {
        "_meta": {
            "description": (
                "PF-1000 (IPPLM Warsaw) — Akel 2021 24-shot validation variant at 16 kV. "
                "All circuit and geometry parameters taken verbatim from Akel et al., "
                "Radiat. Phys. Chem. 188:109633, 2021, Table 1. "
                "L0=25 nH, r0=4-6.5 mOhm (per-shot), V0=16 kV, anode=48 cm. "
                "R0 = 2.3 mOhm per Akel Table 1 (nominal; per-shot r0 overrides in "
                "validation scripts). No empirical correction applied."
            ),
            "device": "PF-1000",
            "geometry": "cylindrical",
            "topology": "mather",
            "reference": "Akel et al., Radiat. Phys. Chem. 188:109633, 2021",
        },
        "grid_shape": [240, 1, 800],
        "dx": 7.5e-4,
        "sim_time": 16e-6,  # 16 us: matches Akel 2021 validation window
        "dt_init": 1e-10,
        "rho0": 2.15e-4,  # 1.05 Torr D2 at 300K (Akel 2021 Table 1); overridden per-shot
        "T0": 300.0,
        "anomalous_alpha": 0.05,
        "anomalous_threshold_model": "lhdi",
        # Circuit: Akel 2021 Table 1 — V0=16 kV, C=1332 uF, L0=25 nH, r0=2.3 mOhm nominal.
        # [KR: radiation-physics-and-chemistry-188-2021-109633.md §Table1 p.4]
        # Per-shot r0 ranges 4.0-6.5 mOhm; validation scripts override R0 per shot.
        "circuit": {
            "C": 1.332e-3,     # 1.332 mF (Akel 2021 Table 1)
            "V0": 16e3,        # 16 kV (Akel 2021 Table 1)
            "L0": 25e-9,       # 25 nH (Akel 2021 Table 1)
            "R0": 2.3e-3,      # 2.3 mOhm nominal (Akel 2021 Table 1)
            "anode_radius": 0.1155,  # 115.5 mm = a=11.55 cm (Akel 2021 Table 1)
            "cathode_radius": 0.16,  # 160 mm (Akel 2021 / Scholz 2006)
            "crowbar_enabled": True,
            "crowbar_mode": "fixed_time",
            "crowbar_time": 10.5e-6,
            "crowbar_resistance": 1.5e-3,
            "crowbar_inductance": 20e-9,
        },
        "geometry": {"type": "cylindrical"},
        "boundary": {"electrode_bc": True},
        "radiation": {"bremsstrahlung_enabled": True, "fld_enabled": True},
        "sheath": {"enabled": True, "boundary": "z_high"},
        "snowplow": {
            "anode_length": 0.48,       # 48 cm = z0 (Akel 2021 Table 1)
            "current_fraction": 0.7,    # fc fixed for all Akel 2021 shots (Table 1)
            "mass_fraction": 0.20,      # fm = 0.20 average (Akel 2021 Table 1)
            "radial_mass_fraction": 0.35,  # fmr: Malek 2025 / engine_validation canonical
            "pinch_column_fraction": 0.14,
        },
    },
    "pf1000_20kv": {
        "_meta": {
            "description": "PF-1000 (IPPLM Warsaw) — 20 kV, 2 Torr D2 operating point",
            "device": "PF-1000",
            "geometry": "cylindrical",
            "topology": "mather",
            "reference": "Akel et al., Radiat. Phys. Chem. 188:109633, 2021 (voltage trend)",
        },
        "grid_shape": [240, 1, 800],
        "dx": 7.5e-4,
        "sim_time": 14e-6,  # 14 us: covers peak (~6.3 us) + post-peak
        "dt_init": 1e-10,
        "rho0": 4.30e-4,  # 2.0 Torr D2 at 300K: P/(kB*T) * m_D2
        "T0": 300.0,
        "anomalous_alpha": 0.05,
        "anomalous_threshold_model": "lhdi",
        # Circuit: Same PF-1000 bank, different operating voltage
        # R0 = 2.3 mOhm baseline (no Akel correction — this is not a per-shot comparison)
        "circuit": {
            "C": 1.332e-3,     # 1.332 mF (same bank)
            "V0": 20e3,        # 20 kV charging voltage
            "L0": 33.5e-9,     # 33.5 nH (same circuit)
            "R0": 2.3e-3,      # 2.3 mOhm baseline (Scholz 2006)
            "anode_radius": 0.115,   # Same geometry
            "cathode_radius": 0.16,  # Same geometry
            "crowbar_enabled": True,
            "crowbar_mode": "fixed_time",
            "crowbar_time": 10.5e-6,  # Same crowbar timing
            "crowbar_resistance": 1.5e-3,  # Same spark gap
            "crowbar_inductance": 20e-9,  # Same ignitron
        },
        "geometry": {"type": "cylindrical"},
        "boundary": {"electrode_bc": True},
        "radiation": {"bremsstrahlung_enabled": True, "fld_enabled": True},
        "sheath": {"enabled": True, "boundary": "z_high"},
        "snowplow": {
            "anode_length": 0.6,  # Same geometry
            # At 20 kV (lower stored energy), lower fill pressure (2 Torr vs 3.5 Torr)
            # means lighter mass loading, similar fc but potentially lower fm
            "current_fraction": 0.7,   # Same as 27 kV (Lee & Saw 2014)
            "mass_fraction": 0.08,     # Same fm as 27 kV (same device geometry)
            "radial_mass_fraction": 0.16,  # Same as 27 kV
            "pinch_column_fraction": 0.14,
        },
    },
    "nx2": {
        "_meta": {
            "description": "NX2 (NIE Singapore) — 1.85 kJ fast miniature DPF",
            "device": "NX2",
            "geometry": "cylindrical",
            "topology": "mather",
            "reference": "Lee & Saw, J. Fusion Energy 27:292 (2008); RADPF Module 1",
            "validation_note": (
                "Published 'experimental' 400 kA is RADPF model output, not Rogowski "
                "measurement (unloaded circuit peak = 402 kA, implying <1% plasma "
                "loading). No digitized waveform exists. Validation uses RADPF-derived "
                "values with reference_only reliability."
            ),
        },
        "grid_shape": [192, 1, 384],
        "dx": 2.5e-4,
        "sim_time": 4e-6,  # 4 us: covers peak (~1.0 us), radial, pinch
        "dt_init": 1e-11,
        "rho0": 6.46e-4,  # 3 Torr D2 at 300K: P/(kB*T) * m_D2
        "T0": 300.0,
        "anomalous_alpha": 0.03,
        "anomalous_threshold_model": "lhdi",
        # Circuit: Lee & Saw, J. Fusion Energy 27:292 (2008); RADPF Module 1
        # V0 = 11.5 kV (Lee & Saw 2008), C = 28 uF, L0 = 20 nH, R0 = 2.3 mOhm
        # Damped peak I_sc * exp(-R0*T/4 / 2L0) = 402 kA ≈ published 400 kA
        "circuit": {
            "C": 28e-6,            # 28 uF — RADPF Module 1 (plasmafocus.net)
            "V0": 11.5e3,          # 11.5 kV — Lee & Saw 2008 Table 1
            "L0": 20e-9,           # 20 nH — RADPF Module 1 (plasmafocus.net)
            "R0": 2.3e-3,          # 2.3 mOhm — RADPF (RESF=0.086)
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
            "mass_fraction": 1.0,    # EMPIRICAL: calibrated to minimize loading vs RADPF 400 kA target
            "radial_mass_fraction": 0.14,  # Arwinder thesis Table 3.34 (C1: fmr=0.14)
            "radial_current_fraction": 0.69,  # Arwinder thesis Table 3.34 (C1: fcr=0.69)
            "pinch_column_fraction": 0.5,  # Small device: larger fraction focuses
        },
    },
    "unu_ictp": {
        "_meta": {
            "description": "UNU-ICTP PFF — 3 kJ deuterium DPF (Lee et al. 1988)",
            "device": "UNU-ICTP",
            "geometry": "cylindrical",
            "topology": "mather",
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
            "current_fraction": 0.7,     # Lee & Saw (2009, 2014): fc=0.7
            "mass_fraction": 0.08,       # Lee & Saw (2014): fm=0.08 (published Lee model fit)
            "radial_mass_fraction": 0.16,  # Lee & Saw (2014): fmr=0.16
            "pinch_column_fraction": 0.06,  # ~1 cm pinch of 16 cm anode
        },
    },
    "llnl_dpf": {
        "_meta": {
            "description": "LLNL compact DPF — 4 kJ diagnostic device",
            "device": "LLNL-DPF",
            "geometry": "cylindrical",
            "topology": "mather",
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
            "description": "MJOLNIR (LLNL) — 2 MJ configuration, MA-class deuterium DPF at 60 kV",
            "device": "MJOLNIR",
            "geometry": "cylindrical",
            "topology": "mather",
            "reference": (
                "Schmidt et al., IEEE TPS (2021) DOI: 10.1109/TPS.2021.3106313; "
                "Goyon et al., Phys. Plasmas 32:033105 (2025); "
                "Petrov et al., Phys. Plasmas 29:062708 (2022)"
            ),
        },
        "grid_shape": [128, 1, 256],
        "dx": 1e-3,
        "sim_time": 14e-6,  # 14 us: covers peak (~5 us), radial, pinch, post-pinch
        "dt_init": 1e-10,
        "rho0": 6e-4,  # ~7 Torr D2 fill
        "T0": 300.0,
        "anomalous_alpha": 0.05,
        "anomalous_threshold_model": "lhdi",
        # Circuit: 2 MJ configuration (6 towers, 24 Marx modules)
        # Goyon et al. 2025: 60 kV typical, 2.8 MA peak current
        # C = 408 uF (24 modules x 2 x 34 uF caps, single-stage erection)
        # L0 = 67.4 nH (measured lumped circuit; Offermann 2021)
        # R0 = 6.25 mOhm (EMPIRICAL: Offermann 2021 gives 12.5 mOhm/tower;
        #       /2 not /6 — includes Marx erection losses + cable impedance)
        # Anode OD = 228.6 mm -> radius 114.3 mm (Goyon 2025)
        # Cathode: 24 rods, inner radius ~157 mm (4.3 cm A-K gap; Petrov 2022)
        "circuit": {
            "C": 408e-6,           # 408 uF — 2 MJ config (Goyon 2025)
            "V0": 60e3,            # 60 kV typical operation (Goyon 2025)
            "L0": 67.4e-9,         # 67.4 nH — measured lumped circuit (Offermann 2021)
            "R0": 6.25e-3,         # 6.25 mOhm — 12.5/2 for 6-tower (Offermann 2021)
            "anode_radius": 0.1143,  # 114.3 mm — 228.6 mm OD / 2 (Goyon 2025)
            "cathode_radius": 0.157,  # ~157 mm — 24-rod cathode (Petrov 2022)
            "n_cathode_rods": 24,  # Goyon 2025 p.2: "a set of 24 cathode rods"
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
            "current_fraction": 0.70,  # EMPIRICAL: standard fc for Mather-type
            "mass_fraction": 1.0,     # EMPIRICAL: calibrated to 2.8 MA at 60 kV (Goyon 2025)
            "radial_mass_fraction": 0.1,
            "pinch_column_fraction": 0.14,  # MA-class geometry: ~14% per Lee & Saw
        },
    },
    "faeton": {
        "_meta": {
            "description": "FAETON-I (Fuse Energy) — 125 kJ, 100 kV, ~1 MA DPF",
            "device": "FAETON-I",
            "geometry": "cylindrical",
            "topology": "mather",
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
            "L0": 220e-9,          # 220 nH static inductance (Damideh 2025)
            "R0": 7.6e-3,          # 7.6 mOhm (estimated from damping, Damideh 2025)
            "anode_radius": 0.05,  # 50 mm (Damideh 2025)
            "cathode_radius": 0.106,  # 106 mm (Damideh 2025 Table 1: 10.6 cm)
            "crowbar_enabled": False,
        },
        "geometry": {"type": "cylindrical"},
        "boundary": {"electrode_bc": True},
        "radiation": {"bremsstrahlung_enabled": True},
        "snowplow": {
            "anode_length": 0.17,      # 170 mm (Damideh 2025)
            "fill_pressure_Pa": 1600.0,  # 12 Torr D2 = 1600 Pa
            "current_fraction": 0.7,   # Lee model fit (Damideh 2025, Lee co-author)
            "mass_fraction": 0.70,     # Lee model fit: fm=0.70 (Damideh 2025)
            "radial_mass_fraction": 0.1,
            "pinch_column_fraction": 0.14,
            # Two-step radial model (Damideh et al. 2025, FFV5-2 Lee code)
            # Re-strikes occur during radial compression, after rundown (~6.2 us).
            # Transition at ~7.0 us (0.8 us into radial phase) matches observed
            # current slope deviation in Damideh 2025 Fig. 4.
            "radial_current_fraction": 0.8,     # f_cr pre-re-strike (Table 3 avg)
            "radial_current_fraction_2": 0.5,   # f_cr2 post-re-strike (Table 3 avg)
            "radial_transition_time": 7.0e-6,   # ~7.0 us: onset of re-strike during radial
        },
    },
    "poseidon": {
        "_meta": {
            "description": "POSEIDON (IPF Stuttgart) — 480 kJ MA-class deuterium DPF",
            "device": "POSEIDON",
            "geometry": "cylindrical",
            "topology": "mather",
            "reference": "Herold et al., Nucl. Fusion 29:33 (1989); Lee & Saw (2014)",
        },
        "grid_shape": [140, 1, 480],
        "dx": 1e-3,
        "sim_time": 12e-6,  # 12 us: covers peak (~5 us) + radial + post-pinch
        "dt_init": 1e-10,
        "rho0": 7.53e-4,  # 3.5 Torr D2 at 300K
        "T0": 300.0,
        "anomalous_alpha": 0.05,
        "anomalous_threshold_model": "lhdi",
        "circuit": {
            "C": 450e-6,           # 450 uF
            "V0": 40e3,            # 40 kV typical
            "L0": 35e-9,           # 35 nH — fitted (was 20nH estimate; Herold 1989 doesn't state L0)
            "R0": 2e-3,            # ~2 mOhm
            "anode_radius": 0.104,
            "cathode_radius": 0.135,
            "crowbar_enabled": True,
            "crowbar_mode": "voltage_zero",
            "crowbar_resistance": 1.5e-3,  # spark gap (removed 30nH inductance, uncited)
        },
        "geometry": {"type": "cylindrical"},
        "boundary": {"electrode_bc": True},
        "radiation": {"bremsstrahlung_enabled": True, "fld_enabled": True},
        "sheath": {"enabled": True, "boundary": "z_high"},
        "snowplow": {
            "anode_length": 0.47,
            "current_fraction": 0.65,
            "mass_fraction": 0.30,     # calibrated (was 0.15, fitted with L0=35nH)
            "radial_mass_fraction": 0.1,
            "pinch_column_fraction": 0.14,
        },
    },
    "poseidon_60kv": {
        "_meta": {
            "description": "POSEIDON (IPF Stuttgart) — 280.8 kJ at 60 kV, IPFS digitized I(t)",
            "device": "POSEIDON-60kV",
            "geometry": "cylindrical",
            "topology": "mather",
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
            "current_fraction": 0.450,   # Calibrated: I_peak 3.175 MA (-0.47% vs 3.19 MA exp)
            "mass_fraction": 0.450,      # Calibrated: Yn 7.24e10 (0.14 dec vs 1e11 exp)
            "radial_mass_fraction": 0.45,  # IPFS Lee model fit (fmr)
            "pinch_column_fraction": 0.14,
        },
    },
    "aecs_pf2": {
        "_meta": {
            "description": (
                "AECS-PF2 (Atomic Energy Commission of Syria) — 2.8 kJ high-impedance "
                "deuterium DPF. Small Mather-type device used for neutron production "
                "and Lee model benchmarking. Published in AAAPT (Asian African "
                "Association for Plasma Training) device survey."
            ),
            "device": "AECS-PF2",
            "geometry": "cylindrical",
            "topology": "mather",
            "reference": "Lee & Saw, AAAPT device survey; Lee (2014) Review",
        },
        "grid_shape": [64, 1, 192],
        "dx": 2.5e-4,
        "sim_time": 4e-6,   # 4 us: covers T/4 (~1.7 us) + radial + pinch
        "dt_init": 1e-11,
        "rho0": 4.30e-4,    # 2 Torr D2 at 300K: P/(kB*T) * m_D2
        "T0": 300.0,
        "anomalous_alpha": 0.03,
        "anomalous_threshold_model": "lhdi",
        # Circuit: Lee & Saw AAAPT device survey
        # RESF = R0/sqrt(L0/C0) = 30e-3/sqrt(110e-9/25e-6) = 1.27 (overdamped!)
        # High impedance is characteristic of small Syrian/AAAPT devices
        "circuit": {
            "C": 25e-6,           # 25 uF
            "V0": 15e3,           # 15 kV (E = 0.5 * 25e-6 * 15e3^2 = 2.8 kJ)
            "L0": 110e-9,         # 110 nH
            "R0": 30e-3,          # 30 mOhm (high impedance, small device)
            "anode_radius": 0.0095,   # 9.5 mm
            "cathode_radius": 0.032,  # 32 mm
            "crowbar_enabled": False,
        },
        "geometry": {"type": "cylindrical"},
        "boundary": {"electrode_bc": True},
        "radiation": {"bremsstrahlung_enabled": True},
        "snowplow": {
            "anode_length": 0.16,        # 160 mm
            "fill_pressure_Pa": 267.0,   # 2 Torr D2 = 267 Pa (midpoint of 1-4 Torr range)
            "current_fraction": 0.7,     # Lee & Saw AAAPT fit: fc=0.7
            "mass_fraction": 0.15,       # Lee & Saw AAAPT fit: fm=0.15
            "pinch_column_fraction": 1.0,
        },
    },
    "pf400j": {
        "_meta": {
            "description": (
                "PF-400J (CCHEN Chile) — 400 J portable DPF. Ultra-compact device "
                "used for neutron production research and radiation biology studies. "
                "Fastest DPF in operation (T/4 ~ 300 ns)."
            ),
            "device": "PF-400J",
            "geometry": "cylindrical",
            "topology": "mather",
            "reference": "Soto et al., Plasma Sources Sci. Technol. 18:015007 (2009)",
        },
        "grid_shape": [32, 1, 64],
        "dx": 3e-4,
        "sim_time": 1.5e-6,  # 1.5 us — covers ~5 quarter periods
        "dt_init": 1e-12,    # Very small: T/4 ~ 300 ns
        "rho0": 1.48e-3,     # 9 mbar D2 at 300K: n*m_D2
        "T0": 300.0,
        "anomalous_alpha": 0.03,
        "anomalous_threshold_model": "lhdi",
        "circuit": {
            "C": 0.95e-6,       # 0.95 uF — Arwinder thesis Table 3.15
            "V0": 28e3,         # 28 kV — Silva et al. APL 2003
            "L0": 40e-9,        # 40 nH — Arwinder thesis Table 3.15
            "R0": 10e-3,        # 10 mOhm — Arwinder thesis Table 3.15
            "anode_radius": 0.006,    # 6 mm — Silva et al. 2003
            "cathode_radius": 0.0155, # 15.5 mm — Silva et al. 2003
            "crowbar_enabled": False,
        },
        "geometry": {"type": "cylindrical"},
        "boundary": {"electrode_bc": True},
        "radiation": {"bremsstrahlung_enabled": True},
        "snowplow": {
            "anode_length": 0.017,       # 17 mm effective — Arwinder thesis
            "fill_pressure_Pa": 900,     # 9 mbar = 900 Pa (optimal per Silva 2003)
            "current_fraction": 0.7,     # Arwinder thesis Table 3.15
            "mass_fraction": 0.08,       # Arwinder thesis Table 3.15
            "radial_mass_fraction": 0.11,  # Arwinder thesis Table 3.15
            "radial_current_fraction": 0.71,  # Arwinder thesis Table 3.15
            "pinch_column_fraction": 1.0,
        },
        "breakdown": {
            "gas_species": "D2",
            "insulator_length": 0.021,   # 21 mm alumina insulator
        },
    },
    "custom": {
        "grid_shape": [32, 1, 64],
        "dx": 6e-4,
        "sim_time": 5e-6,
        "dt_init": 1e-11,
        "rho0": 6.5e-4,
        "T0": 300.0,
        "circuit": {
            "C": 30e-6,        # 30 uF (generic small device)
            "V0": 15000.0,     # 15 kV
            "L0": 50e-9,       # 50 nH
            "R0": 10e-3,       # 10 mOhm
            "anode_radius": 0.01,    # 10 mm
            "cathode_radius": 0.03,  # 30 mm
            "crowbar_enabled": False,
        },
        "geometry": {"type": "cylindrical"},
        "snowplow": {
            "anode_length": 0.15,         # 150 mm
            "fill_pressure_Pa": 400,
            "current_fraction": 0.7,
            "mass_fraction": 0.15,
            "pinch_column_fraction": 1.0,
        },
        "diagnostics": {"hdf5_filename": ":memory:"},
        "_meta": {
            "device": "Custom Device",
            "description": "Blank-slate device — modify all parameters to match your design",
            "topology": "mather",
            "geometry": "cylindrical",
        },
    },
    "cartesian_demo": {
        "_meta": {
            "description": "32^3 Cartesian demo — all physics enabled",
            "device": "Generic",
            "geometry": "cartesian",
            "topology": "cartesian_test",
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
            "topology": "cartesian_test",
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
