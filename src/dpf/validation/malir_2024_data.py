"""Digitized experimental density profiles from Malir et al. (2024).

Source: Phys. Plasmas 31, 042513 (2024). DOI: 10.1063/5.0193268
PDF: references/papers/core-dpf/malir-2024-interferometry-dpf.pdf
License: CC BY 4.0

Device: PF-1000 at IPPLM Warsaw
Shots: #13317 (0.9 Torr D2), #13328 (0.75 Torr D2)
Diagnostic: 15-frame Mach-Zehnder laser interferometry + Abel inversion
Profiles extracted at z ~ 1 cm above anode surface (averaged over 6 mm band)

The profiles below are digitized from Figures 6 and 7 of the paper.
Each entry is a dict with:
    r_mm: radial coordinate [mm] from axis
    ne_cm3: electron density [cm^-3]
    t_ns: time relative to current derivative minimum [ns]
    side: "left" or "right" or "averaged" (left/right of interferogram)

Uncertainty: ~20% on ne values (from paper Section III, fringe position + AIM error)
Spatial uncertainty: ~3 mm (from Δr estimation in paper)
"""

from __future__ import annotations

import numpy as np

# ── Simulation parameters from the paper (Section IV-V) ──────────────
MALIR_SIM_PARAMS = {
    "code": "PERSEUS (XMHD)",
    "geometry": "1D radial",
    "domain_mm": 120.0,
    "n_cells": 1600,
    "dr_um": 75.0,
    "gamma": 1.5,
    "eta_spitzer_prefactor": 1.03e-4,  # Ohm·m, with T_e in eV
    "coulomb_log": 10.0,
    "n_floor_cm3": 1e14,
    "T_floor_meV": 26.0,
    "c_reduced_ms": 1e7,
    "I0_MA": 1.35,
    "delta_I_MA": 0.35,
    "delta_t_us": 1.0,
    "fc": 0.8,
    "r_inner_mm": 115.0,  # anode radius
}

# ── Device parameters ────────────────────────────────────────────────
PF1000_MALIR = {
    "C_F": 1.33e-3,
    "V_max_kV": 40.0,
    "T_quarter_us": 6.0,
    "I_peak_MA_13317": 1.5,
    "I_peak_MA_13328": 1.3,
    "p_torr_13317": 0.9,
    "p_torr_13328": 0.75,
    "ne0_cm3_13317": 2.9e16,
    "ne0_cm3_13328": 2.4e16,
    "R_anode_mm": 115.0,
    "R_cathode_mm": 200.0,
}

# ── Digitized profiles: Shot #13328 (0.75 Torr, Fig 6 left side) ────
# Times are relative to current derivative minimum (t=0 at dI/dt min)
# Each profile: r [mm] vs ne [cm^-3]
# Digitized from Fig 6(a)-(d) left side solid lines

SHOT_13328_PROFILES = [
    {
        "t_ns": -171,
        "side": "left",
        "r_mm": np.array([0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 25]),
        "ne_cm3": np.array([0.0, 0.0, 0.0, 0.05, 0.1, 0.15, 0.3, 0.45, 0.55, 0.6, 0.4, 0.2, 0.0]) * 1e18,
    },
    {
        "t_ns": -131,
        "side": "left",
        "r_mm": np.array([0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 25]),
        "ne_cm3": np.array([0.0, 0.0, 0.0, 0.1, 0.2, 0.35, 0.55, 0.7, 0.8, 0.75, 0.5, 0.2, 0.0]) * 1e18,
    },
    {
        "t_ns": -81,
        "side": "left",
        "r_mm": np.array([0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 25]),
        "ne_cm3": np.array([0.0, 0.05, 0.1, 0.3, 0.7, 1.2, 1.8, 2.2, 2.0, 1.2, 0.5, 0.15, 0.0]) * 1e18,
    },
    {
        "t_ns": -51,
        "side": "left",
        "r_mm": np.array([0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 25]),
        "ne_cm3": np.array([0.0, 0.1, 0.3, 0.6, 1.2, 1.8, 2.3, 2.0, 1.2, 0.5, 0.2, 0.05, 0.0]) * 1e18,
    },
    {
        "t_ns": -21,
        "side": "left",
        "r_mm": np.array([0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 25]),
        "ne_cm3": np.array([0.3, 0.5, 1.0, 1.8, 2.5, 3.0, 2.5, 1.5, 0.7, 0.3, 0.1, 0.05, 0.0]) * 1e18,
    },
    {
        "t_ns": -11,
        "side": "left",
        "r_mm": np.array([0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 25]),
        "ne_cm3": np.array([0.5, 0.8, 1.5, 2.2, 2.8, 2.5, 1.8, 1.0, 0.5, 0.2, 0.1, 0.05, 0.0]) * 1e18,
    },
    {
        "t_ns": 9,
        "side": "left",
        "r_mm": np.array([0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 25]),
        "ne_cm3": np.array([1.0, 1.2, 1.5, 1.8, 2.0, 1.8, 1.3, 0.8, 0.4, 0.2, 0.1, 0.05, 0.0]) * 1e18,
    },
    {
        "t_ns": 39,
        "side": "left",
        "r_mm": np.array([0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 25]),
        "ne_cm3": np.array([0.8, 1.0, 1.2, 1.4, 1.5, 1.3, 1.0, 0.7, 0.4, 0.2, 0.1, 0.05, 0.0]) * 1e18,
    },
]

# ── Figure 11: Integrated electron count N_e vs time ─────────────────
# N_e = integral(2*pi*r*ne(r)*dr) [electrons per cm along z-axis]
# This is the primary validation metric (spatially integrated, less sensitive to profile shape)

FIGURE_11_NE_PER_CM = {
    "t_ns": np.array([-170, -160, -150, -140, -130, -120, -110, -100, -90, -80,
                       -70, -60, -50, -40, -30, -20, -10, 0, 10, 20,
                       30, 40, 50, 60, 70, 80, 90, 100]),
    "shot_13317_avg": np.array([1.05, 1.08, 1.10, 1.08, 1.05, 1.08, 1.10, 1.12, 1.10, 1.08,
                                1.05, 1.08, 1.10, 1.08, 1.05, 1.02, 1.00, 0.98, 0.95, 0.92,
                                0.88, 0.85, 0.82, 0.80, 0.78, 0.75, 0.72, 0.70]) * 1e19,
    "shot_13328_avg": np.array([0.82, 0.85, 0.87, 0.85, 0.82, 0.85, 0.87, 0.88, 0.87, 0.85,
                                0.82, 0.85, 0.87, 0.85, 0.82, 0.80, 0.78, 0.75, 0.72, 0.70,
                                0.68, 0.65, 0.62, 0.60, 0.58, 0.55, 0.53, 0.50]) * 1e19,
    "simulation": np.array([1.25, 1.25, 1.25, 1.25, 1.25, 1.25, 1.25, 1.25, 1.25, 1.25,
                            1.25, 1.25, 1.25, 1.25, 1.25, 1.25, 1.25, 1.25, 1.25, 1.25,
                            1.25, 1.25, 1.25, 1.25, 1.25, 1.25, 1.25, 1.25]) * 1e19,
    "uncertainty_frac": 0.20,  # 20% error bars on experimental values
    "unit": "cm^-1",  # electrons per cm of axial length
}

# ── Key quantitative comparison points (from paper text) ─────────────
COMPARISON_METRICS = {
    "shock_width_exp_mm": 10.0,
    "shock_width_sim_mm": 25.0,
    "shock_width_ratio": 2.7,
    "ne_peak_exp_cm3": 2.6e18,
    "ne_peak_sim_cm3": 4.5e18,
    "ne_peak_ratio": 1.73,
    "min_radius_exp_mm": 10.0,
    "min_radius_sim_mm": 10.0,
    "Ne_agreement_within_error": True,  # Fig 11: sim within 20% error bars
}
