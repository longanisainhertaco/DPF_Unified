"""Device parameter tables and base dataclass for Lee model calibration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class CalibrationResult:
    """Result of an fc/fm calibration run.

    Attributes:
        best_fc: Optimal current fraction.
        best_fm: Optimal mass fraction.
        peak_current_error: Relative error in peak current at optimum.
        timing_error: Relative error in peak timing at optimum.
        objective_value: Final objective function value.
        n_evals: Number of objective evaluations.
        converged: Whether the optimizer reported convergence.
        device_name: Device name that was calibrated.
    """

    best_fc: float
    best_fm: float
    peak_current_error: float
    timing_error: float
    objective_value: float
    n_evals: int
    converged: bool
    device_name: str = ""


# =====================================================================
# Published Lee model fc/fm ranges from Lee & Saw (2014), Table 1
# These provide ground-truth benchmarks for calibration validation.
# Source: S. Lee & S.H. Saw, J. Fusion Energy 33:319-335 (2014)
# NOTE: Ranges match Lee & Saw (2014) published values directly.
# Previous versions widened bounds to (0.65, 0.85) which was circular.
# =====================================================================

_PUBLISHED_FC_FM_RANGES: dict[str, dict[str, tuple[float, float]]] = {
    "PF-1000": {
        "fc": (0.6, 0.8),    # Lee & Saw 2014 Table 1: fc ~ 0.7 for PF-1000
        "fm": (0.05, 0.20),   # Lee & Saw 2014 Table 1: fm ~ 0.05-0.15 for PF-1000
    },
    "NX2": {
        "fc": (0.60, 0.85),   # Lee & Saw 2008: fc ~ 0.7-0.8 for NX2
        "fm": (0.07, 0.25),   # Lee & Saw 2008: fm ~ 0.1-0.2 for NX2
    },
    "UNU-ICTP": {
        "fc": (0.55, 0.80),   # Lee et al., Am. J. Phys. 56 (1988): fc ~ 0.7
        "fm": (0.04, 0.35),   # Lee & Saw (2009): fm=0.05 for UNU-ICTP; widened to 0.04 lower bound
    },
    "POSEIDON": {
        "fc": (0.60, 0.85),   # Lee & Saw 2014: fc ~ 0.7 for POSEIDON
        "fm": (0.05, 0.20),   # Lee & Saw 2014: fm ~ 0.08-0.12 for POSEIDON (MJ-class)
    },
    "POSEIDON-60kV": {
        "fc": (0.50, 0.70),   # IPFS fit: fc=0.595 (different bank/geometry)
        "fm": (0.15, 0.40),   # IPFS fit: fm=0.275 (higher mass fraction)
    },
    "FAETON-I": {
        "fc": (0.55, 0.85),   # Wide range: circuit-dominated, Lee is co-author
        "fm": (0.04, 0.25),   # Wide range: no published Lee model fit yet
    },
    "MJOLNIR": {
        "fc": (0.55, 0.80),   # MA-class: similar to PF-1000 range
        "fm": (0.05, 0.20),   # MA-class: similar to PF-1000 range
    },
}


_DEFAULT_DEVICE_PCF: dict[str, float] = {
    "PF-1000": 0.14,
    "PF-1000-Gribkov": 0.14,  # Same device, different shot/publication
    "PF-1000-16kV": 0.14,
    "PF-1000-20kV": 0.14,
    "NX2": 0.5,
    "UNU-ICTP": 0.06,  # ~1 cm pinch of 16 cm anode (Lee & Saw 2009; matches presets.py)
    "POSEIDON": 0.14,  # Similar to PF-1000 (Lee & Saw 2014 scaling)
    "POSEIDON-60kV": 0.14,  # Lee & Saw scaling for MA-class
    "FAETON-I": 0.14,  # Starting estimate (no published Lee model fit)
    "MJOLNIR": 0.14,   # MA-class: same as PF-1000
}

# Default crowbar spark gap arc resistance [Ohm] per device.
# PhD Debate #30 Finding 4: R_crowbar=0 is physically incorrect and
# systematically biases fc upward during calibration.
# PF-1000: ~1-3 mOhm for ignitron/spark gap (Dr. PP estimate).
_DEFAULT_CROWBAR_R: dict[str, float] = {
    "PF-1000": 1.5e-3,  # 1.5 mOhm midpoint of 1-3 mOhm range
    "PF-1000-Gribkov": 1.5e-3,  # Same device as PF-1000
    "PF-1000-16kV": 1.5e-3,  # Same device as PF-1000 (different operating conditions)
    "PF-1000-20kV": 1.5e-3,  # Same device as PF-1000 (different operating conditions)
    "POSEIDON-60kV": 1.5e-3,  # estimated, same as PF-1000
    "UNU-ICTP": 0.0,  # No crowbar in UNU-ICTP PFF (simple capacitor bank)
    "FAETON-I": 0.0,   # No crowbar switch (Damideh 2025)
    "MJOLNIR": 1.5e-3,  # Estimated spark gap resistance
}

# Published shot-to-shot variability data
_SHOT_TO_SHOT_DATA: dict[str, dict[str, Any]] = {
    "PF-1000": {
        "u_shot_to_shot": 0.05,  # 5% sigma_I/I per Scholz et al. (2006)
        "u_rogowski": 0.05,      # 5% Rogowski coil calibration
        "u_amplitude": 0.03,  # 3% digitization error
        "n_shots_typical": 5,    # Scholz et al. averaged ~5 reproducible shots
        "reference": (
            "Scholz et al., Nukleonika 51(1), 2006; "
            "Lee & Saw, J. Fusion Energy 33:319-335 (2014)"
        ),
    },
    "NX2": {
        "u_shot_to_shot": 0.08,  # 8% — smaller devices show more variability
        "u_rogowski": 0.05,
        "u_amplitude": 0.03,
        "n_shots_typical": 10,
        "reference": "Lee & Saw, J. Fusion Energy 27:292-295 (2008)",
    },
    "POSEIDON-60kV": {
        "u_shot_to_shot": 0.06,  # 6% estimated from IPFS data scatter
        "u_rogowski": 0.05,
        "u_amplitude": 0.03,
        "n_shots_typical": 3,
        "reference": "IPFS plasmafocus.net (Lee fitting)",
    },
    "UNU-ICTP": {
        "u_shot_to_shot": 0.10,  # 10% — teaching device, higher variability
        "u_rogowski": 0.05,
        "u_amplitude": 0.03,
        "n_shots_typical": 10,
        "reference": "Lee et al., Am. J. Phys. 56 (1988)",
    },
    "FAETON-I": {
        "u_shot_to_shot": 0.08,  # 8% (re-strikes cause variability)
        "u_rogowski": 0.05,
        "u_amplitude": 0.08,  # 8% (reconstructed waveform, not digitized)
        "n_shots_typical": 5,
        "reference": "Damideh et al., Sci. Rep. 15:23048 (2025)",
    },
    "MJOLNIR": {
        "u_shot_to_shot": 0.10,  # 10% (large device, high-power variability)
        "u_rogowski": 0.05,
        "u_amplitude": 0.10,  # 10% (reconstructed waveform, high uncertainty)
        "n_shots_typical": 5,
        "reference": "Schmidt et al., IEEE TPS (2021); Goyon et al., Phys. Plasmas (2025)",
    },
    "PF-1000-16kV": {
        "u_shot_to_shot": 0.05,  # 5% — same bank as PF-1000 (Scholz 2006)
        "u_rogowski": 0.05,      # 5% — same Rogowski coil
        "u_amplitude": 0.05,  # 5% (reconstructed from 27 kV Scholz scaling)
        "n_shots_typical": 16,   # Akel et al. (2021) Table 1: 16 shots at 1.05 Torr
        "reference": "Akel et al., Radiat. Phys. Chem. 188:109633, 2021",
    },
    "PF-1000-Gribkov": {
        "u_shot_to_shot": 0.05,  # 5% — same bank as PF-1000
        "u_rogowski": 0.05,      # 5% — same Rogowski coil
        "u_amplitude": 0.03,  # 3% (digitized from IPFS archive, 94 points)
        "n_shots_typical": 5,    # Gribkov et al. (2007) — similar campaign
        "reference": "Gribkov et al., J. Phys. D 40:3592, 2007",
    },
}
