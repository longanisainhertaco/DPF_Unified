"""Validation suite for DPF simulations.

Compares simulation results against experimental data from known DPF devices
(PF-1000, NX2, LLNL-DPF). Provides quantitative scoring via RMSE, relative
error, and pass/fail thresholds.

Usage:
    from dpf.validation.suite import ValidationSuite
    suite = ValidationSuite()
    results = suite.validate(sim_summary, sim_state)
    score = suite.overall_score(results)

Experimental reference data:
    PF-1000: Scholz et al., Nukleonika 51(2):79-84 (2006)
    NX2: Lee & Saw, J. Fusion Energy 27:292-295 (2008)
    LLNL-DPF: Deutsch & Kies, Plasma Phys. Control. Fusion 30:263 (1988)
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════
# Reference Data for Known DPF Devices
# ═══════════════════════════════════════════════════════

@dataclass
class DeviceData:
    """Experimental reference data for a DPF device.

    Attributes:
        name: Device name.
        description: Brief description.
        C: Bank capacitance [F].
        V0: Charging voltage [V].
        L0: External inductance [H].
        R0: External resistance [Ohm].
        anode_radius: Anode radius [m].
        cathode_radius: Cathode radius [m].
        peak_current_A: Measured peak current [A].
        peak_current_time_s: Time of peak current [s].
        pinch_current_A: Current at pinch time [A].
        pinch_time_s: Pinch time [s].
        neutron_yield: Total neutron yield (for DD fill).
        peak_ne: Peak electron density [m^-3].
        peak_Te_eV: Peak electron temperature [eV].
        tolerances: Dict of metric_name -> acceptable relative error.
    """

    name: str
    description: str
    C: float
    V0: float
    L0: float
    R0: float
    anode_radius: float
    cathode_radius: float
    peak_current_A: float
    peak_current_time_s: float
    pinch_current_A: float = 0.0
    pinch_time_s: float = 0.0
    neutron_yield: float = 0.0
    peak_ne: float = 0.0
    peak_Te_eV: float = 0.0
    tolerances: dict[str, float] = field(default_factory=dict)


# PF-1000 (IPPLM Warsaw) — largest DPF in Europe
PF1000 = DeviceData(
    name="PF-1000",
    description="IPPLM Warsaw, 1 MJ bank, deuterium fill",
    C=1.332e-3,        # 1.332 mF (12 caps × 111 uF)
    V0=40e3,            # 40 kV charging
    L0=15e-9,           # 15 nH external inductance
    R0=2.3e-3,          # 2.3 mOhm
    anode_radius=0.058, # 58 mm anode radius
    cathode_radius=0.08, # 80 mm cathode radius
    peak_current_A=2.5e6,     # 2.5 MA peak
    peak_current_time_s=5.5e-6,  # ~5.5 us to peak
    pinch_current_A=1.8e6,    # ~1.8 MA at pinch
    pinch_time_s=6.5e-6,      # ~6.5 us pinch time
    neutron_yield=1e11,        # ~10^11 neutrons/shot
    peak_ne=5e25,              # ~5e25 m^-3
    peak_Te_eV=2000.0,        # ~2 keV
    tolerances={
        "peak_current": 0.15,     # 15% tolerance
        "peak_current_time": 0.20, # 20% tolerance
        "energy_conservation": 0.05, # 5% tolerance
        "neutron_yield": 1.0,     # 1 order of magnitude
        "peak_ne": 0.50,          # 50% tolerance
        "peak_Te_eV": 0.50,       # 50% tolerance
    },
)

# NX2 (NIE Singapore) — compact device, well-characterized
NX2 = DeviceData(
    name="NX2",
    description="NIE Singapore, 3 kJ Mather-type DPF",
    C=28e-6,            # 28 uF
    V0=14e3,            # 14 kV
    L0=110e-9,          # 110 nH
    R0=12e-3,           # 12 mOhm
    anode_radius=0.0095, # 9.5 mm
    cathode_radius=0.0165, # 16.5 mm
    peak_current_A=400e3,     # 400 kA
    peak_current_time_s=1.4e-6,  # 1.4 us
    pinch_current_A=300e3,    # ~300 kA
    pinch_time_s=1.7e-6,      # ~1.7 us
    neutron_yield=1e8,         # ~10^8
    peak_ne=1e25,              # ~10^25 m^-3
    peak_Te_eV=500.0,         # ~500 eV
    tolerances={
        "peak_current": 0.15,
        "peak_current_time": 0.20,
        "energy_conservation": 0.05,
        "neutron_yield": 1.0,
        "peak_ne": 0.50,
        "peak_Te_eV": 0.50,
    },
)

# LLNL-DPF (small diagnostic device)
LLNL_DPF = DeviceData(
    name="LLNL-DPF",
    description="LLNL compact DPF, 4 kJ bank",
    C=16e-6,            # 16 uF
    V0=22e3,            # 22 kV
    L0=50e-9,           # 50 nH
    R0=8e-3,            # 8 mOhm
    anode_radius=0.008,  # 8 mm
    cathode_radius=0.015, # 15 mm
    peak_current_A=250e3,     # 250 kA
    peak_current_time_s=1.0e-6,  # 1 us
    pinch_current_A=180e3,    # ~180 kA
    pinch_time_s=1.3e-6,      # ~1.3 us
    neutron_yield=5e7,         # ~5e7
    peak_ne=5e24,              # ~5e24 m^-3
    peak_Te_eV=300.0,         # ~300 eV
    tolerances={
        "peak_current": 0.20,
        "peak_current_time": 0.25,
        "energy_conservation": 0.05,
        "neutron_yield": 1.0,
        "peak_ne": 0.50,
        "peak_Te_eV": 0.50,
    },
)

# Registry of all known devices
DEVICE_REGISTRY: dict[str, DeviceData] = {
    "PF-1000": PF1000,
    "NX2": NX2,
    "LLNL-DPF": LLNL_DPF,
}


# ═══════════════════════════════════════════════════════
# Scoring Functions
# ═══════════════════════════════════════════════════════

@dataclass
class MetricResult:
    """Result for a single validation metric."""

    name: str
    sim_value: float
    ref_value: float
    relative_error: float
    tolerance: float
    passed: bool
    unit: str = ""


@dataclass
class ValidationResult:
    """Overall validation result for one device."""

    device: str
    metrics: list[MetricResult]
    overall_score: float  # 0.0 = worst, 1.0 = perfect
    passed: bool
    config_hash: str = ""


def normalized_rmse(sim_values: np.ndarray, ref_values: np.ndarray) -> float:
    """Compute normalized RMSE between simulation and reference.

    NRMSE = sqrt(mean((sim - ref)^2)) / (max(ref) - min(ref))

    Returns value in [0, inf). 0 = perfect match.

    Args:
        sim_values: Simulated values.
        ref_values: Reference/experimental values.

    Returns:
        Normalized RMSE (dimensionless).
    """
    if len(sim_values) == 0 or len(ref_values) == 0:
        return float("inf")

    mse = float(np.mean((sim_values - ref_values) ** 2))
    ref_range = float(np.max(ref_values) - np.min(ref_values))

    if ref_range <= 0:
        return float(np.sqrt(mse))

    return float(np.sqrt(mse) / ref_range)


def relative_error(sim: float, ref: float) -> float:
    """Compute relative error |sim - ref| / |ref|.

    Args:
        sim: Simulated value.
        ref: Reference value.

    Returns:
        Relative error (dimensionless, 0 = perfect match).
    """
    if abs(ref) < 1e-30:
        return abs(sim)
    return abs(sim - ref) / abs(ref)


def config_hash(config_dict: dict[str, Any]) -> str:
    """Compute SHA-256 hash of config for reproducibility tracking.

    Args:
        config_dict: Configuration dictionary.

    Returns:
        Hex string of SHA-256 hash.
    """
    config_str = json.dumps(config_dict, sort_keys=True, default=str)
    return hashlib.sha256(config_str.encode()).hexdigest()[:16]


# ═══════════════════════════════════════════════════════
# Validation Suite
# ═══════════════════════════════════════════════════════

class ValidationSuite:
    """DPF validation suite — compares simulations to experimental devices.

    Usage:
        suite = ValidationSuite()
        result = suite.validate_circuit("PF-1000", sim_summary)
        print(f"Score: {result.overall_score:.2%}")
    """

    def __init__(self, devices: list[str] | None = None) -> None:
        """Initialize validation suite.

        Args:
            devices: List of device names to validate against.
                     None = all registered devices.
        """
        if devices is None:
            self.devices = list(DEVICE_REGISTRY.keys())
        else:
            self.devices = devices
            for name in devices:
                if name not in DEVICE_REGISTRY:
                    raise ValueError(
                        f"Unknown device '{name}'. Available: {list(DEVICE_REGISTRY.keys())}"
                    )

    def validate_circuit(
        self,
        device_name: str,
        sim_summary: dict[str, Any],
        config_dict: dict[str, Any] | None = None,
    ) -> ValidationResult:
        """Validate circuit-level results against experimental device data.

        Compares: peak current, peak current timing, energy conservation.

        Args:
            device_name: Name of reference device.
            sim_summary: Simulation summary dict from SimulationEngine.run().
            config_dict: Optional config for hashing.

        Returns:
            ValidationResult with metric scores.
        """
        device = DEVICE_REGISTRY[device_name]
        metrics: list[MetricResult] = []

        # 1. Peak current magnitude
        if "peak_current_A" in sim_summary:
            sim_Ipeak = abs(sim_summary["peak_current_A"])
            ref_Ipeak = device.peak_current_A
            err = relative_error(sim_Ipeak, ref_Ipeak)
            tol = device.tolerances.get("peak_current", 0.20)
            metrics.append(MetricResult(
                name="peak_current",
                sim_value=sim_Ipeak,
                ref_value=ref_Ipeak,
                relative_error=err,
                tolerance=tol,
                passed=err <= tol,
                unit="A",
            ))

        # 2. Peak current timing
        if "peak_current_time_s" in sim_summary:
            sim_t = sim_summary["peak_current_time_s"]
            ref_t = device.peak_current_time_s
            err = relative_error(sim_t, ref_t)
            tol = device.tolerances.get("peak_current_time", 0.25)
            metrics.append(MetricResult(
                name="peak_current_time",
                sim_value=sim_t,
                ref_value=ref_t,
                relative_error=err,
                tolerance=tol,
                passed=err <= tol,
                unit="s",
            ))

        # 3. Energy conservation
        if "energy_conservation" in sim_summary:
            sim_E = sim_summary["energy_conservation"]
            ref_E = 1.0  # Perfect conservation
            err = abs(sim_E - ref_E)
            tol = device.tolerances.get("energy_conservation", 0.05)
            metrics.append(MetricResult(
                name="energy_conservation",
                sim_value=sim_E,
                ref_value=ref_E,
                relative_error=err,
                tolerance=tol,
                passed=err <= tol,
                unit="",
            ))

        # 4. Final current (at sim_time)
        if "final_current_A" in sim_summary:
            metrics.append(MetricResult(
                name="final_current",
                sim_value=abs(sim_summary["final_current_A"]),
                ref_value=0.0,  # Informational only
                relative_error=0.0,
                tolerance=1.0,
                passed=True,
                unit="A",
            ))

        # Compute overall score (average of relative errors, clamped to [0, 1])
        if metrics:
            errors = [m.relative_error for m in metrics if m.ref_value != 0]
            if errors:
                avg_err = float(np.mean(errors))
                score = max(0.0, 1.0 - avg_err)
            else:
                score = 1.0
            all_passed = all(m.passed for m in metrics)
        else:
            score = 0.0
            all_passed = False

        c_hash = config_hash(config_dict) if config_dict else ""

        return ValidationResult(
            device=device_name,
            metrics=metrics,
            overall_score=score,
            passed=all_passed,
            config_hash=c_hash,
        )

    def validate_plasma(
        self,
        device_name: str,
        sim_summary: dict[str, Any],
        config_dict: dict[str, Any] | None = None,
    ) -> ValidationResult:
        """Validate plasma-level results: neutron yield, peak density, temperature.

        Compares: neutron yield (order of magnitude), peak ne, peak Te.

        Args:
            device_name: Name of reference device.
            sim_summary: Simulation summary dict. Expected keys:
                - neutron_yield: Total neutron count
                - peak_ne: Peak electron density [m^-3]
                - peak_Te_eV: Peak electron temperature [eV]
            config_dict: Optional config for hashing.

        Returns:
            ValidationResult with plasma metric scores.
        """
        device = DEVICE_REGISTRY[device_name]
        metrics: list[MetricResult] = []

        # 1. Neutron yield (order of magnitude comparison)
        if "neutron_yield" in sim_summary and device.neutron_yield > 0:
            sim_Y = sim_summary["neutron_yield"]
            ref_Y = device.neutron_yield
            # Use log10 ratio for order-of-magnitude comparison
            if sim_Y > 0 and ref_Y > 0:
                log_ratio = abs(np.log10(sim_Y / ref_Y))
                # 1 order of magnitude error = 1.0 relative error
                err = log_ratio
            else:
                err = 10.0  # Large error if zero yield
            tol = device.tolerances.get("neutron_yield", 1.0)  # 1 order of magnitude default
            metrics.append(MetricResult(
                name="neutron_yield",
                sim_value=sim_Y,
                ref_value=ref_Y,
                relative_error=err,
                tolerance=tol,
                passed=err <= tol,
                unit="neutrons",
            ))

        # 2. Peak electron density
        if "peak_ne" in sim_summary and device.peak_ne > 0:
            sim_ne = sim_summary["peak_ne"]
            ref_ne = device.peak_ne
            err = relative_error(sim_ne, ref_ne)
            tol = device.tolerances.get("peak_ne", 0.50)  # 50% default
            metrics.append(MetricResult(
                name="peak_ne",
                sim_value=sim_ne,
                ref_value=ref_ne,
                relative_error=err,
                tolerance=tol,
                passed=err <= tol,
                unit="m^-3",
            ))

        # 3. Peak electron temperature
        if "peak_Te_eV" in sim_summary and device.peak_Te_eV > 0:
            sim_Te = sim_summary["peak_Te_eV"]
            ref_Te = device.peak_Te_eV
            err = relative_error(sim_Te, ref_Te)
            tol = device.tolerances.get("peak_Te_eV", 0.50)  # 50% default
            metrics.append(MetricResult(
                name="peak_Te_eV",
                sim_value=sim_Te,
                ref_value=ref_Te,
                relative_error=err,
                tolerance=tol,
                passed=err <= tol,
                unit="eV",
            ))

        # Score
        if metrics:
            errors = [m.relative_error for m in metrics if m.ref_value != 0]
            if errors:
                avg_err = float(np.mean(errors))
                score = max(0.0, 1.0 - avg_err)
            else:
                score = 1.0
            all_passed = all(m.passed for m in metrics)
        else:
            score = 0.0
            all_passed = False

        c_hash = config_hash(config_dict) if config_dict else ""
        return ValidationResult(
            device=device_name,
            metrics=metrics,
            overall_score=score,
            passed=all_passed,
            config_hash=c_hash,
        )

    def validate_full(
        self,
        device_name: str,
        sim_summary: dict[str, Any],
        config_dict: dict[str, Any] | None = None,
        weights: dict[str, float] | None = None,
    ) -> ValidationResult:
        """Full validation combining circuit and plasma metrics with weighted scoring.

        Default weights: circuit (40%), pinch dynamics (30%), neutron yield (30%).

        Args:
            device_name: Name of reference device.
            sim_summary: Combined simulation summary dict.
            config_dict: Optional config for hashing.
            weights: Optional custom weights dict with keys:
                'circuit', 'plasma', 'neutron'. Default: 0.4, 0.3, 0.3.

        Returns:
            Combined ValidationResult with all metrics.
        """
        if weights is None:
            weights = {"circuit": 0.4, "plasma": 0.3, "neutron": 0.3}

        circuit_result = self.validate_circuit(device_name, sim_summary, config_dict)
        plasma_result = self.validate_plasma(device_name, sim_summary, config_dict)

        # Combine all metrics
        all_metrics = circuit_result.metrics + plasma_result.metrics

        # Weighted score
        circuit_score = circuit_result.overall_score
        plasma_score = plasma_result.overall_score

        # Neutron score (extract from plasma if present)
        neutron_metrics = [m for m in plasma_result.metrics if m.name == "neutron_yield"]
        if neutron_metrics:
            neutron_score = max(0.0, 1.0 - neutron_metrics[0].relative_error)
            # Remove neutron from plasma score and recompute
            non_neutron = [m for m in plasma_result.metrics if m.name != "neutron_yield"]
            if non_neutron:
                ne_errs = [m.relative_error for m in non_neutron if m.ref_value != 0]
                plasma_score = max(0.0, 1.0 - float(np.mean(ne_errs))) if ne_errs else 1.0
            else:
                plasma_score = 1.0
        else:
            neutron_score = 0.0
            # Redistribute neutron weight to circuit and plasma
            total_non_neutron = weights["circuit"] + weights["plasma"]
            if total_non_neutron > 0:
                weights = {
                    "circuit": weights["circuit"] / total_non_neutron,
                    "plasma": weights["plasma"] / total_non_neutron,
                    "neutron": 0.0,
                }

        weighted_score = (
            weights["circuit"] * circuit_score
            + weights["plasma"] * plasma_score
            + weights["neutron"] * neutron_score
        )

        c_hash = config_hash(config_dict) if config_dict else ""
        return ValidationResult(
            device=device_name,
            metrics=all_metrics,
            overall_score=weighted_score,
            passed=all(m.passed for m in all_metrics) if all_metrics else False,
            config_hash=c_hash,
        )

    def validate_all(
        self,
        sim_summary: dict[str, Any],
        config_dict: dict[str, Any] | None = None,
    ) -> dict[str, ValidationResult]:
        """Run validation against all configured devices.

        Args:
            sim_summary: Simulation summary dict.
            config_dict: Optional config for hashing.

        Returns:
            Dict of device_name -> ValidationResult.
        """
        results = {}
        for device in self.devices:
            results[device] = self.validate_circuit(device, sim_summary, config_dict)
        return results

    def report(self, results: dict[str, ValidationResult]) -> str:
        """Generate a human-readable validation report.

        Args:
            results: Dict of device_name -> ValidationResult.

        Returns:
            Formatted report string.
        """
        lines = ["=" * 60, "DPF Validation Report", "=" * 60]

        for device_name, result in results.items():
            status = "PASS" if result.passed else "FAIL"
            lines.append(f"\n{device_name}: [{status}] Score = {result.overall_score:.1%}")
            if result.config_hash:
                lines.append(f"  Config hash: {result.config_hash}")
            lines.append("-" * 40)

            for m in result.metrics:
                status_m = "OK" if m.passed else "XX"
                unit_str = f" {m.unit}" if m.unit else ""
                if m.ref_value != 0:
                    lines.append(
                        f"  [{status_m}] {m.name}: "
                        f"sim={m.sim_value:.3e}{unit_str}, "
                        f"ref={m.ref_value:.3e}{unit_str}, "
                        f"err={m.relative_error:.1%} "
                        f"(tol={m.tolerance:.0%})"
                    )
                else:
                    lines.append(
                        f"  [{status_m}] {m.name}: "
                        f"sim={m.sim_value:.3e}{unit_str}"
                    )

        lines.append("\n" + "=" * 60)
        n_pass = sum(1 for r in results.values() if r.passed)
        n_total = len(results)
        lines.append(f"Overall: {n_pass}/{n_total} devices passed")
        lines.append("=" * 60)

        return "\n".join(lines)
