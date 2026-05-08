"""DPF scaling law diagnostics.

Implements the known empirical scaling laws for neutron yield, pinch
temperature, and pinch radius as functions of device parameters.

Key scaling laws:
    Yn ~ I_pinch^4 (Lee 2008, single device)
    Yn ~ I_pinch^3.3 (cross-device, saturation at high I)
    Yn ~ E_bank^2 (small devices)
    T_pinch ~ I^2 / N_l (Bennett equilibrium)
    r_pinch ~ a * (fm)^0.5 (mass conservation)

References:
    Lee S. & Saw S.H., J. Fusion Energy 27:292 (2008) — scaling laws.
    Huba J.D., NRL Plasma Formulary (2019).
    Soto L. et al., Phys. Plasmas 17:112702 (2010) — small DPF scaling.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

# Physical constants
MU_0 = 4.0 * math.pi * 1e-7
K_B = 1.380649e-23
EV = 1.602e-19


def _default_source_basis() -> dict[str, str]:
    return {
        "lee_I4": (
            "Lee/Saw numerical-experiment scaling using pinch current; a "
            "current waveform fit is needed when I_pinch is not measured."
        ),
        "cross_device": (
            "Cross-device empirical scaling; KnowledgeReference includes "
            "warnings that large installations can break simple neutron "
            "yield scaling."
        ),
        "energy_E2": (
            "Small-device energy scaling retained as an empirical estimate, "
            "not as MHD solver validation."
        ),
        "bennett": (
            "Bennett equilibrium temperature diagnostic from current and "
            "line density, not a validation of simulated pinch state."
        ),
    }


def _default_validity_notes() -> dict[str, str]:
    return {
        "role": (
            "Scaling-law outputs are diagnostic estimates and regime warnings; "
            "they are not solver-validation metrics."
        ),
        "current_input": (
            "The neutron-current laws are stated in terms of pinch current; "
            "using peak bank/circuit current makes the estimate rough."
        ),
        "saturation": (
            "At high current or unfavorable geometry, I^4 scaling can "
            "overpredict yield because finite pinch size and drive/electrode "
            "effects matter."
        ),
    }


@dataclass
class ScalingResult:
    """Scaling law predictions for a DPF device."""

    # Device parameters
    I_pinch_kA: float
    E_bank_kJ: float
    a_mm: float
    b_mm: float

    # Scaling predictions
    Yn_lee_I4: float       # Lee single-device: Yn ~ I^4
    Yn_cross_I33: float    # Cross-device: Yn ~ I^3.3
    Yn_energy_E2: float    # Energy scaling: Yn ~ E^2
    T_bennett_keV: float   # Bennett equilibrium temperature
    r_pinch_mm: float      # Estimated pinch radius

    # Where this device sits
    device_class: str      # "small" (<10 kJ), "medium" (10-100 kJ), "large" (>100 kJ)
    saturation_flag: bool  # True if I > saturation threshold
    model_role: str = "diagnostic_estimate"
    validation_role: str = "not_solver_validation"
    source_basis: dict[str, str] = field(default_factory=_default_source_basis)
    validity_notes: dict[str, str] = field(default_factory=_default_validity_notes)

    def to_summary_dict(self) -> dict[str, object]:
        """Return a diagnostics-safe scaling summary."""
        return {
            "Yn_I4": self.Yn_lee_I4,
            "Yn_I33": self.Yn_cross_I33,
            "Yn_E2": self.Yn_energy_E2,
            "T_bennett_keV": self.T_bennett_keV,
            "r_pinch_mm": self.r_pinch_mm,
            "device_class": self.device_class,
            "saturation": self.saturation_flag,
            "model_role": self.model_role,
            "validation_role": self.validation_role,
            "source_basis": dict(self.source_basis),
            "validity_notes": dict(self.validity_notes),
        }


def compute_scaling(
    I_pinch_kA: float,
    E_bank_kJ: float,
    a_mm: float,
    b_mm: float,
    fill_pressure_Pa: float = 400.0,
    fm: float = 0.15,
    ion_mass_kg: float = 3.34e-27,
) -> ScalingResult:
    """Compute scaling law predictions for a DPF device.

    Args:
        I_pinch_kA: Peak pinch current [kA].
        E_bank_kJ: Bank energy [kJ].
        a_mm: Anode radius [mm].
        b_mm: Cathode radius [mm].
        fill_pressure_Pa: Fill gas pressure [Pa].
        fm: Mass fraction swept.
        ion_mass_kg: Ion mass [kg].

    Returns:
        ScalingResult with all predictions.
    """
    I_A = I_pinch_kA * 1e3
    a_m = a_mm * 1e-3

    # --- Yn ~ I^4 (Lee 2008, single device) ---
    # Calibrated to PF-1000: Yn ~ 1e8 at I ~ 1.7 MA
    # C_I4 = 1e8 / (1.7e6)^4 = 1.2e-18
    C_I4 = 1.2e-18
    Yn_I4 = C_I4 * I_A**4

    # --- Yn ~ I^3.3 (cross-device, includes saturation) ---
    # From Lee & Saw 2008 multi-device fit
    # Calibrated: Yn ~ 1e8 at I ~ 1.7 MA
    C_I33 = 1e8 / (1.7e6)**3.3
    Yn_I33 = C_I33 * I_A**3.3

    # --- Yn ~ E^2 (energy scaling for small devices) ---
    # Soto 2010: Yn ~ (E/E_ref)^2 * Yn_ref
    # PF-400J: ~1e4 at 287 J, UNU-ICTP: ~1e6 at 3 kJ
    C_E2 = 1e6 / (3.0)**2  # Calibrated to UNU-ICTP
    Yn_E2 = C_E2 * E_bank_kJ**2

    # --- Bennett temperature ---
    # T_B = mu_0 * I^2 / (8*pi*N_l*2*kB)
    # N_l = linear density = n * pi * a^2
    n_fill = fill_pressure_Pa / (K_B * 300.0)
    N_l = n_fill * math.pi * a_m**2 * fm
    if N_l > 0:
        T_bennett_K = MU_0 * I_A**2 / (8.0 * math.pi * N_l * 2.0 * K_B)
        T_bennett_keV = T_bennett_K * K_B / (1000.0 * EV)
    else:
        T_bennett_keV = 0.0

    # --- Pinch radius ---
    # Mass conservation: rho_pinch * pi * r_p^2 = rho_fill * pi * a^2 * fm
    # r_p = a * sqrt(fm * rho_fill / rho_pinch)
    # Typical compression: rho_pinch ~ 10 * rho_fill
    compression = 10.0  # EMPIRICAL
    r_pinch_m = a_m * math.sqrt(fm / compression)
    r_pinch_mm = r_pinch_m * 1e3

    # --- Device classification ---
    if E_bank_kJ < 10:
        device_class = "small"
    elif E_bank_kJ < 100:
        device_class = "medium"
    else:
        device_class = "large"

    # Saturation: above ~2 MA, Yn scaling weakens
    saturation_flag = I_pinch_kA > 2000

    return ScalingResult(
        I_pinch_kA=I_pinch_kA,
        E_bank_kJ=E_bank_kJ,
        a_mm=a_mm,
        b_mm=b_mm,
        Yn_lee_I4=Yn_I4,
        Yn_cross_I33=Yn_I33,
        Yn_energy_E2=Yn_E2,
        T_bennett_keV=T_bennett_keV,
        r_pinch_mm=r_pinch_mm,
        device_class=device_class,
        saturation_flag=saturation_flag,
    )


def scaling_narrative(result: ScalingResult) -> str:
    """Generate human-readable scaling analysis."""
    lines = [
        "## Scaling Law Diagnostic Estimate",
        "",
        f"Device class: **{result.device_class}** ({result.E_bank_kJ:.0f} kJ)",
        "Model role: diagnostic estimate, not solver validation.",
        "",
        "| Scaling Law | Predicted Yn | Reference |",
        "|-------------|-------------|-----------|",
        f"| Yn ~ I^4 (single device) | {result.Yn_lee_I4:.2e} | Lee & Saw 2008 |",
        f"| Yn ~ I^3.3 (cross-device) | {result.Yn_cross_I33:.2e} | Lee & Saw 2008 |",
        f"| Yn ~ E^2 (energy) | {result.Yn_energy_E2:.2e} | Soto 2010 |",
        "",
        f"Bennett temperature: **{result.T_bennett_keV:.2f} keV**",
        f"Estimated pinch radius: **{result.r_pinch_mm:.1f} mm**",
    ]
    if result.saturation_flag:
        lines.append(
            "\n> **Saturation warning:** At I > 2 MA, beam-target yield saturates "
            "due to finite pinch size. The I^4 scaling overestimates."
        )
    return "\n".join(lines)
