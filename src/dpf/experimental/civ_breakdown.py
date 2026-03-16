"""Critical Ionization Velocity (CIV) breakdown model for DPF Phase 1.

Implements Alfven's CIV theory (1954) to compute self-consistent initial
conditions for the snowplow model: sheath thickness, electron temperature,
and ionization fraction at breakdown.

When the E x B drift velocity exceeds the critical ionization velocity
v_crit = sqrt(2 * e * V_i / m_i), the neutral gas ionizes via the CIV
mechanism rather than classical Paschen breakdown.

Physics:
    - CIV: v_crit = sqrt(2*e*V_i/m_i) where V_i is first ionization potential
    - E x B drift: v_ExB = E/B where E = V0/d_gap, B = mu_0*I_seed/(2*pi*r_mid)
    - Townsend avalanche: alpha = A*p*exp(-B*p*d/V) with gas-specific A, B
    - Paschen threshold: V_bd = B*p*d / ln(A*p*d / ln(1 + 1/gamma_se))
    - Initial sheath thickness: delta ~ lambda_mfp * (v_ExB / v_crit)
    - Electron temperature from drift energy: T_e = 0.5 * m_e * v_ExB^2 / k_B

References:
    Alfven, H., "On the Origin of the Solar System" (1954)
    Brenning, N., Space Sci. Rev. 59:209-314 (1992) — CIV review
    Danielsson, L., Phys. Fluids 13:2288 (1970) — CIV lab experiments
    Haerendel, G., Z. Naturforsch. 37a:728 (1982) — CIV in space
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from dpf.constants import e, eV, k_B, m_e, mu_0, pi

# --- Gas properties database ---

@dataclass(frozen=True)
class GasProperties:
    """Physical properties of a fill gas species."""

    name: str
    formula: str
    ion_mass: float          # Ion mass [kg]
    V_ionization: float      # First ionization potential [eV]
    v_crit: float            # CIV threshold [m/s]
    sigma_en: float          # Electron-neutral cross section [m^2]
    paschen_A: float         # Townsend first coefficient A [1/(m*Pa)]
    paschen_B: float         # Townsend second coefficient B [V/(m*Pa)]
    gamma_se: float          # Secondary electron emission coefficient
    purpose: str             # What this gas is used for in DPF


# Precomputed CIV: v_crit = sqrt(2 * e * V_i / m_i)
_GAS_DB: dict[str, GasProperties] = {}


def _register_gas(
    name: str,
    formula: str,
    ion_mass_amu: float,
    V_ionization_eV: float,
    sigma_en: float,
    paschen_A: float,
    paschen_B: float,
    gamma_se: float,
    purpose: str,
) -> None:
    ion_mass = ion_mass_amu * 1.6605e-27  # amu to kg
    v_crit = math.sqrt(2.0 * e * V_ionization_eV / ion_mass)
    _GAS_DB[name] = GasProperties(
        name=name,
        formula=formula,
        ion_mass=ion_mass,
        V_ionization=V_ionization_eV,
        v_crit=v_crit,
        sigma_en=sigma_en,
        paschen_A=paschen_A,
        paschen_B=paschen_B,
        gamma_se=gamma_se,
        purpose=purpose,
    )


# Register all common DPF fill gases
# Sources: NIST Atomic Spectra Database, Lieberman & Lichtenberg (2005)
# Sources: NIST Atomic Spectra Database, Lieberman & Lichtenberg (2005) Table 1.2
# Ionization potentials: NIST (2024), molecular values for D2/H2/N2
# Paschen A,B coefficients: Meek & Craggs (1978), Lieberman & Lichtenberg (2005)
# Cross sections: Lieberman & Lichtenberg Appendix A (~1-10 eV)
_register_gas("D2", "D₂", 4.028, 15.467, 1.0e-19, 5.0, 130.0, 0.05,
              "Neutron production (D-D fusion)")
_register_gas("H2", "H₂", 2.016, 15.426, 1.0e-19, 5.0, 130.0, 0.05,
              "Light gas studies")
_register_gas("He", "He", 4.003, 24.587, 0.5e-19, 3.0, 34.0, 0.02,
              "Inert fill, X-ray window studies")
_register_gas("Ne", "Ne", 20.18, 21.565, 0.8e-19, 4.0, 100.0, 0.04,
              "Soft X-ray production")
_register_gas("Ar", "Ar", 39.95, 15.759, 2.0e-19, 12.0, 180.0, 0.07,
              "Hard X-ray production, spectroscopy")
_register_gas("Kr", "Kr", 83.80, 14.000, 3.0e-19, 17.0, 240.0, 0.06,
              "EUV lithography source")
_register_gas("Xe", "Xe", 131.29, 12.130, 4.0e-19, 26.0, 350.0, 0.08,
              "EUV source, ion thruster studies")
_register_gas("N2", "N₂", 28.01, 15.581, 1.8e-19, 9.0, 257.0, 0.04,
              "Nitrogen plasma studies")


def get_gas(name: str) -> GasProperties:
    """Look up gas properties by name (case-insensitive)."""
    key = name.strip().upper()
    # Map full names to symbols
    _name_map = {
        "DEUTERIUM": "D2", "ARGON": "AR", "NEON": "NE",
        "HELIUM": "HE", "KRYPTON": "KR", "XENON": "XE",
        "HYDROGEN": "H2", "NITROGEN": "N2",
    }
    key = _name_map.get(key, key)
    # Capitalize element symbols: "ar" → "Ar", "ne" → "Ne", "he" → "He"
    _symbol_map = {
        "AR": "Ar", "NE": "Ne", "HE": "He", "KR": "Kr", "XE": "Xe",
    }
    key = _symbol_map.get(key, key)
    if key not in _GAS_DB:
        raise ValueError(f"Unknown gas '{name}'. Available: {list(_GAS_DB.keys())}")
    return _GAS_DB[key]


def list_gases() -> list[str]:
    """Return available gas names."""
    return list(_GAS_DB.keys())


# --- Breakdown result ---

@dataclass
class BreakdownResult:
    """Result of the CIV/Paschen breakdown calculation.

    Provides initial conditions for the snowplow model.
    """

    mechanism: str              # "CIV" or "Paschen"
    gas: GasProperties          # Gas species used
    v_crit: float               # CIV threshold velocity [m/s]
    v_ExB: float                # E x B drift velocity [m/s]
    civ_ratio: float            # v_ExB / v_crit (>1 = CIV active)
    sheath_thickness: float     # Initial sheath thickness delta [m]
    Te_initial: float           # Initial electron temperature [K]
    Te_initial_eV: float        # Initial electron temperature [eV]
    ionization_fraction: float  # Fraction of gas ionized (0-1)
    breakdown_time: float       # Estimated breakdown time [s]
    E_field: float              # Applied electric field at breakdown [V/m]
    B_seed: float               # Seed magnetic field [T]
    paschen_voltage: float      # Paschen breakdown voltage [V]
    V_applied: float            # Applied voltage [V]
    fill_pressure_Pa: float     # Fill gas pressure [Pa]
    electron_mfp: float         # Electron mean free path [m]
    larmor_radius_e: float      # Electron Larmor radius [m]
    is_magnetized: bool         # True if larmor_radius < mfp (magnetized electrons)
    summary: str                # Human-readable summary


def compute_breakdown(
    V0: float,
    fill_pressure_Pa: float,
    anode_radius: float,
    cathode_radius: float,
    insulator_length: float = 0.05,
    gas_name: str = "D2",
    B_seed: float | None = None,
    I_seed: float | None = None,
) -> BreakdownResult:
    """Compute breakdown initial conditions from CIV or Paschen theory.

    Args:
        V0: Initial capacitor voltage [V].
        fill_pressure_Pa: Fill gas pressure [Pa].
        anode_radius: Anode radius [m].
        cathode_radius: Cathode radius [m].
        insulator_length: Insulator surface length [m] (for Paschen pd).
        gas_name: Fill gas species (e.g., "D2", "Ar", "Ne").
        B_seed: Seed magnetic field [T]. If None, computed from I_seed.
        I_seed: Seed current [A] for B_seed calculation. Default: 100 A
            (displacement current at voltage rise, ~10 ns risetime).

    Returns:
        BreakdownResult with initial conditions for snowplow.
    """
    gas = get_gas(gas_name)

    # --- Geometry ---
    gap = cathode_radius - anode_radius  # radial gap [m]
    r_mid = 0.5 * (anode_radius + cathode_radius)  # midpoint radius

    # --- Electric field ---
    # Radial E-field between coaxial electrodes: E = V / (r * ln(b/a))
    # Simplified: E ~ V0 / gap for uniform field approximation
    ln_ratio = math.log(cathode_radius / anode_radius)
    E_field = V0 / (r_mid * ln_ratio)

    # --- Seed magnetic field ---
    # During voltage rise, displacement current I_d = C * dV/dt creates B_theta
    # For a 30 uF cap at 15 kV with 10 ns risetime: I_d ~ C*V/t = 30e-6 * 15e3/10e-9 = 45 kA
    # But in practice, leakage current through gas provides ~100 A seed
    if B_seed is None:
        if I_seed is None:
            I_seed = 100.0  # Conservative seed current [A]
        B_seed = mu_0 * I_seed / (2.0 * pi * r_mid)

    # --- E x B drift velocity ---
    # In crossed E and B fields, particles drift at v_ExB = E/B
    # Clamp to speed of light (v_ExB > c means B is too weak for drift physics)
    c_light = 3.0e8  # m/s
    v_ExB_raw = E_field / B_seed if B_seed > 0 else 0.0
    v_ExB = min(v_ExB_raw, c_light)

    # If v_ExB hit the c clamp, the B-field is too weak for CIV to apply.
    # At initial breakdown, B ~ 0 — Paschen dominates. CIV only applies
    # during the axial phase when current is already flowing at ~kA level.
    b_field_sufficient = v_ExB_raw < c_light

    # --- CIV ratio ---
    civ_ratio = v_ExB / gas.v_crit if gas.v_crit > 0 else 0.0

    # --- Paschen breakdown voltage ---
    # V_bd = B*p*d / ln(A*p*d / ln(1 + 1/gamma_se))
    pd = fill_pressure_Pa * insulator_length  # pressure-distance product
    denom_inner = math.log(1.0 + 1.0 / gas.gamma_se)
    apd = gas.paschen_A * pd
    if apd > denom_inner and denom_inner > 0:
        V_paschen = gas.paschen_B * pd / math.log(apd / denom_inner)
    else:
        V_paschen = 1e6  # Very high — Paschen doesn't apply at this pd

    # --- Electron mean free path ---
    # lambda_mfp = 1 / (n * sigma)
    n_gas = fill_pressure_Pa / (k_B * 300.0)  # number density at room temp
    electron_mfp = 1.0 / (n_gas * gas.sigma_en) if n_gas > 0 else 1.0

    # --- Electron Larmor radius ---
    # For thermal electrons at ~1 eV: v_th = sqrt(2*e*1eV/m_e) ~ 5.9e5 m/s
    v_th_e = math.sqrt(2.0 * eV / m_e)  # ~1 eV thermal speed
    larmor_e = m_e * v_th_e / (e * B_seed) if B_seed > 0 else float("inf")
    is_magnetized = larmor_e < electron_mfp

    # --- Determine mechanism and compute ICs ---
    # CIV requires: (1) v_ExB > v_crit AND (2) B-field strong enough for
    # magnetized drift physics (v_ExB < c). At initial breakdown with only
    # seed current, B ~ 0 → v_ExB → infinity → always Paschen.
    # CIV activates during axial phase when I > ~1 kA.
    if civ_ratio > 1.0 and b_field_sufficient:
        mechanism = "CIV"

        # CIV ionization: sheath forms where E x B exceeds v_crit
        # Sheath thickness scales with ion mean free path * velocity ratio
        # Danielsson 1970: delta ~ lambda_i * (v_ExB/v_crit)^0.5
        ion_mfp = 1.0 / (n_gas * gas.sigma_en * 0.1) if n_gas > 0 else 0.01
        # Ion-neutral cross section ~ 10x smaller than e-n at these energies
        sheath_thickness = ion_mfp * math.sqrt(civ_ratio)
        sheath_thickness = max(sheath_thickness, 1e-4)  # minimum 0.1 mm
        sheath_thickness = min(sheath_thickness, gap * 0.5)  # can't exceed half-gap

        # Electron temperature from CIV: kinetic energy of drift
        # T_e ~ 0.5 * m_i * v_crit^2 / e = V_ionization
        # This is the CIV energy threshold — electrons gain ionization energy
        Te_eV = gas.V_ionization * 0.5  # Half ionization potential
        Te_K = Te_eV * eV / k_B

        # CIV ionization fraction: rapid for v >> v_crit
        # Brenning 1992: ionization rate ~ n_n * sigma * v_rel
        # For v_ExB >> v_crit, ionization is essentially complete
        ionization_fraction = min(1.0, 0.5 * civ_ratio)

        # CIV breakdown time: time for ionization front to cross gap
        # t_bd ~ gap / v_ExB (ionization front propagates at ~v_ExB)
        # Minimum ~1 ns: ionization requires multiple electron-neutral collisions
        breakdown_time = gap / v_ExB if v_ExB > 0 else 1e-6
        breakdown_time = max(breakdown_time, 1e-9)  # Physical minimum ~1 ns

    else:
        mechanism = "Paschen"

        # Classical Paschen/Townsend breakdown
        # Sheath thickness ~ Debye length after breakdown
        # T_e ~ a few eV from Townsend avalanche
        Te_eV = 2.0  # Typical Townsend electron temperature
        Te_K = Te_eV * eV / k_B

        # Sheath is thin — a few Debye lengths
        # lambda_D = sqrt(epsilon_0 * k_B * T_e / (n_e * e^2))
        # After breakdown, n_e ~ 0.01 * n_gas initially
        n_e_initial = 0.01 * n_gas
        if n_e_initial > 0:
            from dpf.constants import epsilon_0
            lambda_D = math.sqrt(epsilon_0 * k_B * Te_K / (n_e_initial * e**2))
            sheath_thickness = 10.0 * lambda_D  # ~ 10 Debye lengths
        else:
            sheath_thickness = 1e-3  # 1 mm default

        sheath_thickness = max(sheath_thickness, 1e-4)
        sheath_thickness = min(sheath_thickness, gap * 0.5)

        # Paschen ionization is slower — Townsend avalanche
        ionization_fraction = 0.1  # Partial ionization initially

        # Paschen breakdown time: formative time lag
        # t_bd ~ (pd)^0.5 / V * scaling (empirical, ~100 ns to few us)
        if V_paschen < V0:
            overvoltage = V0 / V_paschen
            breakdown_time = 1e-7 / overvoltage  # ~ 100 ns / overvoltage
        else:
            breakdown_time = 1e-5  # 10 us — slow breakdown

    # --- Build summary ---
    summary_lines = [
        f"Breakdown mechanism: {mechanism}",
        f"Gas: {gas.formula} ({gas.purpose})",
        f"CIV threshold: v_crit = {gas.v_crit/1e3:.1f} km/s",
        f"E x B drift: v_ExB = {v_ExB/1e3:.1f} km/s",
        f"CIV ratio: v_ExB/v_crit = {civ_ratio:.2f}",
    ]

    if mechanism == "CIV":
        summary_lines.extend([
            "CIV ionization ACTIVE (ratio > 1)",
            f"Sheath thickness: {sheath_thickness*1e3:.2f} mm",
            f"Initial T_e: {Te_eV:.1f} eV ({Te_K:.0f} K)",
            f"Ionization fraction: {ionization_fraction*100:.0f}%",
            f"Breakdown time: {breakdown_time*1e9:.0f} ns",
        ])
    else:
        summary_lines.extend([
            "Classical Paschen breakdown (CIV ratio < 1)",
            f"Paschen voltage: {V_paschen:.0f} V (applied: {V0:.0f} V)",
            f"Sheath thickness: {sheath_thickness*1e3:.2f} mm",
            f"Initial T_e: {Te_eV:.1f} eV",
            f"Ionization fraction: {ionization_fraction*100:.0f}%",
            f"Breakdown time: {breakdown_time*1e9:.0f} ns",
        ])

    summary_lines.extend([
        f"Electron mfp: {electron_mfp*1e3:.2f} mm",
        f"Electron Larmor radius: {larmor_e*1e3:.2f} mm",
        f"Electrons {'magnetized' if is_magnetized else 'unmagnetized'}",
    ])

    return BreakdownResult(
        mechanism=mechanism,
        gas=gas,
        v_crit=gas.v_crit,
        v_ExB=v_ExB,
        civ_ratio=civ_ratio,
        sheath_thickness=sheath_thickness,
        Te_initial=Te_K,
        Te_initial_eV=Te_eV,
        ionization_fraction=ionization_fraction,
        breakdown_time=breakdown_time,
        E_field=E_field,
        B_seed=B_seed,
        paschen_voltage=V_paschen,
        V_applied=V0,
        fill_pressure_Pa=fill_pressure_Pa,
        electron_mfp=electron_mfp,
        larmor_radius_e=larmor_e,
        is_magnetized=is_magnetized,
        summary="\n".join(summary_lines),
    )


def compute_liftoff_delay(breakdown: BreakdownResult) -> float:
    """Estimate the liftoff delay from breakdown result.

    The liftoff delay is the time between voltage application and
    current sheet detachment from the insulator. It includes:
    1. Statistical time lag (stochastic, ~10-100 ns)
    2. Formative time lag (ionization buildup, from breakdown calculation)
    3. Sheet formation time (current concentration, ~50-200 ns)

    Args:
        breakdown: Result from compute_breakdown().

    Returns:
        Estimated liftoff delay [s].
    """
    # Statistical lag: random, typically 10-100 ns
    statistical_lag = 50e-9  # 50 ns average

    # Formative lag: from breakdown calculation
    formative_lag = breakdown.breakdown_time

    # Sheet formation: current concentrates into a thin layer
    # Faster for CIV (already localized), slower for Paschen (diffuse)
    if breakdown.mechanism == "CIV":
        formation_lag = 50e-9  # CIV: fast, localized ionization
    else:
        formation_lag = 200e-9  # Paschen: slower, diffuse avalanche

    return statistical_lag + formative_lag + formation_lag


def compute_initial_sheath_state(
    breakdown: BreakdownResult,
    anode_radius: float,
    cathode_radius: float,
    fill_pressure_Pa: float,
) -> dict[str, float]:
    """Convert breakdown result into initial conditions for snowplow/MHD.

    Returns a dictionary that can be used to initialize the snowplow model
    or seed the MHD grid with a physically motivated initial state.

    Args:
        breakdown: Result from compute_breakdown().
        anode_radius: Anode radius [m].
        cathode_radius: Cathode radius [m].
        fill_pressure_Pa: Fill gas pressure [Pa].

    Returns:
        Dict with keys:
            - sheath_position_z: Initial z position of sheath [m]
            - sheath_thickness: Radial sheath thickness [m]
            - Te: Electron temperature [K]
            - Ti: Ion temperature [K] (cold ions initially)
            - ionization_fraction: 0-1
            - rho_sheath: Sheath mass density [kg/m^3]
            - v_sheath_z: Initial axial sheath velocity [m/s]
            - liftoff_delay: Estimated delay before rundown begins [s]
    """
    gap = cathode_radius - anode_radius
    n_gas = fill_pressure_Pa / (k_B * 300.0)

    # Sheath mass density: swept gas in thin layer
    # rho_sheath = n_gas * m_ion * (gap / delta) for compression into delta
    rho_sheath = n_gas * breakdown.gas.ion_mass
    if breakdown.sheath_thickness > 0:
        compression = min(gap / breakdown.sheath_thickness, 100.0)
        rho_sheath *= compression

    # Initial axial velocity: J x B acceleration during breakdown
    # Very small — sheath hasn't started moving yet
    v_sheath_z = 0.0  # Starts from rest

    # Ion temperature: cold initially (CIV heats electrons, not ions)
    Ti = 300.0  # Room temperature

    liftoff = compute_liftoff_delay(breakdown)

    return {
        "sheath_position_z": 0.0,  # At insulator surface
        "sheath_thickness": breakdown.sheath_thickness,
        "Te": breakdown.Te_initial,
        "Ti": Ti,
        "ionization_fraction": breakdown.ionization_fraction,
        "rho_sheath": rho_sheath,
        "v_sheath_z": v_sheath_z,
        "liftoff_delay": liftoff,
        "breakdown_time": breakdown.breakdown_time,
        "mechanism": breakdown.mechanism,
    }


def breakdown_narrative(breakdown: BreakdownResult) -> str:
    """Generate a human-readable narrative of the breakdown physics.

    Suitable for display in the web UI Physics Narrative tab.
    """
    gas = breakdown.gas
    lines = []

    lines.append("## Gas Breakdown & Plasma Formation (Phase 1)")
    lines.append("")

    if breakdown.mechanism == "CIV":
        lines.append(
            f"At {breakdown.V_applied/1e3:.0f} kV, the electric field between the "
            f"electrodes ({breakdown.E_field/1e3:.0f} kV/m) creates an E x B drift "
            f"velocity of {breakdown.v_ExB/1e3:.0f} km/s."
        )
        lines.append("")
        lines.append(
            f"This exceeds the **Critical Ionization Velocity** (CIV) of "
            f"{gas.formula}: v_crit = {gas.v_crit/1e3:.0f} km/s "
            f"(ratio = {breakdown.civ_ratio:.1f}x)."
        )
        lines.append("")
        lines.append(
            "Alfven's CIV mechanism (1954): when neutral gas moves through a "
            "magnetized plasma faster than v_crit = sqrt(2eV_i/m_i), the kinetic "
            "energy of relative motion exceeds the ionization potential. Electrons "
            "are heated to ~V_i/2, causing rapid ionization. This creates a "
            "self-sustaining ionization front."
        )
        lines.append("")
        lines.append(
            f"The {gas.formula} gas ionizes in ~{breakdown.breakdown_time*1e9:.0f} ns, "
            f"forming a current sheath {breakdown.sheath_thickness*1e3:.1f} mm thick "
            f"at T_e = {breakdown.Te_initial_eV:.1f} eV."
        )
    else:
        lines.append(
            f"At {breakdown.V_applied/1e3:.0f} kV and "
            f"{breakdown.fill_pressure_Pa:.0f} Pa ({breakdown.fill_pressure_Pa/133.322:.1f} Torr), "
            f"the {gas.formula} gas breaks down via the classical **Paschen mechanism**."
        )
        lines.append("")
        lines.append(
            f"The Paschen breakdown voltage is {breakdown.paschen_voltage:.0f} V. "
            f"With {breakdown.V_applied:.0f} V applied "
            f"({breakdown.V_applied/breakdown.paschen_voltage:.1f}x overvoltage), "
            "a Townsend avalanche ionizes the gas along the insulator surface."
        )
        lines.append("")
        lines.append(
            f"Breakdown takes ~{breakdown.breakdown_time*1e9:.0f} ns. "
            f"The initial sheath is {breakdown.sheath_thickness*1e3:.1f} mm thick "
            f"at T_e = {breakdown.Te_initial_eV:.1f} eV with "
            f"{breakdown.ionization_fraction*100:.0f}% ionization."
        )

    lines.append("")
    lines.append(
        f"Electrons are {'magnetized' if breakdown.is_magnetized else 'unmagnetized'} "
        f"(Larmor radius {breakdown.larmor_radius_e*1e3:.1f} mm "
        f"{'<' if breakdown.is_magnetized else '>'} "
        f"mean free path {breakdown.electron_mfp*1e3:.1f} mm)."
    )

    return "\n".join(lines)
