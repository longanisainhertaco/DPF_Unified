"""Physics narrative generator — step-by-step explanation with math."""
from __future__ import annotations

from typing import Any

import numpy as np


def generate_narrative(d: dict[str, Any]) -> str:
    cc = d["circuit"]
    sc = d["snowplow_cfg"]
    gas = d["gas"]

    V0_kV = cc["V0"] / 1e3
    C_uF = cc["C"] * 1e6
    C_mF = cc["C"] * 1e3
    L0_nH = cc["L0"] * 1e9
    R0_mOhm = cc.get("R0", 0) * 1e3
    a_mm = cc["anode_radius"] * 1e3
    b_mm = cc["cathode_radius"] * 1e3
    ln_ba = np.log(cc["cathode_radius"] / cc["anode_radius"])
    E_kJ = d["E_bank_kJ"]
    T_LC = d["T_LC_us"]

    I_sc = cc["V0"] * np.sqrt(cc["C"] / cc["L0"]) / 1e6
    loading = d["I_peak"] / I_sc * 100 if I_sc > 0 else 0

    sections = []

    gap_mm = b_mm - a_mm
    sections.append(f"""## What is a Dense Plasma Focus?

A **Dense Plasma Focus (DPF)** is a pulsed plasma device that compresses gas to \
extreme temperatures (tens of millions of degrees) and densities using powerful \
magnetic fields. It works like a magnetic cannon:

1. A **capacitor bank** stores electrical energy (like a giant camera flash)
2. The energy discharges through a gas between two coaxial metal cylinders (electrodes)
3. The current creates a magnetic field that pushes the gas like a piston (**axial rundown**)
4. At the end of the inner electrode, the magnetic piston squeezes the gas inward (**radial implosion**)
5. The gas compresses into a tiny, ultra-hot column called the **pinch** — hot enough for nuclear fusion

This simulation models device **{d.get('device', 'unknown')}** — a Mather-type DPF \
with a {a_mm:.0f} mm anode inside a {b_mm:.0f} mm cathode ({gap_mm:.0f} mm gap), \
powered by a {E_kJ:.0f} kJ capacitor bank.

---

## Notation Guide

> **Key symbols used in this analysis** (reference this when you see unfamiliar notation):
>
> | Symbol | Name | Value | Meaning |
> |--------|------|-------|---------|
> | $\\mu_0$ | Permeability of free space | $4\\pi \\times 10^{{-7}}$ H/m | How strongly space supports magnetic fields |
> | $k_B$ | Boltzmann constant | $1.38 \\times 10^{{-23}}$ J/K | Converts temperature to energy per particle |
> | $\\gamma$ | Adiabatic index | 5/3 for monatomic gas | How compressible the gas is |
> | $\\langle\\sigma v\\rangle$ | Reactivity | varies with T | Probability of fusion per ion pair per second |
> | $f_c$ | Current fraction | 0.5 - 0.9 (fitted) | Fraction of circuit current in the sheath |
> | $f_m$ | Mass fraction | 0.05 - 0.2 (fitted) | Fraction of fill gas swept by the sheath |
> | $N_l$ | Line density | particles/m | Number of ions per meter of pinch length |
> | MA | Mega-amperes | $10^6$ A | Unit for peak current |
> | keV | kilo-electron-volts | 11.6 million degrees C | Unit for plasma temperature |

---

## How Pulsed Power Works

> **For newcomers:** "Pulsed power" means storing energy slowly (seconds) and releasing it extremely fast (microseconds). It's the same principle as a camera flash — the battery charges the capacitor over seconds, then the flash dumps all that energy in a millisecond, producing a brief but intense burst of light. A DPF does the same thing, but with electrical current instead of light, and the "flash" lasts millionths of a second.

**The key components:**
- **Capacitor bank** — stores electrical energy as charge on metal plates separated by insulator. Measured in Farads (F) or microfarads (uF). Larger capacitance = more energy stored.
- **Spark gap / switch** — holds the energy until triggered, then connects the capacitor to the electrodes in nanoseconds
- **Transmission lines / cables** — carry the current from the capacitor bank to the DPF electrodes. Their inductance (L0) and resistance (R0) limit how fast the energy can be delivered.
- **Crowbar switch** (optional) — fires after the first current peak to prevent the current from reversing and damaging the capacitors

**Why pulsed?** Fusion requires extreme conditions — tens of millions of degrees, densities 100x normal gas. Steady-state power can't create these conditions in a small device. But by compressing {E_kJ:.0f} kJ into ~5 microseconds, the instantaneous power reaches billions of watts — enough to create fusion conditions in a plasma column the size of a pencil lead.

---

## Step 1: Energy Storage & Circuit

> **Purpose:** Before anything happens, we need to know how much energy is available \
and how fast it can be delivered. The capacitor bank is the "fuel tank" — its voltage \
and capacitance determine the total energy, while the circuit inductance determines \
how quickly that energy converts to current.

The capacitor bank stores electrostatic energy — this formula tells us the **total \
energy budget** for the entire experiment:

$$E_0 = \\frac{{1}}{{2}} C V_0^2 \
= \\frac{{1}}{{2}} \\times {C_mF:.3f}\\text{{ mF}} \\times ({V0_kV:.0f}\\text{{ kV}})^2 \
= {E_kJ:.0f}\\text{{ kJ}}$$

> *For context: {E_kJ:.0f} kJ is roughly the kinetic energy of a \
{max(1, E_kJ / 0.5):.0f} kg object moving at 1 m/s — released in microseconds.*

The **LC oscillation period** tells us the natural timescale of the circuit — how \
fast the current rises and falls without any plasma load:

$$T_{{LC}} = 2\\pi\\sqrt{{L_0 C}} \
= 2\\pi\\sqrt{{{L0_nH:.1f}\\text{{ nH}} \\times {C_uF:.0f}\\text{{ uF}}}} \
= {T_LC:.1f}\\text{{ us}}$$

> *The current peaks at T_LC/4 = {T_LC/4:.1f} us. Everything interesting happens \
in the first quarter-cycle.*

The **short-circuit peak current** is the theoretical maximum — what you'd get if \
the electrodes were shorted with no plasma:

$$I_{{sc}} = V_0 \\sqrt{{\\frac{{C}}{{L_0}}}} \
= {V0_kV:.0f}\\text{{ kV}} \\times \\sqrt{{\\frac{{{C_uF:.0f}\\text{{ uF}}}}{{{L0_nH:.1f}\\text{{ nH}}}}}} \
= {I_sc:.2f}\\text{{ MA}}$$

The actual peak is **{d['I_peak']:.3f} MA** — only {loading:.0f}% of $I_{{sc}}$. \
The missing energy went into accelerating and compressing the plasma. This "loading" \
is actually a good sign — it means the plasma is absorbing energy efficiently.""")

    if R0_mOhm > 0:
        RESF = cc.get("R0", 0) / np.sqrt(cc["L0"] / cc["C"])
        sections[-1] += f"""

External resistance $R_0 = {R0_mOhm:.1f}$ mOhm \
(RESF = $R_0/\\sqrt{{L_0/C}}$ = {RESF:.3f})."""

    p_torr = sc.get("fill_pressure_Pa", 400) / 133.322
    gas_purpose = {
        "D2": "Deuterium is used because D-D fusion produces detectable neutrons — the primary diagnostic of a working DPF.",
        "pB11": "Proton-boron-11 is an aneutronic fuel — fusion produces alpha particles instead of neutrons, with potential for direct energy conversion.",
        "He": "Helium is used for plasma studies without fusion reactions (noble gas, fully ionized at lower temperatures).",
        "Ne": "Neon produces intense soft X-rays when highly ionized — used for lithography and radiography applications.",
        "Ar": "Argon is commonly used for X-ray source applications and plasma diagnostics.",
        "Kr": "Krypton produces hard X-rays useful for imaging applications.",
        "Xe": "Xenon produces the hardest X-rays of the noble gases — used for radiographic applications.",
        "N2": "Nitrogen is used for EUV radiation studies.",
        "H2": "Hydrogen is used for basic plasma studies (lightest gas, fastest sheath speeds).",
    }.get(d.get("gas_key", "D2"), "")

    sections.append(f"""## Step 2: Fill Gas & Electrode Geometry

> **Purpose:** The choice of gas determines what the DPF produces (neutrons, X-rays, \
or EUV radiation). The electrode geometry determines how efficiently the magnetic \
piston compresses the gas. These are the "design choices" of the device.

**Gas**: {gas['name']} (atomic mass $A = {gas['A']}$, charge $Z = {gas['Z']}$, \
ion mass $m = {gas['m_mol']:.2e}$ kg)

> *{gas_purpose}*

**Fill pressure**: {p_torr:.1f} Torr = {sc.get('fill_pressure_Pa', 400):.0f} Pa

> *Lower pressure = less mass = higher temperature but fewer reactions. \
Higher pressure = more mass = more reactions but harder to compress. \
There's an optimum for each device.*

The gas density is calculated from the **ideal gas law** at room temperature \
(the gas starts cold before the discharge):

$$\\rho_0 = \\frac{{P \\cdot m_{{ion}}}}{{k_B \\cdot T}} = {d['rho0']:.3e}\\text{{ kg/m}}^3$$

**Coaxial electrodes** (Mather configuration): The device has two concentric \
cylinders — the **anode** (inner, radius $a = {a_mm:.1f}$ mm) and the **cathode** \
(outer, radius $b = {b_mm:.1f}$ mm). The gas fills the {gap_mm:.0f} mm gap between them.

The ratio $\\ln(b/a)$ appears in every inductance and force formula — it captures \
the geometry of the coaxial magnetic field:

$$\\ln\\left(\\frac{{b}}{{a}}\\right) = \\ln\\left(\\frac{{{b_mm:.1f}}}{{{a_mm:.1f}}}\\right) \
= {ln_ba:.3f}$$

> *Larger $\\ln(b/a)$ = stronger magnetic field per unit current = more force on the sheath.*""")

    if d.get("has_mhd") and not d.get("has_snowplow"):
        sections.append(_mhd_narrative(d))
        sections.append(_circuit_coupling_section(d, cc))
        sections.append(_summary_section(d, gas, E_kJ, is_mhd=True))
        return "\n\n---\n\n".join(sections)

    if not d.get("has_snowplow"):
        sections.append("## No Snowplow Model\n\nPure RLC circuit without plasma load.")
        return "\n\n---\n\n".join(sections)

    fc = sc.get("current_fraction", 0.7)
    fm = sc.get("mass_fraction", 0.15)
    L_anode_mm = sc.get("anode_length", 0.16) * 1e3

    t_arr = d["t_us"]
    I_arr = d["I_MA"]

    rundown_end_idx = None
    for i, p in enumerate(d["phases"]):
        if p != "rundown":
            rundown_end_idx = i
            break

    if rundown_end_idx is not None:
        t_rundown = t_arr[rundown_end_idx]
        I_at_rundown = I_arr[rundown_end_idx]
    else:
        t_rundown = t_arr[-1]
        I_at_rundown = I_arr[-1]

    sections.append("""## Step 2.5: Gas Breakdown & Plasma Formation (Phase 1)

> **Why this matters:** Before anything can move, the neutral gas between the electrodes must become a plasma — a hot, electrically conducting gas where electrons are stripped from atoms. This happens in the first few nanoseconds of the discharge and is called "breakdown."

When the switch fires and voltage appears across the electrodes, the electric field ionizes the fill gas near the insulator surface at the breech (z = 0). This happens via **Townsend avalanche** — free electrons (from cosmic rays or UV) accelerate in the electric field, hit gas atoms, knock out more electrons, and the process cascades exponentially.

Within ~10-100 nanoseconds, a thin layer of fully ionized plasma forms across the insulator surface. This conducting layer carries the circuit current and creates the magnetic field that will drive the snowplow.

> *The simulator skips this phase (it happens too fast to affect the circuit waveform) and starts directly with the current sheath already formed. In experiments, breakdown quality affects shot-to-shot reproducibility — uneven breakdown leads to tilted or asymmetric sheaths.*""")

    sections.append(f"""## Step 3: Axial Rundown (Phase 2)

> **Why this matters:** The axial rundown is where the plasma sheath accelerates \
to ~10$^5$ m/s before hitting the anode tip. The speed at arrival determines how \
violently the radial implosion compresses the plasma — and ultimately how many \
neutrons are produced. This is the "gun barrel" phase of the DPF.

The current sheath forms at the insulator and is accelerated axially by the \
$\\mathbf{{J}} \\times \\mathbf{{B}}$ magnetic piston force \
*(Lee & Saw, J. Fusion Energy, 2008; Lee, Radiations in Plasmas, 1984)*:

$$F_{{mag}} = \\frac{{\\mu_0}}{{4\\pi}} \\ln\\left(\\frac{{b}}{{a}}\\right) (f_c I)^2$$

where $f_c = {fc:.2f}$ is the **current fraction** — the fraction of total circuit \
current that actually flows through the plasma sheath (the rest leaks through the \
hot plasma behind it). Think of it as the "efficiency" of the magnetic piston.

The sheath sweeps fill gas, accumulating mass like a snowplow:

$$m(z) = f_m \\cdot \\rho_0 \\cdot \\pi(b^2 - a^2) \\cdot z$$

where $f_m = {fm:.3f}$ is the **mass fraction** — how much of the gas gets \
picked up vs. left behind. Lower $f_m$ means less mass to compress later, \
which leads to higher temperatures but weaker neutron yield.

The equation of motion (Newton's second law for the growing mass):

$$\\frac{{d}}{{dt}}[m(z) \\cdot v] = F_{{mag}} - F_{{pressure}}$$

$$m \\frac{{dv}}{{dt}} = F_{{mag}} - p_0 \\pi(b^2 - a^2) - v \\frac{{dm}}{{dt}}$$

The third term $v \\cdot dm/dt$ is the **rocket equation drag** --- as the sheath \
picks up stationary gas, momentum is consumed accelerating it.

As the sheath moves, the plasma inductance grows linearly \
*(Potter, Phys. Fluids, 1971)*:

$$L_p(z) = \\frac{{\\mu_0}}{{2\\pi}} \\ln\\left(\\frac{{b}}{{a}}\\right) \\cdot z$$

This increasing inductance loads the circuit, causing $dL/dt > 0$ back-EMF \
that opposes current growth. This is why the actual peak current is always \
lower than the short-circuit value.

**Result**: Sheath reaches anode end ($z_0 = {L_anode_mm:.0f}$ mm) at \
$t \\approx {t_rundown:.1f}$ us with $I \\approx {abs(I_at_rundown):.3f}$ MA.""")

    fmr = sc.get("radial_mass_fraction", fm)
    pcf = sc.get("pinch_column_fraction", 1.0)
    z_f_mm = L_anode_mm * pcf

    sections.append(f"""## Step 4: Radial Implosion (Phase 3)

> **Why this matters:** This is where the magic happens. The cylindrical shock \
wave compresses plasma to millions of degrees and densities 100x the fill gas. \
The compression is so intense that deuterium nuclei fuse, producing neutrons. \
The quality of this implosion determines everything — neutron yield, X-ray \
emission, and whether the device "works."

At the anode end, the sheath curves inward. A cylindrical shock wave \
implodes radially from $r = b$ toward the axis *(Lee, J. Fusion Energy, 2009)*. \
Only a fraction of the anode length participates: \
$z_f = {pcf:.2f} \\times {L_anode_mm:.0f} = {z_f_mm:.0f}$ mm.

The radial $\\mathbf{{J}} \\times \\mathbf{{B}}$ force on a cylindrical current sheet:

$$F_{{rad}} = \\frac{{\\mu_0}}{{4\\pi}} \\frac{{(f_c I)^2 z_f}}{{r_s}}$$

Note the $1/r_s$ dependence --- the force **intensifies** as the shock converges \
toward the axis, producing rapid acceleration. This is why z-pinches are such \
effective plasma compressors.

The slug model equation of motion *(Potter, Phys. Fluids, 1971)*:

$$M_{{slug}} \\frac{{dv_r}}{{dt}} = -F_{{rad}} + F_{{back}} - v_r \\frac{{dM}}{{dt}}$$

where $M_{{slug}} = f_{{mr}} \\rho_0 \\pi(b^2 - r_s^2) z_f$ with $f_{{mr}} = {fmr:.2f}$.

Opposing the implosion is adiabatic back-pressure from compressed gas:

$$p_{{back}}(r_s) = p_0 \\left(\\frac{{b}}{{r_s}}\\right)^{{2\\gamma}}$$

For $\\gamma = 5/3$ (monatomic), this rises steeply as $r_s \\to 0$. \
This back-pressure is what ultimately stops the implosion — when it balances \
the magnetic pressure, you get the **Bennett equilibrium** \
*(Bennett, Phys. Rev., 1934)*.

The plasma inductance now grows logarithmically:

$$L_p = L_{{axial}} + \\frac{{\\mu_0}}{{2\\pi}} z_f \\ln\\left(\\frac{{b}}{{r_s}}\\right)$$

$$\\frac{{dL}}{{dt}} = -\\frac{{\\mu_0}}{{2\\pi}} \\frac{{z_f \\cdot v_r}}{{r_s}} > 0 \
\\quad (v_r < 0)$$

This rapid inductance increase produces the **characteristic current dip** --- \
the circuit cannot supply current fast enough to overcome the back-EMF. \
Experimentalists use this dip as the primary diagnostic of a successful pinch.""")

    if d["dip_pct"] > 0:
        sections[-1] += f"""

**Result**: Current dips from {d['I_pre_dip']:.3f} to \
{d['I_dip']:.3f} MA (**{d['dip_pct']:.0f}% dip**) at $t = {d['t_dip']:.1f}$ us."""

    sp = d["snowplow_obj"]
    r_pinch_mm = sp.shock_radius * 1e3 if sp else 0

    comp_ratio = b_mm / r_pinch_mm if r_pinch_mm > 0 else 0
    sections.append(f"""## Step 5: Pinch & Stagnation (Phase 4)

> **Why this matters:** This is the moment of truth — where all the kinetic energy \
of the imploding sheath converts to heat. The plasma reaches millions of degrees \
for a few nanoseconds. If the conditions are right, deuterium nuclei fuse, \
producing 2.45 MeV neutrons. This is where fusion happens.

The shock converges to a minimum radius $r_{{min}} \\approx {r_pinch_mm:.2f}$ mm \
— the plasma has been compressed {comp_ratio:.0f}:1 from its starting radius.

At stagnation, three things happen simultaneously:
1. **Kinetic energy → thermal energy**: the imploding gas slams into itself and heats up
2. **Peak density and temperature**: the plasma column is at its hottest and densest
3. **Fusion/radiation emission**: neutrons (from D-D), X-rays (from high-Z), or both

The **Bennett pinch equilibrium** *(Bennett, Phys. Rev., 1934)* describes the \
balance between the inward magnetic squeeze and the outward plasma pressure:

$$\\frac{{\\mu_0 I^2}}{{8\\pi}} = N_l k_B (T_e + T_i)$$

> *In plain English: the magnetic field from the current (left side) is squeezing \
the plasma inward. The hot plasma pressure (right side) is pushing outward. \
When they balance, you have a stable pinch. $N_l$ is the number of particles \
per meter of pinch length.*

**Compression ratio**: $b / r_{{min}} = {comp_ratio:.0f}:1$ — the plasma is \
{comp_ratio**2:.0f}x denser than the fill gas (density scales as $r^{{-2}}$ for \
cylindrical compression).""")

    if d["scaling"]:
        s = d["scaling"]
        sections[-1] += f"""

### Implosion Scaling (Goyon et al. 2025)

From 1D strong-shock theory:

$$v_{{imp}} = \\frac{{950 \\times 10^3 \\cdot I_{{MA}}}}{{R_{{cm}} \\sqrt{{P_{{Torr}}}}}} \
= {s['v_imp']/1e3:.0f}\\text{{ km/s}}$$

$$T_{{stag}} = \\frac{{21 \\cdot I_{{MA}}^2}}{{R_{{cm}}^2 \\cdot P_{{Torr}}}} \
= {s['T_stag_keV']:.1f}\\text{{ keV}}$$

$$\\tau_{{exp}} = \\frac{{31.5 \\cdot R_{{cm}}^2 \\sqrt{{P_{{Torr}}}}}}{{CR \\cdot I_{{MA}}}} \
= {s['tau_exp_ns']:.0f}\\text{{ ns}}$$

$$\\tau_{{m=0}} = \\frac{{31 \\cdot R_{{cm}}^2 \\sqrt{{P_{{Torr}}}}}}{{CR \\cdot I_{{MA}}}} \
= {s['tau_m0_ns']:.0f}\\text{{ ns}}$$

If $\\tau_{{exp}} > \\tau_{{m=0}}$, the pinch disrupts via m=0 sausage \
instability before it can expand --- this limits confinement time."""

    ny = d.get("neutron_yield")
    if ny and ny.get("Y_neutron", 0) > 0:
        T_eff = ny.get("T_eff_keV", ny.get("T_bennett_keV", 0.5))
        V_kV = ny.get("V_pinch_kV", 0)
        rad_cool = ny.get("rad_cooling_factor", 1.0)
        bt_pct = ny.get("bt_fraction", 0) * 100

        yield_section = f"""## D-D Neutron Yield

> **Why this matters:** Neutron yield is the single most important performance \
metric for a DPF. It tells you how many fusion reactions occurred — and therefore \
how hot and dense the pinch actually got. There are two ways neutrons are produced:

**Two neutron production mechanisms** *(Lee & Saw, J. Fusion Energy, 2008; \
Haines et al., Phys. Rev. Lett., 2011)*:

$$Y_n = Y_{{thermo}} + Y_{{BT}}$$

### 1. Thermonuclear component (hot plasma collisions)

> *Imagine billions of deuterium ions bouncing around at millions of degrees. \
Occasionally, two collide head-on with enough energy to fuse. The rate depends \
on temperature (hotter = more fusions), density (denser = more collisions), \
volume, and confinement time.*

$$Y_{{thermo}} = \\frac{{1}}{{2}} n_i^2 \\langle\\sigma v\\rangle V_{{pinch}} \\tau_{{pinch}}$$

Each term:
- $n_i$ = ion density [m$^{{-3}}$] — how many ions per cubic meter in the pinch
- $\\langle\\sigma v\\rangle$ = **reactivity** [m$^3$/s] — probability of fusion per ion pair \
*(Bosch & Hale, Nuclear Fusion, 1992)*
- $V_{{pinch}}$ = pinch volume [m$^3$] — how big the hot region is
- $\\tau_{{pinch}}$ = confinement time [s] — how long it stays hot (typically 10-100 ns)
- The $1/2$ avoids double-counting (D+D is same as D+D reversed)

**Bennett temperature** — the equilibrium temperature from the pinch balance:

$$T_B = \\frac{{\\mu_0 I^2}}{{8\\pi N_l \\cdot 2 k_B}} = {ny.get('T_bennett_keV', T_eff):.2f}\\text{{ keV}}$$

> *1 keV ≈ 11.6 million degrees Celsius. For comparison, the Sun's core is 1.3 keV.*"""

        if rad_cool < 0.95:
            yield_section += f"""

**Radiation-corrected temperature**: Bremsstrahlung cooling reduces the \
effective temperature:

$$T_{{eff}} = \\frac{{T_B}}{{1 + \\tau_{{pinch}} / \\tau_{{rad}}}} \
= {T_eff:.2f}\\text{{ keV}} \\quad (\\times{rad_cool:.2f})$$"""

        n_pinch = ny.get('n_pinch', 0)
        sigma_v = ny.get('sigma_v', 0)
        V_cm3 = ny.get('V_pinch_cm3', 0)
        tau_ns = ny.get('tau_ns', 0)
        Y_thermo = ny.get('Y_thermonuclear', 0)

        if n_pinch > 0:
            yield_section += f"""

Pinch density: $n_{{pinch}} = {n_pinch:.2e}$ m$^{{-3}}$

D-D reactivity (Bosch-Hale at {T_eff:.2f} keV): $\\langle\\sigma v\\rangle = {sigma_v:.2e}$ m$^3$/s

Pinch volume: ${V_cm3:.2f}$ cm$^3$, confinement: ${tau_ns:.0f}$ ns

$Y_{{thermo}} = {Y_thermo:.2e}$"""
        else:
            yield_section += f"""

$Y_{{thermo}} = {Y_thermo:.2e}$, $\\tau = {tau_ns:.0f}$ ns"""

        yield_section += """

### 2. Beam-target component (fast ions hitting stationary plasma)

> *When the pinch breaks up (via m=0 "sausage" instability), the rapidly \
changing magnetic field creates a huge voltage spike. This voltage accelerates \
some ions to very high energies — like a particle accelerator. These fast ions \
then slam into the stationary background plasma and cause fusion reactions. \
In most DPF devices, this beam-target mechanism produces MORE neutrons than \
the thermonuclear mechanism.*"""

        if V_kV > 1:
            yield_section += f"""

The pinch voltage comes from the rapid inductance change during disruption:

$$V_{{pinch}} = I \\cdot \\frac{{dL}}{{dt}} = {V_kV:.0f}\\text{{ kV}}$$

This accelerates a beam of ions to energy $E_{{beam}} = e \\cdot V_{{pinch}} = {V_kV:.0f}$ keV:

$$\\frac{{dY_{{BT}}}}{{dt}} = f_{{beam}} \\frac{{I_{{pinch}}}}{{e}} \
\\cdot n_{{target}} \\cdot \\sigma_{{DD}}(E_{{cm}}) \\cdot L_{{target}}$$

> *$f_{{beam}}$ ≈ 0.14 is the fraction of current carried by beam ions. \
$\\sigma_{{DD}}$ is the D-D fusion cross-section at the beam energy — \
much larger than at thermal energies because the ions are faster.*"""
        else:
            yield_section += """

Beam energy estimated from $E_{beam} \\sim 3 T_B$ (no significant pinch voltage in this simulation)."""

        yield_section += f"""

$Y_{{BT}} = {ny['Y_beam_target']:.2e}$ ({bt_pct:.0f}% of total)

### Total yield

**$Y_n = {ny['Y_neutron']:.2e}$ neutrons per shot** \
(thermonuclear: {100-bt_pct:.0f}%, beam-target: {bt_pct:.0f}%)"""

        sections.append(yield_section)

    sections.append(_circuit_coupling_section(d, cc))
    sections.append(_summary_section(d, gas, E_kJ, is_mhd=False, r_pinch_mm=r_pinch_mm))

    sections.append("""## References

- **Lee S.** "Radiations in Plasmas" (1984) — Original Lee model formulation
- **Lee S. & Saw S.H.** "Neutron scaling laws from a numerical experiment", J. Fusion Energy 27, 292-295 (2008) — Scaling laws and neutron yield model
- **Lee S.** "Plasma Focus Radiative Model", J. Fusion Energy 28, 9-14 (2009) — Radial phase with radiation
- **Potter D.E.** "Numerical Studies of the Plasma Focus", Phys. Fluids 14, 1911 (1971) — Slug model for radial implosion
- **Bennett W.H.** "Magnetically self-focussing streams", Phys. Rev. 45, 890 (1934) — Bennett equilibrium
- **Bosch H.S. & Hale G.M.** "Improved formulas for fusion cross-sections", Nuclear Fusion 32, 611 (1992) — D-D reactivity
- **Haines M.G. et al.** "Ion kinetic energy and neutron yield", Phys. Rev. Lett. 106, 075002 (2011) — Beam-target neutron model
- **Mather J.W.** "Formation of a high-density deuterium plasma focus", Phys. Fluids 8, 366 (1965) — Mather-type DPF
- **Filipov N.V. et al.** "Dense high-temperature plasma in a non-cylindrical Z-pinch", Nucl. Fusion Suppl. 2, 577 (1962) — Filipov-type DPF
- **Goyon C. et al.** "Instability timing in dense plasma focus", Phys. Plasmas 32, 012702 (2025) — m=0 instability timing formula""")

    return "\n\n---\n\n".join(sections)


def _mhd_narrative(d: dict[str, Any]) -> str:
    backend = d.get("backend", "unknown")
    grid = d.get("grid_shape", (0, 0, 0))
    n_steps = d["n_steps"]

    backend_desc = {
        "metal_plm": "PLM reconstruction + HLL Riemann solver + SSP-RK2 time integration",
        "metal_weno5": "WENO5-Z reconstruction + HLLD Riemann solver + SSP-RK3 time integration",
        "athena": "PPM reconstruction + HLLD Riemann solver (Athena++ C++ engine)",
        "python": "WENO5 reconstruction + HLLD Riemann solver (Python/NumPy engine)",
    }.get(backend, backend)

    rho_max = d.get("rho_max", [])
    B_max = d.get("B_max", [])
    T_max = d.get("T_max", [])
    rho0 = d.get("rho0", 1e-4)

    rho_ratio = float(np.max(rho_max)) / rho0 if len(rho_max) > 0 and rho0 > 0 else 0
    B_peak = float(np.max(B_max)) if len(B_max) > 0 else 0
    T_peak = float(np.max(T_max)) if len(T_max) > 0 else 0

    return f"""## Step 3: MHD Simulation

This simulation uses a **full magnetohydrodynamic (MHD)** solver instead of the \
0D Lee snowplow model. The MHD equations are solved on a {grid[0]}x{grid[1]}x{grid[2]} \
computational grid.

**Solver**: {backend_desc}

The ideal MHD system of conservation laws:

$$\\frac{{\\partial \\rho}}{{\\partial t}} + \\nabla \\cdot (\\rho \\mathbf{{v}}) = 0$$

$$\\frac{{\\partial (\\rho \\mathbf{{v}})}}{{\\partial t}} + \\nabla \\cdot \
\\left(\\rho \\mathbf{{v}} \\mathbf{{v}} - \\frac{{\\mathbf{{B}}\\mathbf{{B}}}}{{\\mu_0}} \
+ p_{{tot}} \\mathbf{{I}}\\right) = 0$$

$$\\frac{{\\partial E}}{{\\partial t}} + \\nabla \\cdot \
\\left[(E + p_{{tot}}) \\mathbf{{v}} - \\frac{{(\\mathbf{{v}} \\cdot \\mathbf{{B}})\\mathbf{{B}}}}{{\\mu_0}}\\right] = 0$$

$$\\frac{{\\partial \\mathbf{{B}}}}{{\\partial t}} - \\nabla \\times (\\mathbf{{v}} \\times \\mathbf{{B}}) \
= -\\nabla \\times (\\eta \\mathbf{{J}})$$

where $p_{{tot}} = p + B^2/(2\\mu_0)$ is total (thermal + magnetic) pressure.

**Results after {n_steps} timesteps**:
- Peak density ratio: $\\rho_{{max}} / \\rho_0 = {rho_ratio:.1f}$
- Peak magnetic field: $|B|_{{max}} = {B_peak:.3f}$ T
- Peak temperature proxy: $T_{{max}} \\sim {T_peak:.0f}$ K"""


def _circuit_coupling_section(d: dict[str, Any], cc: dict) -> str:
    text = """## Circuit-Plasma Coupling

> **Why this matters:** The circuit and the plasma are locked in a feedback loop. \
The plasma's motion changes the circuit's inductance, which changes the current, \
which changes the force on the plasma. Understanding this coupling is essential — \
it's why the current waveform is the primary diagnostic of DPF operation.

The circuit equation with time-varying plasma inductance \
*(Kirchhoff's voltage law applied to the DPF circuit)*:

$$L(t) \\frac{dI}{dt} + I \\frac{dL}{dt} + R(t) I = V_{cap}(t)$$

Each term explained:
- $L \\cdot dI/dt$ — voltage across the inductance (like inertia for current)
- $I \\cdot dL/dt$ — **back-EMF** from the moving plasma. This is the key coupling \
term: as the sheath moves and the inductance increases, it opposes the current
- $R \\cdot I$ — resistive voltage drop (energy lost to heat in cables and plasma)
- $V_{cap}$ — the capacitor voltage driving the whole system

> *During radial implosion, $dL/dt$ spikes because the inductance grows as \
$\\ln(b/r_s)$ — and $r_s$ is shrinking fast. This spike causes the \
characteristic current dip that experimentalists use to confirm a pinch.*

**Energy conservation** — at any instant, the total bank energy is split three ways:

$$E_0 = E_{cap}(t) + E_{ind}(t) + E_{res}(t)$$

$$E_{cap} = \\frac{1}{2} C V^2, \\quad \
E_{ind} = \\frac{1}{2} L I^2, \\quad \
E_{res} = \\int_0^t R I^2 \\, dt'$$

> *$E_{cap}$ = energy still in the capacitor. $E_{ind}$ = energy stored in the \
magnetic field (doing useful work compressing plasma). $E_{res}$ = energy lost \
to resistance (wasted as heat). See the "Energy" panel in the Physics Breakdown tab.*"""

    if d.get("crowbar_t"):
        text += f"""

**Crowbar switch** fires at $t = {d['crowbar_t']:.1f}$ us (first voltage \
zero-crossing). This disconnects the capacitor bank and adds crowbar \
resistance $R_{{cb}} = {cc.get('crowbar_resistance', 0)*1e3:.1f}$ mOhm, \
causing the current to decay as $I \\sim e^{{-Rt/L}}$."""

    return text


def _summary_section(
    d: dict[str, Any], gas: dict, E_kJ: float,
    is_mhd: bool = False, r_pinch_mm: float = 0,
) -> str:
    lines = [
        "## Summary",
        "",
        "| Quantity | Value |",
        "|----------|-------|",
        f"| Peak current | {d['I_peak']:.3f} MA at {d['t_peak']:.1f} us |",
    ]
    if not is_mhd:
        lines.append(f"| Current dip | {d['dip_pct']:.0f}% |")
        lines.append(f"| Pinch radius | {r_pinch_mm:.2f} mm |")
    if is_mhd:
        backend = d.get("backend", "?")
        grid = d.get("grid_shape", (0, 0, 0))
        lines.append(f"| Backend | {backend} |")
        lines.append(f"| Grid | {grid[0]}x{grid[1]}x{grid[2]} |")
    ny = d.get("neutron_yield")
    if ny and ny.get("Y_neutron", 0) > 0:
        bt_pct = ny.get("bt_fraction", 0) * 100
        lines.append(f"| D-D neutron yield | {ny['Y_neutron']:.2e} ({bt_pct:.0f}% BT) |")
        T_b = ny.get("T_bennett_keV", 0)
        if T_b > 0:
            lines.append(f"| Bennett temperature | {T_b:.2f} keV |")
        T_eff = ny.get("T_eff_keV")
        if T_eff and T_b > 0 and T_eff != T_b:
            lines.append(f"| Effective temperature | {T_eff:.2f} keV (rad-corrected) |")
        V_kV = ny.get("V_pinch_kV", 0)
        if V_kV > 1:
            lines.append(f"| Pinch voltage | {V_kV:.0f} kV |")

    lines.extend([
        f"| Bank energy | {E_kJ:.0f} kJ |",
        f"| Gas | {gas['name']} |",
        f"| Simulation | {d['n_steps']} steps in {d['elapsed_s']:.2f}s |",
    ])
    return "\n".join(lines)
