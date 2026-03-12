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

    sections.append(f"""## Step 1: Energy Storage & Circuit

The capacitor bank stores electrostatic energy:

$$E_0 = \\frac{{1}}{{2}} C V_0^2 \
= \\frac{{1}}{{2}} \\times {C_mF:.3f}\\text{{ mF}} \\times ({V0_kV:.0f}\\text{{ kV}})^2 \
= {E_kJ:.0f}\\text{{ kJ}}$$

The natural LC oscillation period (no plasma load):

$$T_{{LC}} = 2\\pi\\sqrt{{L_0 C}} \
= 2\\pi\\sqrt{{{L0_nH:.1f}\\text{{ nH}} \\times {C_uF:.0f}\\text{{ uF}}}} \
= {T_LC:.1f}\\text{{ us}}$$

Short-circuit peak current (theoretical maximum):

$$I_{{sc}} = V_0 \\sqrt{{\\frac{{C}}{{L_0}}}} \
= {V0_kV:.0f}\\text{{ kV}} \\times \\sqrt{{\\frac{{{C_uF:.0f}\\text{{ uF}}}}{{{L0_nH:.1f}\\text{{ nH}}}}}} \
= {I_sc:.2f}\\text{{ MA}}$$

Actual peak: **{d['I_peak']:.3f} MA** ({loading:.0f}% of $I_{{sc}}$) \
--- the deficit is energy absorbed by the plasma load.""")

    if R0_mOhm > 0:
        RESF = cc.get("R0", 0) / np.sqrt(cc["L0"] / cc["C"])
        sections[-1] += f"""

External resistance $R_0 = {R0_mOhm:.1f}$ mOhm \
(RESF = $R_0/\\sqrt{{L_0/C}}$ = {RESF:.3f})."""

    p_torr = sc.get("fill_pressure_Pa", 400) / 133.322
    sections.append(f"""## Step 2: Fill Gas & Electrode Geometry

**Gas**: {gas['name']} ($A = {gas['A']}$, $Z = {gas['Z']}$, \
$m = {gas['m_mol']:.2e}$ kg)

**Fill pressure**: {p_torr:.1f} Torr = {sc.get('fill_pressure_Pa', 400):.0f} Pa

Mass density from ideal gas law at $T = 300$ K:

$$\\rho_0 = \\frac{{P \\cdot m_{{mol}}}}{{k_B \\cdot T}} = {d['rho0']:.3e}\\text{{ kg/m}}^3$$

**Coaxial electrodes**: anode radius $a = {a_mm:.1f}$ mm, \
cathode radius $b = {b_mm:.1f}$ mm

$$\\ln\\left(\\frac{{b}}{{a}}\\right) = \\ln\\left(\\frac{{{b_mm:.1f}}}{{{a_mm:.1f}}}\\right) \
= {ln_ba:.3f}$$""")

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

    sections.append(f"""## Step 3: Axial Rundown (Phase 2)

The current sheath forms at the insulator and is accelerated axially by the \
$\\mathbf{{J}} \\times \\mathbf{{B}}$ magnetic piston force:

$$F_{{mag}} = \\frac{{\\mu_0}}{{4\\pi}} \\ln\\left(\\frac{{b}}{{a}}\\right) (f_c I)^2$$

where $f_c = {fc:.2f}$ is the **current fraction** (not all circuit current flows \
in the sheath --- some leaks through the plasma behind it).

The sheath sweeps fill gas, accumulating mass:

$$m(z) = f_m \\cdot \\rho_0 \\cdot \\pi(b^2 - a^2) \\cdot z$$

where $f_m = {fm:.3f}$ is the **mass fraction** swept up.

The equation of motion (Newton's second law for the growing mass):

$$\\frac{{d}}{{dt}}[m(z) \\cdot v] = F_{{mag}} - F_{{pressure}}$$

$$m \\frac{{dv}}{{dt}} = F_{{mag}} - p_0 \\pi(b^2 - a^2) - v \\frac{{dm}}{{dt}}$$

The third term $v \\cdot dm/dt$ is the **rocket equation drag** --- as the sheath \
picks up stationary gas, momentum is consumed accelerating it.

As the sheath moves, the plasma inductance grows linearly:

$$L_p(z) = \\frac{{\\mu_0}}{{2\\pi}} \\ln\\left(\\frac{{b}}{{a}}\\right) \\cdot z$$

This increasing inductance loads the circuit, causing $dL/dt > 0$ back-EMF \
that opposes current growth.

**Result**: Sheath reaches anode end ($z_0 = {L_anode_mm:.0f}$ mm) at \
$t \\approx {t_rundown:.1f}$ us with $I \\approx {abs(I_at_rundown):.3f}$ MA.""")

    fmr = sc.get("radial_mass_fraction", fm)
    pcf = sc.get("pinch_column_fraction", 1.0)
    z_f_mm = L_anode_mm * pcf

    sections.append(f"""## Step 4: Radial Implosion (Phase 3)

At the anode end, the sheath curves inward. A cylindrical shock wave \
implodes radially from $r = b$ toward the axis. Only a fraction of the \
anode length participates: $z_f = {pcf:.2f} \\times {L_anode_mm:.0f} = {z_f_mm:.0f}$ mm.

The radial $\\mathbf{{J}} \\times \\mathbf{{B}}$ force on a cylindrical current sheet:

$$F_{{rad}} = \\frac{{\\mu_0}}{{4\\pi}} \\frac{{(f_c I)^2 z_f}}{{r_s}}$$

Note the $1/r_s$ dependence --- the force **intensifies** as the shock converges \
toward the axis, producing rapid acceleration.

The slug model equation of motion:

$$M_{{slug}} \\frac{{dv_r}}{{dt}} = -F_{{rad}} + F_{{back}} - v_r \\frac{{dM}}{{dt}}$$

where $M_{{slug}} = f_{{mr}} \\rho_0 \\pi(b^2 - r_s^2) z_f$ with $f_{{mr}} = {fmr:.2f}$.

Opposing the implosion is adiabatic back-pressure from compressed gas:

$$p_{{back}}(r_s) = p_0 \\left(\\frac{{b}}{{r_s}}\\right)^{{2\\gamma}}$$

For $\\gamma = 5/3$ (monatomic), this rises steeply as $r_s \\to 0$.

The plasma inductance now grows logarithmically:

$$L_p = L_{{axial}} + \\frac{{\\mu_0}}{{2\\pi}} z_f \\ln\\left(\\frac{{b}}{{r_s}}\\right)$$

$$\\frac{{dL}}{{dt}} = -\\frac{{\\mu_0}}{{2\\pi}} \\frac{{z_f \\cdot v_r}}{{r_s}} > 0 \
\\quad (v_r < 0)$$

This rapid inductance increase produces the **characteristic current dip** --- \
the circuit cannot supply current fast enough to overcome the back-EMF.""")

    if d["dip_pct"] > 0:
        sections[-1] += f"""

**Result**: Current dips from {d['I_pre_dip']:.3f} to \
{d['I_dip']:.3f} MA (**{d['dip_pct']:.0f}% dip**) at $t = {d['t_dip']:.1f}$ us."""

    sp = d["snowplow_obj"]
    r_pinch_mm = sp.shock_radius * 1e3 if sp else 0

    sections.append(f"""## Step 5: Pinch & Stagnation (Phase 4)

The shock converges to a minimum radius $r_{{min}} \\approx {r_pinch_mm:.2f}$ mm \
(~{r_pinch_mm / a_mm * 100:.0f}% of anode radius). At this point:

- Kinetic energy converts to thermal energy (stagnation heating)
- The plasma column reaches peak temperature and density
- Neutron/X-ray emission occurs (for D-D or high-Z gases)

The **Bennett pinch equilibrium** balances magnetic pressure against plasma pressure:

$$\\frac{{\\mu_0 I^2}}{{8\\pi}} = N_l k_B (T_e + T_i)$$

where $N_l$ is the line density (particles per unit length).

**Compression ratio**: $b / r_{{min}} = {b_mm / r_pinch_mm:.0f}:1$""")

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
    if ny and ny["Y_neutron"] > 0:
        sections.append(f"""## D-D Neutron Yield Estimate

For deuterium fill gas, the thermonuclear neutron yield is estimated from:

$$Y_n = \\frac{{1}}{{2}} n_i^2 \\langle\\sigma v\\rangle V_{{pinch}} \\tau_{{pinch}}$$

**Bennett temperature** (from pinch equilibrium):

$$T_B = \\frac{{\\mu_0 I^2}}{{8\\pi N_l \\cdot 2 k_B}} = {ny['T_bennett_keV']:.2f}\\text{{ keV}}$$

Ion density: $n_i = {ny['n_ion']:.2e}$ m$^{{-3}}$

D-D reactivity (Bosch-Hale): $\\langle\\sigma v\\rangle = {ny['sigma_v']:.2e}$ m$^3$/s

Pinch volume: ${ny['V_pinch_cm3']:.2f}$ cm$^3$, confinement time: ${ny['tau_ns']:.0f}$ ns

**Estimated yield: {ny['Y_neutron']:.2e} neutrons per shot**

*Note: This is a thermonuclear estimate only. Beam-target neutrons from \
instability-accelerated ions can dominate in real devices.*""")

    sections.append(_circuit_coupling_section(d, cc))
    sections.append(_summary_section(d, gas, E_kJ, is_mhd=False, r_pinch_mm=r_pinch_mm))

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

The circuit equation with time-varying plasma inductance:

$$L(t) \\frac{dI}{dt} + I \\frac{dL}{dt} + R(t) I = V_{cap}(t)$$

The term $I \\cdot dL/dt$ is the **back-EMF** from the moving sheath/plasma. \
During radial implosion, $dL/dt$ spikes, causing the current dip.

Energy partition at any time:

$$E_0 = E_{cap}(t) + E_{ind}(t) + E_{res}(t)$$

$$E_{cap} = \\frac{1}{2} C V^2, \\quad \
E_{ind} = \\frac{1}{2} L I^2, \\quad \
E_{res} = \\int_0^t R I^2 \\, dt'$$"""

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
    if ny and ny["Y_neutron"] > 0:
        lines.append(f"| D-D neutron yield | {ny['Y_neutron']:.2e} |")
        lines.append(f"| Bennett temperature | {ny['T_bennett_keV']:.2f} keV |")

    lines.extend([
        f"| Bank energy | {E_kJ:.0f} kJ |",
        f"| Gas | {gas['name']} |",
        f"| Simulation | {d['n_steps']} steps in {d['elapsed_s']:.2f}s |",
    ])
    return "\n".join(lines)
