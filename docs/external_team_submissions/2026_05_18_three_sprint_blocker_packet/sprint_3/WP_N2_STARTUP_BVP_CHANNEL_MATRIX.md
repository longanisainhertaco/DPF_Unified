# WP-N2 Startup BVP Channel Matrix

Date: 2026-05-19
Branch: `codex/corpus`
Lane: Allowed parallel lane 3 (WP-N2 startup BVP channel matrix) of
`docs/FIRST_PRINCIPLES_CODEX_AUDIT_SPRINT2_FOLLOWUP_2026_05_19.md`.

## Scope and Authority

This packet is a source-grounded research packet only. It does not promote
validation, acceptance, or first-principles authority, and it marks no runtime
feature `implemented`: no code or tests are part of this diff. It does not edit
any Sprint 2.2-owned file and does not run the periodic audit.

`can_support_first_principles_acceptance = false` for every channel in this
packet. The seeded startup mode (`seeded_layer`) and the uniform/profile
startup modes remain rejected for first-principles claims (see section 7).

Every physics claim below cites a local `KnowledgeReference/` file with a path,
line range, and the equation/figure/table referenced. Where no local source
exists for a channel, the channel is marked `blocked` and listed in section 4.

## Purpose

The whole-shot first-principles run must begin at the start of the voltage
discharge, not at the end of rundown. The current runtime accepted-startup
contract (`src/dpf/first_principles/startup_bvp.py`) declares 18 required
startup channels but the runtime only produces engineering-candidate values
for them through the CIV/Paschen scaffold (`src/dpf/experimental/civ_breakdown.py`),
which is explicitly `civ_paschen_startup_scaffold` /
`civ_paschen_gas_coefficients_source_packets_missing` /
`not_validation_evidence`. This packet enumerates every whole-shot startup
channel, classifies each against the local corpus, names the runtime fields
each channel must produce/consume, and specifies the fail-closed negative
controls and the proposed runtime startup-packet schema.

---

## 1. Source-Backed Findings (per channel)

Each channel below records: what the local corpus supports, the governing
relation with its KR citation (equation/figure, symbols, units, validity
range), and what the corpus does NOT supply.

### 1.1 Breakdown

The DPF starts with gas breakdown along the exterior of the cylindrical
insulator at the base of the anode; this surface discharge takes "from a few
to a hundred nanoseconds" and is non-equilibrium kinetic (avalanche,
streamers)
[KR: gribkov-2007-pf1000-jphysd-part2.md L62-66, prose, "first stage is gas
breakdown developing along the exterior of a cylindrical insulator ... takes
from a few to a hundred nanoseconds"].

The Townsend mechanism is governed by the primary ionization coefficient
`alpha` and the secondary coefficient `gamma`. The self-sustaining Townsend
breakdown condition for an inhomogeneous field is
`gamma * integral_0^d ( alpha(x) * exp( integral_0^x alpha(x') dx' ) ) dx = 1`
[KR: the-influence-of-the-magnetic-field-on-dc-and-the-impulse-breakdown-of-noble-gases-3.md
L196-198, Eq. (6)]. The streamer-mechanism alternative is
`integral_0^d alpha(x) dx = 10.5`
[KR: the-influence-of-the-magnetic-field-on-dc-and-the-impulse-breakdown-of-noble-gases-3.md
L201-203, Eq. (7)].
Symbols/units: `alpha` primary (impact) ionization coefficient [1/m]; `gamma`
dimensionless secondary coefficient; `d` inter-electrode (or insulator surface)
path length [m]; `x` distance along the field [m].
Validity range: `gamma` ranges 1e-3..1e-8 and depends on electrode material,
treatment, and gas
[KR: the-influence-of-the-magnetic-field-on-dc-and-the-impulse-breakdown-of-noble-gases-3.md
L116-117, prose]. Townsend dominates when the gap is comparable
to the electron mean free path; streamer dominates when the gap greatly
exceeds it
[KR: the-influence-of-the-magnetic-field-on-dc-and-the-impulse-breakdown-of-noble-gases-3.md
L218-224, prose].
The first Townsend coefficient by definition is
`alpha(x) = n0 * integral_{eps_i}^{inf} sigma_i(eps) * v * f(eps) deps`
[KR: the-influence-of-the-magnetic-field-on-dc-and-the-impulse-breakdown-of-noble-gases-3.md
L130-133, Eq. (1)]; with a Maxwellian `f(eps)` this reduces to
`alpha(Te) = (4 * M * sigma_i0) / (R * sqrt(pi) * p) * ((eps_i + 2*Te)/Te) *
exp(-eps_i/Te)`
[KR: the-influence-of-the-magnetic-field-on-dc-and-the-impulse-breakdown-of-noble-gases-3.md
L150-156, Eq. (3)], where `p` gas pressure, `M` gas molar mass,
`sigma_i0` effective ionization cross section at energy `eps_i`, `R` Rydberg
constant, `Te` free-electron temperature [same units as `eps_i`].

CRITICAL CONTRADICTION FROM THE CORPUS. The local DPF-specific source warns
that Paschen-type physics is "fragile" for DPFs: the Paschen curve refers to
ions traveling to the cathode and releasing secondaries in a feedback loop,
"this is not the type of breakdown that occurs in DPFs and thus, once a plasma
is formed, such physics should no longer apply"
[KR: effect-of-current-sheath-initiation-on-the-radial-collapse-and-energetic-particle-accelera-b2e95b88.md
L631-639, prose]. Therefore the Townsend/Paschen equations above are
`candidate` for a DPF surface-flashover BVP, not `supported`, because the
corpus explicitly says the canonical Paschen feedback loop does not describe
DPF insulator breakdown. A DPF surface-flashover model needs a surface-physics
closure the local corpus does not provide.

What the corpus does NOT supply: a reviewed surface-flashover BVP for a DPF
insulator; a closed equation set that produces the runtime initial fields
(E/B/J, density, Te/Ti) directly from breakdown; numerical DPF-specific
`alpha`, `gamma`, `sigma_i0` values for D2/H2/Ne/Ar against alumina or pyrex.

### 1.2 Flashover

Surface flashover is the DPF breakdown channel: "Breakdown of the test gas
occurs along the insulator and this breakdown essentially determines [the
sheath]"; the breakdown "takes place in several places around the insulator"
and forms "a radial, striated light pattern ... a few tenths of a microsecond
after application of the breakdown voltage"
[KR: design-and-construction-of-a-dense-plasma-focus-device-12205ba4.md
L512-514 and L583-589, prose]. PF-1000 confirms the same: the first stage is a
surface discharge along the cylindrical insulator
[KR: gribkov-2007-pf1000-jphysd-part2.md L62-66, prose].

Energy-density scaling exists in the corpus: Kies estimated an upper limit of
energy density into the sheath of about 100 J/cm^2 that decreases with
increasing insulator radius
[KR: the-dense-plasma-focus-a-versatile-dense-pinch-for-diverse-applications.md
L538-541, prose]. Symbol/unit: surface energy density [J/cm^2]; decreases with
insulator radius [m].

Pressure regime ties breakdown morphology to sheath quality: at higher
pressure the sheath is filamentary (detrimental to neutrons); at lower
pressure breakdown is uniform (better pinch)
[KR: the-dense-plasma-focus-a-versatile-dense-pinch-for-diverse-applications.md
L527-534, prose]. The optimum operating pressure across
successful DPFs centers near ~10 mbar regardless of device energy
[KR: the-dense-plasma-focus-a-versatile-dense-pinch-for-diverse-applications.md
L569-573, prose].

What the corpus does NOT supply: an equation for surface-flashover delay,
striation-to-uniform transition time, or the flashover voltage as a closed
function of insulator material, length, radius, gas, and applied voltage. The
~100 J/cm^2 figure is an upper-limit estimate, not a runtime model.

### 1.3 Preionization

Preionization is documented as a real DPF startup control. A beta-source
(Ni-63 mesh, or depleted U-238) placed near the insulator sleeve preionizes
the gas; measured neutron yield rose by up to 25% (Ni-63) and by (50 +/- 5)%
(U-238), with a broadened emission pressure range and improved shot-to-shot
reproducibility
[KR: the-dense-plasma-focus-a-versatile-dense-pinch-for-diverse-applications.md
L1490-1517, prose]. The mechanism stated by the corpus: improved ionization at
insulator breakdown creates a more uniform current sheet, and a larger active
volume
[KR: the-dense-plasma-focus-a-versatile-dense-pinch-for-diverse-applications.md
L1518-1530, prose].

What the corpus does NOT supply: a quantitative preionization model (seed
electron density [m^-3] or ionization fraction as a function of source
activity, geometry, and time). Preionization in the corpus is an experimental
intervention with measured yield deltas, not a runtime initial-condition
generator.

### 1.4 Secondary Emission

The secondary coefficient `gamma` is defined as the number of new free
secondary electrons per primary avalanche; it appears in the Townsend
condition Eq. (6) of section 1.1
[KR: the-influence-of-the-magnetic-field-on-dc-and-the-impulse-breakdown-of-noble-gases-3.md
L113-117, prose]. Range: `gamma` is 1e-3..1e-8 and depends on electrode
material and surface treatment [same lines].

A modelable boundary condition exists in the corpus. Secondary electron
emission from positive-ion impact on the cathode is "a crucial mechanism that
sustains the negative corona discharge" and is modeled as an electron flux
boundary condition `Gamma_e = gamma * c_p * |u_p|` [mol m^-2 s^-1], with
`gamma` a dimensionless number between 1e-3 and 1e-2
[KR: theory-and-finite-element-simulation-methodology-of-gas-discharge-plasmas.md
L1187-1196, prose with the flux relation]. Symbols/units: `c_p` positive-ion
molar concentration [mol m^-3]; `u_p` ion drift speed [m s^-1]; `Gamma_e`
electron number flux [mol m^-2 s^-1]; `gamma` dimensionless.

What the corpus does NOT supply: secondary-emission coefficients for DPF
electrode/insulator materials (copper, pyrex, alumina) under D2/H2/Ne/Ar at
DPF voltages; ion-induced electron-emission yields specific to the DPF anode
or insulator surface. The cited `gamma` ranges are generic gas-discharge
values, not DPF-material-specific. (A related general kinetic ion-induced
electron-emission model exists at
`KnowledgeReference/general-kinetic-ion-induced-electron-emission-model-for-metallic-walls-applied-to-biased-z-pinch.md`
but is for biased z-pinch metallic walls, not the DPF insulator surface, so it
is a candidate cross-scope reference only and is not cited as support here.)

### 1.5 Surface Plasma

The corpus describes the surface plasma qualitatively. After flashover the
current flows through the plasma on the insulator surface from the inner
electrode along the insulator to the outer electrode end plate; this is the
"surface discharge" stage of kinetic character
[KR: design-and-construction-of-a-dense-plasma-focus-device-12205ba4.md
L605-607, prose; gribkov-2007-pf1000-jphysd-part2.md L62-66, prose]. The
initial breakdown has a multifilamentary but cylindrically symmetric pattern;
the filaments must blend into a uniform radially symmetric sheath about 1
microsecond after high-voltage application for a strong focus
[KR: design-and-construction-of-a-dense-plasma-focus-device-12205ba4.md
L585-589 and L646-651, prose]. The corpus also gives the inverse-pinch phase
in which `F = i dL x B` (per-segment magnetic force) drives the surface plasma
radially outward from the insulator to the cathode
[KR: design-and-construction-of-a-dense-plasma-focus-device-12205ba4.md
L632-639, prose with the force relation `F = i dL x B`].

What the corpus does NOT supply: surface-plasma sheet density, thickness,
conductivity, or temperature as a closed runtime field set. The 1-microsecond
striation-to-uniform timescale is an experimental observation.

### 1.6 Initial E / B / J

Initial E-field: a radial E-field exists between the coaxial electrodes when
high voltage is applied; the corpus states the coaxial geometry and the
applied-voltage breakdown but does not give a closed initial-E field for the
BVP
[KR: design-and-construction-of-a-dense-plasma-focus-device-12205ba4.md
L508-512, prose].

Initial B-field and circuit: the corpus gives the azimuthal field from the
discharge current `B_theta = mu * I / (2 * pi * r)` [T]
[KR: fully-electromagnetic-hybrid-pic-fluid-dpf-neutron-yield-acb71fa9.md
L748-751, Eq. (34)] and the external-circuit current equation
`d(L0 * I)/dt = V0 - r0*I - U_DPF - (1/C0) * integral I dt`
[KR: fully-electromagnetic-hybrid-pic-fluid-dpf-neutron-yield-acb71fa9.md
L753-757, Eq. (35)], with `L0 = 110 nH`, `V0 = 10 kV`,
`r0 = 12 mOhm` for that LLNL-like device
[KR: fully-electromagnetic-hybrid-pic-fluid-dpf-neutron-yield-acb71fa9.md
L757-760, prose].
Symbols/units: `I` circuit current [A]; `r` radius [m]; `mu` permeability
[H m^-1]; `L0` circuit inductance [H]; `V0` charge voltage [V]; `r0` stray
resistance [Ohm]; `C0` capacitance [F]; `U_DPF` device voltage drop [V].
Validity range: the hybrid PIC-fluid model uses Eq. (34)/(35) for the
implosion-to-pinch phase of an LLNL-like compact DPF, NOT for the breakdown
phase.

Initial J: the surface current density `J` flows through the insulator-surface
plasma at breakdown
[KR: design-and-construction-of-a-dense-plasma-focus-device-12205ba4.md
L601-605, prose, Fig. 2a "the plasma carries current density indicated by the
symbol J"], but no closed initial `J` distribution for the BVP is given.

What the corpus does NOT supply: a closed, source-derived initial E, B, and J
field set at the start of the voltage discharge. Eq. (34)/(35) are circuit/
boundary relations for the implosion phase, not breakdown-phase initial
fields. At initial breakdown the corpus and the scaffold both note `B ~ 0`
(see section 1.11 and `civ_breakdown.py` mechanism note).

### 1.7 Density / Species

For the END-OF-RUNDOWN state (NOT the start of the shot) the hybrid PIC-fluid
paper gives explicit values for an LLNL-like compact DPF: prefilled deuterium
background number density `n0 = 6.7e22 m^-3` at temperature `T1 ~ 0.026 eV`
(~300 K); a pre-accelerated current-sheath population with sheath number
density `ns,0 = 3.3e23 m^-3` in a thin slab of axial thickness `delta_z ~ 1 mm`
adjacent to the anode, at temperature `T2 ~ 7.2e5 K` (~62 eV) with an axial
drift `vd ~ 1.1e5 m s^-1`
[KR: fully-electromagnetic-hybrid-pic-fluid-dpf-neutron-yield-acb71fa9.md
L661-682, prose and Table 1; Fig. 6]. Species: deuterium (D2 / D ions).
Symbols/units: `n0`, `ns,0` number density [m^-3]; `vd` drift speed [m s^-1];
temperatures [K] or [eV].
Validity range: these are end-of-rundown / start-of-implosion conditions,
explicitly "approximates the end of rundown state seen in fully kinetic
simulations" and "The simulation initiates at the rundown phase conclusion"
[KR: fully-electromagnetic-hybrid-pic-fluid-dpf-neutron-yield-acb71fa9.md
L682-690, prose]. They are NOT whole-shot startup densities.

What the corpus does NOT supply: density and species fields at the START of
the voltage discharge (the breakdown BVP). The values above are an
end-of-rundown handoff state, which is the engineering-only startup mode in
the runtime contract, not the whole-shot startup.

### 1.8 Ionization

The avalanche/ionization buildup is governed by `alpha` and the Townsend
condition (section 1.1, Eqs. 1, 3, 6). The hydrodynamic gas-discharge source
term for electrons is
`R_e = alpha*c_e*|u_e| - eta*c_e*|u_e| - NA*beta_ep*c_e*c_p + R0 + R_ph`
[KR: theory-and-finite-element-simulation-methodology-of-gas-discharge-plasmas.md
L256-258, Eq. (6)], with `eta` the attachment coefficient [1/m] (zero for
non-electronegative D2/H2), `beta_ep` electron-ion recombination [m^3 s^-1],
`R0` background ionization rate, and `R_ph` photoionization rate
[mol m^-3 s^-1]
[KR: theory-and-finite-element-simulation-methodology-of-gas-discharge-plasmas.md
L262-292, prose definitions].
Symbols/units: `c_e`, `c_p` molar concentrations [mol m^-3]; `u_e` electron
drift speed [m s^-1]; `alpha`, `eta` [1/m]; `NA` Avogadro number [1/mol].

The DPF-specific source notes the ionization path length `Liz(P)` scales with
pressure, with the optimal insulator length scaling such that
`Liz(P) / Li = 2.4` at optimal pressure across the three insulator lengths
studied
[KR: effect-of-current-sheath-initiation-on-the-radial-collapse-and-energetic-particle-accelera-b2e95b88.md
L656-662, prose]. Symbol/unit: `Liz` ionization path length [m]; `Li`
insulator length [m]; ratio dimensionless.

What the corpus does NOT supply: a closed ionization-fraction field for the
DPF startup BVP. `R_e` is a general gas-discharge hydrodynamic source term;
the corpus provides no DPF-specific `alpha(E/p)`, `beta_ep`, `R_ph` for
D2/H2/Ne/Ar at DPF voltages, and the `Liz/Li = 2.4` ratio is an empirical fit,
not an ionization model.

### 1.9 Te / Ti

Free-electron temperature in the breakdown region:
`Te = k*T = xi * lambda * e * E = xi * lambda * e * U / d`, valid only for a
homogeneous or pseudo-homogeneous field
[KR: the-influence-of-the-magnetic-field-on-dc-and-the-impulse-breakdown-of-noble-gases-3.md
L160-167, Eq. (4)]; the mean free-electron energy is
`eps_mean = 0.8 * e * E * lambda / sqrt(delta)`
[KR: the-influence-of-the-magnetic-field-on-dc-and-the-impulse-breakdown-of-noble-gases-3.md
L169-172, Eq. (5)]. Symbols/units: `xi` thermalization form
factor (dimensionless); `lambda` mean free path [m]; `E` electric field
[V m^-1]; `U` applied voltage [V]; `d` inter-electrode distance [m]; `e`
elementary charge [C]; `delta` ratio of electron mass to gas molar mass.
The DPF-specific source states the initial plasma is "typically a few eV" and
the breakdown physics depends on the evolving plasma temperature; it assumes
`Te ~ 4 eV` for all fill pressures in its analysis
[KR: effect-of-current-sheath-initiation-on-the-radial-collapse-and-energetic-particle-accelera-b2e95b88.md
L643-655, prose]. The end-of-rundown sheath value is `T2 ~ 62 eV` (section
1.7).

Ti: the corpus gives the end-of-rundown background at ~300 K and the sheath at
~62 eV (with the electron fluid initialized `Te = Ti` per cell)
[KR: fully-electromagnetic-hybrid-pic-fluid-dpf-neutron-yield-acb71fa9.md
L676-682, prose].

What the corpus does NOT supply: a DPF-specific closed Te/Ti field for the
breakdown BVP. Eq. (4)/(5) require a homogeneous field (the DPF coaxial gap is
not homogeneous), and `Te ~ 4 eV` is an analysis assumption. The 62 eV / 300 K
values are end-of-rundown handoff values, not whole-shot startup.

### 1.10 Sheath Surface

The corpus describes the initial sheath qualitatively: a plasma sheath "some
millimeters thick" lifts off the insulator
[KR: unlimited-release-printed-september-2009-alegra-hedp-simulations-of-the-dense-plasma-focus.md
L255-259, prose; sand2009-6373-b93aec67.md L317-321, prose]. The end-of-rundown
sheath axial thickness from ion-density profiles is `delta_sheath ~ 0.15..0.20
cm` (8-10 z-cells at `delta_z = 0.02 cm`)
[KR: fully-electromagnetic-hybrid-pic-fluid-dpf-neutron-yield-acb71fa9.md
L703-708, prose]. The Yee/staggered grid placement is explicit: densities and
temperatures at grid nodes; `B_theta` at cell centers; `(Er, Ez)` and
`(Jr, Jz)` at face midpoints
[KR: fully-electromagnetic-hybrid-pic-fluid-dpf-neutron-yield-acb71fa9.md
L607-614, prose; Fig. 4].

What the corpus does NOT supply: a closed sheath-surface initial state (mask,
thickness, density, conductivity, velocity) for the breakdown BVP. The 0.15-
0.20 cm thickness is an end-of-rundown measured value.

### 1.11 Handoff Interval

The corpus is explicit and consistent that an MHD whole-shot run cannot start
with an arbitrary seed and must instead import a first-principles breakdown
description. ALEGRA "is completing work on a new capability to directly import
data derived from these PIC simulations and use it to initiate our MHD
simulations"
[KR: unlimited-release-printed-september-2009-alegra-hedp-simulations-of-the-dense-plasma-focus.md
L268-272, prose]. The problem was otherwise "initialized with a uniform,
room-temperature deuterium gas fill, and a thin layer of higher-temperature
gas in a thin layer along the insulator. MHD DPF simulations require choosing
this seed ionized gas rather arbitrarily; we chose a thin layer of 1 eV
temperature"
[KR: sand2009-6373-b93aec67.md L470-475, prose]. The accepted alternative is
to import ion and electron densities, temperatures, and magnetic field from a
PIC sheath calculation
[KR: sand2009-6373-b93aec67.md L682-690, prose]. The first phase to be
described is breakdown along the insulator, then lift-off (II), then run-down
(III)
[KR: sand2009-6373-b93aec67.md L317-323, prose; Figure 1].

This is the source basis for the runtime contract's two accepted modes
(`imported_pic_sheath_state`, `surface_breakdown_bvp`) and for rejecting the
arbitrary seed. The handoff interval is the time window
[breakdown-onset, sheath-liftoff-complete] handed from the breakdown model to
the MHD rundown solver.

What the corpus does NOT supply: a numerical handoff-interval definition (the
exact `t_start`, `t_handoff` and tolerance) or a same-device reviewed PIC
import payload. The "1 eV thin layer" is described by the source itself as
arbitrary and is NOT a whole-shot startup.

---

## 2. Channel Classification Table

Status definitions:
- `supported` - the local corpus supplies a governing relation AND enough to
  produce/consume the runtime field for a DPF startup BVP.
- `candidate` - the local corpus supplies a relation or qualitative basis, but
  it is generic, cross-scope, contradicted for DPFs, or lacks the closure /
  numerical values to drive the runtime field. Usable to seed engineering runs
  only; cannot support whole-shot first-principles acceptance.
- `blocked` - the local corpus supplies no usable source for a DPF startup BVP
  of this channel.

| # | Channel | Status | Basis (KR citation) | Why not higher |
|---|---------|--------|---------------------|----------------|
| 1 | breakdown | candidate | Townsend Eq. (6)/(7), `alpha` Eq. (1)/(3) [KR: the-influence-of-the-magnetic-field-on-dc-and-the-impulse-breakdown-of-noble-gases-3.md L130-203] | Corpus says Paschen/Townsend feedback "should no longer apply" to DPFs [KR: effect-of-current-sheath-...-b2e95b88.md L631-639]; no DPF surface-flashover closure |
| 2 | flashover | candidate | Surface-discharge description; ~100 J/cm^2 energy-density limit [KR: the-dense-plasma-focus-a-versatile-dense-pinch-...md L527-541; gribkov-2007-...md L62-66] | Qualitative + an upper-limit estimate; no closed flashover-delay/voltage model |
| 3 | preionization | candidate | Measured yield deltas with Ni-63/U-238 preionizers [KR: the-dense-plasma-focus-a-versatile-dense-pinch-...md L1490-1530] | No quantitative seed-density model; experimental intervention, not an IC generator |
| 4 | secondary emission | candidate | `gamma` definition + flux BC `Gamma_e = gamma c_p u_p` [KR: the-influence-of-the-magnetic-field-...-3.md L113-117; theory-and-finite-element-...gas-discharge-plasmas.md L1187-1196] | `gamma` ranges generic (1e-3..1e-2 / 1e-3..1e-8); no DPF-material (Cu/pyrex/alumina) values |
| 5 | surface plasma | candidate | Surface-discharge / inverse-pinch `F = i dL x B` [KR: design-and-construction-...-12205ba4.md L583-639; gribkov-2007-...md L62-66] | Qualitative only; no closed surface-plasma field set |
| 6 | initial E / B / J | candidate | `B_theta = mu I/(2 pi r)` Eq. (34); circuit Eq. (35) [KR: fully-electromagnetic-hybrid-pic-fluid-...-acb71fa9.md L748-757] | Eqs. are implosion-phase circuit/boundary relations, not breakdown-phase initial fields |
| 7 | density / species | candidate | End-of-rundown `n0=6.7e22`, `ns,0=3.3e23 m^-3`, D2 [KR: fully-electromagnetic-hybrid-pic-fluid-...-acb71fa9.md L661-690, Table 1] | Values are end-of-rundown handoff, not start-of-shot breakdown densities |
| 8 | ionization | candidate | `alpha` Eqs. (1)/(3); source term `R_e` Eq. (6); `Liz/Li=2.4` [KR: the-influence-...-3.md L130-156; theory-...gas-discharge-plasmas.md L256-292; effect-of-current-sheath-...-b2e95b88.md L656-662] | Generic gas-discharge source term + empirical ratio; no DPF-specific ionization-fraction field |
| 9 | Te / Ti | candidate | `Te = xi lambda e U/d` Eq. (4); `eps = 0.8 e E lambda/sqrt(delta)` Eq. (5); `Te~4 eV` assumption [KR: the-influence-...-3.md L160-172; effect-of-current-sheath-...-b2e95b88.md L643-655] | Eq. (4) needs a homogeneous field (DPF gap is not); `Te~4 eV` is an analysis assumption |
| 10 | sheath surface | candidate | End-of-rundown `delta_sheath~0.15-0.20 cm`; Yee grid placement [KR: fully-electromagnetic-hybrid-pic-fluid-...-acb71fa9.md L607-614, L703-708] | End-of-rundown measured value, not a breakdown-BVP sheath state |
| 11 | handoff interval | candidate | PIC-to-MHD import requirement; reject arbitrary seed [KR: sand2009-6373-b93aec67.md L317-323, L470-475, L682-690; unlimited-release-...alegra-hedp-...md L268-272] | No numerical handoff-interval definition or same-device reviewed PIC payload |

Summary counts: `supported` = 0, `candidate` = 11, `blocked` = 0.

Note on the absence of `blocked` rows. The local corpus contains at least a
relation or qualitative basis for every one of the 11 channels, so no channel
is fully source-empty. However, NO channel reaches `supported`: every channel
lacks the DPF-specific closure or numerical values needed to drive the runtime
field, and channel 1 (breakdown) is additionally contradicted for DPFs by the
local corpus itself. The whole-shot first-principles startup BVP is therefore
blocked as a whole even though no single channel is `blocked`. Section 4 lists
the specific missing parameters that hold every channel at `candidate`.

This classification is consistent with the existing runtime contract:
`src/dpf/experimental/civ_breakdown.py` already self-declares
`model_role = civ_paschen_startup_scaffold`,
`source_status = civ_paschen_gas_coefficients_source_packets_missing`,
`validation_status = not_validation_evidence`, and
`src/dpf/first_principles/startup_breakdown.py` packetizes it as
`candidate_civ_paschen_breakdown_audit_engineering_only` with
`can_support_first_principles_acceptance = False`.

---

## 3. Runtime Fields Required (produce / consume per channel)

The runtime startup contract already enumerates 18 required channels in
`REQUIRED_STARTUP_CHANNELS` (`src/dpf/first_principles/startup_bvp.py`
L93-112). The table below maps each WP-N2 physics channel to the concrete
state fields it must PRODUCE (write into the startup packet) and CONSUME (read
from device/gas/circuit inputs), plus the grid centering required by the Yee
staggered grid
[KR: fully-electromagnetic-hybrid-pic-fluid-dpf-neutron-yield-acb71fa9.md
L607-614, Fig. 4].

| Channel | Consumes (inputs) | Produces (state fields) | Grid centering |
|---------|-------------------|-------------------------|----------------|
| breakdown | device geometry, insulator length/material, gas species/pressure/temperature, bank voltage, `alpha`/`gamma` packet | `breakdown_onset_time_s`, `breakdown_mechanism`, `breakdown_path_length_m` | scalar (run-level) |
| flashover | insulator length/radius/material, surface energy-density limit, applied voltage | `flashover_complete_time_s`, `surface_energy_density_J_cm2`, `striation_to_uniform_time_s` | scalar (run-level) |
| preionization | preionizer model (source activity/geometry) or "none" | `preionization_seed_density_m3`, `preionization_ionization_fraction` | node-centered (n,T grid) |
| secondary emission | electrode/insulator material, ion flux at surface | `secondary_emission_coefficient_gamma`, surface electron flux BC `Gamma_e_mol_m2_s` | face-centered (boundary) |
| surface plasma | flashover output, circuit current | `surface_plasma_mask`, `surface_sheet_thickness_m`, `surface_sheet_conductivity_S_m` | mask on nodes; sigma node-centered |
| initial E / B / J | bank voltage, circuit `L0/V0/r0/C0`, geometry | `initial_E_field_V_m` (Er,Ez), `initial_B_field_T` (B_theta), `initial_J_A_m2` (Jr,Jz) | E,J at face midpoints; B at cell centers |
| density / species | gas species/pressure/temperature, breakdown ionization | `total_number_density_m3`, `ion_density_m3`, `electron_density_m3`, `species_name`, `ionization_fraction` | node-centered |
| ionization | `alpha(E/p)` packet, recombination/photoionization rates | `ionization_fraction` field, `ionization_source_rate_mol_m3_s` | node-centered |
| Te / Ti | applied field, gas, mean free path | `electron_temperature_K`, `ion_temperature_K` | node-centered |
| sheath surface | surface-plasma output, end-of-rundown handoff (engineering mode) | `sheath_mask`, `sheath_thickness_m`, `sheath_drift_velocity_m_s`, `sheath_density_m3` | mask + node-centered; velocity at face midpoints |
| handoff interval | breakdown/flashover/sheath outputs, MHD solver readiness | `handoff_start_time_s`, `handoff_end_time_s`, `handoff_tolerance_s`, `handoff_mode` | scalar (run-level) |

Cross-cutting consistency fields every startup packet must also produce (these
correspond to the existing required channels
`charge_current_divb_energy_consistency` and
`source_paths_hashes_units_and_review`,
`src/dpf/first_principles/startup_bvp.py` L110-111):
`charge_consistency_check`, `current_continuity_check`, `divB_check`,
`energy_consistency_check`, `source_references` (list of KR path + line range),
`source_packet_hashes`, `units` map, `evidence_status`, `source_scope`.

The runtime already exposes startup field carriers in
`FirstPrinciples3DDeck` (`src/dpf/first_principles/runner.py` L143-211:
`background_density_m3`, `initial_ionization_fraction`,
`electron_temperature_K`, `ion_temperature_K`, `initial_E_x_V_m`,
`initial_B_z_T`, `startup_payload`) and assembles them in
`FirstPrinciples3DDeck.startup_packet()` (`runner.py` L401-470). Those carriers
are the consume/produce surface this matrix targets; today they are populated
with engineering-candidate defaults, not source-backed BVP outputs.

---

## 4. Missing Parameters (no KR source for a DPF startup BVP)

Every channel is `candidate` (section 2) because of one or more missing
parameters below. None of these has a local `KnowledgeReference/` source for a
DPF startup BVP; each must be supplied by a tracked verified extract packet
before the corresponding channel can move to `supported`.

| # | Missing parameter | Channel(s) blocked | What is needed |
|---|-------------------|--------------------|----------------|
| M1 | DPF surface-flashover BVP closure (insulator-surface physics) | breakdown, flashover, surface plasma | A reviewed equation set that produces initial E/B/J/n/Te/Ti along the insulator from applied voltage; corpus explicitly says canonical Paschen does not apply [KR: effect-of-current-sheath-...-b2e95b88.md L631-639] |
| M2 | `alpha`, `gamma`, `sigma_i0`, `eta`, `beta_ep`, `R_ph` numerical values for D2/H2/Ne/Ar at DPF voltages | breakdown, ionization, secondary emission | DPF-specific gas-coefficient packets (the scaffold's `_GAS_DB` values in `civ_breakdown.py` L120-135 are flagged `*_source_packets_missing`) |
| M3 | Secondary-emission coefficients for DPF materials (Cu anode, pyrex/alumina insulator) | secondary emission | Material-specific ion-induced electron-emission yields for the DPF surface, not generic gas-discharge `gamma` |
| M4 | Quantitative preionization model | preionization | Seed electron density [m^-3] / ionization fraction vs preionizer source activity, geometry, time |
| M5 | Closed flashover delay / voltage / striation-to-uniform timescale | flashover | A model giving `flashover_complete_time_s` and flashover voltage as a function of insulator material/length/radius, gas, applied voltage |
| M6 | Breakdown-phase initial E and J field distributions | initial E/B/J | A source-derived E(r,z) and J(r,z) at the start of the discharge; Eq. (34)/(35) are implosion-phase relations only |
| M7 | Start-of-shot density/species and Te/Ti fields | density/species, Te/Ti, sheath surface | The corpus only supplies end-of-rundown handoff values [KR: fully-electromagnetic-hybrid-pic-fluid-...-acb71fa9.md L661-690]; start-of-discharge fields are absent |
| M8 | Numerical handoff-interval definition and a same-device reviewed PIC import payload | handoff interval | `t_start`, `t_handoff`, tolerance, and an imported PIC sheath payload for the SAME device being simulated [KR: sand2009-6373-b93aec67.md L470-475 calls the alternative seed "arbitrary"] |
| M9 | DPF-specific homogeneous-field validity for `Te` Eq. (4) | Te/Ti | Eq. (4) is valid only for a homogeneous/pseudo-homogeneous field [KR: the-influence-...-3.md L162-167]; the DPF coaxial gap is inhomogeneous, so a DPF-valid `Te` relation is missing |

---

## 5. Proposed Tests and Fail-Closed Negative Controls

These are PROPOSED test designs. No tests are implemented in this packet. Each
test asserts the channel FAILS CLOSED when its source packet OR its runtime
field is missing. The runtime contract already has matching positive scaffolds
(`negative_test_policy` in `build_startup_bvp_packet`,
`src/dpf/first_principles/startup_bvp.py` L278-287) and existing tests in
`tests/test_first_principles_startup_bvp.py` and
`tests/test_startup_breakdown_audit.py`; the designs below extend that
coverage to the per-channel granularity of this matrix.

### 5.1 Per-channel missing-source negative controls

For each of the 11 channels, one negative control:

- N1 `test_breakdown_channel_blocked_without_flashover_closure` - with no M1
  surface-flashover BVP packet attached, the startup packet for a
  `surface_breakdown_bvp` mode must yield
  `status != accepted_startup_bvp_packet` and the `startup_channel_status`
  entry for `breakdown_or_flashover_model` must be
  `candidate_input_only_not_acceptance` or `missing_or_blocked`.
- N2 `test_flashover_channel_blocked_without_flashover_model` - missing M5
  flashover model: `surface_flashover_equations` payload field reported
  `missing_payload_channel` and the packet does not promote.
- N3 `test_preionization_channel_candidate_only` - a preionization input
  present without M4 quantitative model leaves `preionization_state` at
  `candidate_input_only_not_acceptance`, never `accepted`.
- N4 `test_secondary_emission_blocked_without_material_gamma` - missing M3
  DPF-material `gamma`: `secondary_emission_or_material_model` payload field
  `missing_payload_channel`; packet blocked.
- N5 `test_surface_plasma_blocked_without_closure` - missing M1: no
  `surface_plasma_mask` produced; channel `missing_or_blocked`.
- N6 `test_initial_field_blocked_without_breakdown_phase_fields` - missing M6:
  `initial_electric_field` / `initial_current_density_distribution` channels
  not `accepted`.
- N7 `test_density_species_blocked_without_start_of_shot_fields` - only
  end-of-rundown handoff values present (M7): the packet for a whole-shot
  request must not pass; `source_scope` mismatch flagged.
- N8 `test_ionization_blocked_without_gas_coefficients` - missing M2: ionization
  channel `candidate_input_only_not_acceptance`.
- N9 `test_te_ti_blocked_without_dpf_valid_relation` - `Te` derived from the
  homogeneous-field Eq. (4) is rejected for a DPF (M9): a startup packet that
  declares `Te` from Eq. (4) without a DPF-valid relation must not promote.
- N10 `test_sheath_surface_blocked_without_breakdown_bvp` - missing M1/M7: no
  `sheath_mask` produced for a whole-shot start.
- N11 `test_handoff_interval_blocked_without_numerical_definition` - missing M8:
  `sheath_liftoff_and_handoff_interval` channel not `accepted`; no
  `handoff_start_time_s`/`handoff_end_time_s` in the packet.

### 5.2 Mode-level negative controls (extend existing coverage)

- N12 `test_seeded_layer_startup_always_rejected` - `mode = seeded_layer` must
  return `status = rejected_startup_mode_for_first_principles` even when every
  channel is declared accepted (mirrors the existing
  `test_seeded_layer_rejection_is_immune_to_declared_channels`).
- N13 `test_uniform_and_profile_startup_rejected` - `source_backed_candidate_uniform`
  and `source_backed_profile` modes must be rejected.
- N14 `test_end_rundown_mode_cannot_support_whole_shot` - `source_backed_end_rundown_sheath`
  is engineering-only and must not pass the whole-shot gate (mirrors existing
  `test_engineering_only_modes_cannot_support_whole_shot`).
- N15 `test_arbitrary_1eV_thin_layer_rejected` - a startup declaring the
  SAND2009 "1 eV thin layer along the insulator" arbitrary seed
  [KR: sand2009-6373-b93aec67.md L470-475] must be classified non-promoting.
- N16 `test_unreviewed_pic_import_rejected` - `imported_pic_sheath_state` with
  `evidence_status != reviewed/accepted` must not promote.
- N17 `test_civ_paschen_scaffold_cannot_promote` - a
  `candidate_civ_paschen_breakdown_audit` packet always carries
  `can_support_first_principles_acceptance = False` and cannot lift any channel
  to `accepted` (mirrors existing
  `test_candidate_breakdown_audit_cannot_promote_startup`).
- N18 `test_cross_scope_startup_rejected` - a startup payload whose
  `source_scope` differs from the declared run scope must fail closed.

### 5.3 Missing-runtime-field negative controls

- N19 `test_missing_field_payload_blocks_acceptance` - for
  `imported_pic_sheath_state`, omitting any of the 16 mode-required payload
  fields (`MODE_REQUIRED_PAYLOADS`, `src/dpf/first_principles/startup_bvp.py`
  L114-132) yields `startup_payload_incomplete` and a blocked headline status.
- N20 `test_missing_consistency_check_blocks_acceptance` - omitting any of
  `charge_consistency_check`, `current_continuity_check`, `divB_check`,
  `energy_consistency_check` blocks the packet.
- N21 `test_missing_source_hashes_block_acceptance` - omitting `source_references`,
  `source_packet_hashes`, or `units` blocks the packet (the
  `source_paths_hashes_units_and_review` required channel).

Fail-closed principle: a passing positive test for any channel requires BOTH
its source packet AND all of its runtime fields. The negative controls assert
that removing either one flips the channel to non-accepted and blocks the
overall startup packet. The startup packet must never auto-promote a missing
channel to `accepted`; the existing `_startup_channel_statuses` already does
this (`startup_bvp.py` L294-315) and the tests above lock that behavior.

---

## 6. Exact Implementation Recommendations

These are recommendations for a FUTURE diff (no code/tests here). They are
consistent with the existing fail-closed contract in
`src/dpf/first_principles/startup_bvp.py` and must not be marked `implemented`
until code and tests ship together.

### 6.1 Proposed runtime startup-packet schema

A `StartupPacket` produced by the runtime and consumed by the certificate gate
(`startup_packet_accepted` channel, `src/dpf/first_principles/certificate_gate.py`
L53, L99). Field names are proposed; types and units are mandatory.

```text
StartupPacket
  status: str                 # accepted_startup_bvp_packet | blocked_* | rejected_*
  mode: str                   # imported_pic_sheath_state | surface_breakdown_bvp
                               #   | source_backed_end_rundown_sheath (engineering)
                               #   | seeded_layer | source_backed_candidate_uniform
                               #   | source_backed_profile (all rejected)
  evidence_status: str         # reviewed | accepted | accepted_same_scope_source
                               #   | not_reviewed
  source_scope: str            # must match the run's validation/source scope
  can_support_first_principles_acceptance: bool   # false unless every gate met

  channels: map[channel_name -> ChannelEntry]     # all 11 WP-N2 channels
    ChannelEntry
      status: str              # supported | candidate | blocked
      source_references: list[SourceRef]          # required if status != blocked
      runtime_fields: map[field_name -> Quantity] # produced fields
      validity_range: str
  consistency:
    charge_consistency_check: bool
    current_continuity_check: bool
    divB_check: bool
    energy_consistency_check: bool
  provenance:
    source_packet_hashes: list[str]               # non-empty required
    units: map[field_name -> str]
    reviewer_metadata: ReviewerMetadata | null
  handoff: HandoffContract                        # see 6.2

SourceRef = { path: str, lines: str, equation_or_figure: str }
Quantity  = { value: float | array, unit: str, centering: str }
```

Acceptance rule (matches `build_startup_bvp_packet`,
`src/dpf/first_principles/startup_bvp.py` L211-228): the packet can support a
whole-shot first-principles claim ONLY when `mode` is in
`ACCEPTED_STARTUP_MODES`, `evidence_status` is reviewed/accepted, every one of
the 18 `REQUIRED_STARTUP_CHANNELS` is `accepted`, the mode-required payload
fields are all present, every `consistency` check is true, and
`source_packet_hashes` is non-empty. Any missing element forces a blocked
status. The packet must not be reachable as `accepted` while any of M1-M9
(section 4) is unresolved.

### 6.2 Handoff-interval contract

```text
HandoffContract
  handoff_mode: str            # imported_pic_sheath_state | surface_breakdown_bvp
  handoff_start_time_s: float  # t_start: start of the voltage discharge
  handoff_end_time_s: float    # t_handoff: sheath-liftoff complete; MHD rundown begins
  handoff_tolerance_s: float   # allowed jitter on t_handoff
  source_references: list[SourceRef]              # PIC import or flashover BVP source
  imported_state_hash: str | null                 # required for imported_pic_sheath_state
  same_device: bool            # true only if the imported PIC payload is the same device
  monotonic: bool              # t_start < t_handoff strictly
  can_support_whole_shot_acceptance: bool          # false unless all of the above hold
```

Handoff-interval rules:
1. `handoff_start_time_s < handoff_end_time_s` strictly; otherwise the contract
   fails closed.
2. For `imported_pic_sheath_state`, `imported_state_hash` must be non-null and
   `same_device` must be true (the PIC sheath import must be the SAME device
   being simulated, per [KR: sand2009-6373-b93aec67.md L682-690], which states
   ALEGRA imports ion/electron densities, temperatures, and magnetic field from
   the PIC calculation).
3. For `surface_breakdown_bvp`, the handoff source must be a reviewed
   surface-flashover BVP (M1); until M1 exists this mode cannot reach
   `accepted`.
4. The arbitrary "1 eV thin layer along the insulator" seed
   [KR: sand2009-6373-b93aec67.md L470-475] must NOT satisfy the contract; it
   is the rejected `seeded_layer` mode.
5. The end-of-rundown handoff state (`n0=6.7e22`, `ns,0=3.3e23 m^-3`,
   `T2~62 eV`, `delta_z~1 mm`, `vd~1.1e5 m/s`
   [KR: fully-electromagnetic-hybrid-pic-fluid-...-acb71fa9.md L661-690]) is a
   valid ENGINEERING handoff (`source_backed_end_rundown_sheath`) but
   `can_support_whole_shot_acceptance` must remain false for it, because it
   starts the run at the end of rundown, not at the start of the discharge.

### 6.3 Integration points (existing code, do not edit in this diff)

- `src/dpf/first_principles/startup_bvp.py` - the `StartupPacket` above
  generalizes the existing `build_startup_bvp_packet` output; the 11 WP-N2
  channels map onto the existing 18 `REQUIRED_STARTUP_CHANNELS`.
- `src/dpf/first_principles/runner.py` L401-470 - `startup_packet()` is the
  production site; the proposed `channels` map and `HandoffContract` would be
  emitted here.
- `src/dpf/first_principles/certificate_gate.py` L99 - `startup_packet_accepted`
  maps to the `startup_bvp` channel; `rejected_startup_mode_for_first_principles`
  is already a `BLOCKING_UPSTREAM_STATUS` (L81).
- `src/dpf/experimental/civ_breakdown.py` - remains an engineering scaffold;
  the proposed schema does not promote it. Its `_GAS_DB` coefficients stay
  flagged `civ_paschen_gas_coefficients_source_packets_missing` until M2 is
  resolved.

---

## 7. Do-Not-Promote Notes

1. Seeded startup remains REJECTED for first-principles claims. The
   `seeded_layer` mode, and the `source_backed_candidate_uniform` and
   `source_backed_profile` modes, must stay in `REJECTED_STARTUP_MODES`
   (`src/dpf/first_principles/startup_bvp.py` L87-91) and must always return
   `status = rejected_startup_mode_for_first_principles`. The corpus itself
   calls the thin-layer seed "arbitrary"
   [KR: sand2009-6373-b93aec67.md L470-475]; an arbitrary seed cannot support a
   first-principles whole-shot claim.

2. The CIV/Paschen scaffold (`src/dpf/experimental/civ_breakdown.py`) must NOT
   be promoted to a startup BVP. It is self-declared
   `civ_paschen_startup_scaffold` /
   `civ_paschen_gas_coefficients_source_packets_missing` /
   `not_validation_evidence`, and `build_candidate_startup_breakdown_audit`
   packetizes it as `candidate_civ_paschen_breakdown_audit_engineering_only`
   with `can_support_first_principles_acceptance = False`
   (`src/dpf/first_principles/startup_breakdown.py` L120-128). This packet does
   not change that. The Townsend/Paschen equations (section 1.1) are
   `candidate` because the local DPF source states canonical Paschen feedback
   "should no longer apply" to DPFs
   [KR: effect-of-current-sheath-...-b2e95b88.md L631-639].

3. The end-of-rundown sheath state is an ENGINEERING handoff only. The hybrid
   PIC-fluid values [KR: fully-electromagnetic-hybrid-pic-fluid-...-acb71fa9.md
   L661-690] describe the start of the implosion phase, explicitly "the
   rundown phase conclusion". Used as a whole-shot startup they would skip
   breakdown, flashover, and rundown. `source_backed_end_rundown_sheath` must
   stay in `ENGINEERING_ONLY_STARTUP_MODES` with
   `can_support_whole_shot_acceptance = false`.

4. This packet promotes no validation, no acceptance, and no first-principles
   authority. `can_support_first_principles_acceptance` is `false` for every
   channel and for the packet as a whole. No runtime feature is marked
   `implemented`: no code or tests are in this diff.

5. No channel may be implemented from an inferred formula. Every channel stays
   `candidate` until its missing parameters (M1-M9, section 4) are supplied by
   a local `KnowledgeReference/` source or a tracked verified extract packet
   that attaches the exact equation, units, symbol map, and validity range.

---

## Source Reference Index

All citations are local `KnowledgeReference/` files (or named source code).

- `KnowledgeReference/the-influence-of-the-magnetic-field-on-dc-and-the-impulse-breakdown-of-noble-gases-3.md`
  - L113-117 (`alpha`, `gamma`, `eta` definitions and `gamma` range)
  - L130-133 Eq. (1) first Townsend coefficient by definition
  - L150-156 Eq. (3) `alpha(Te)` with Maxwellian distribution
  - L160-167 Eq. (4) `Te = xi lambda e U/d`
  - L169-172 Eq. (5) mean free-electron energy
  - L196-203 Eq. (6) Townsend and Eq. (7) streamer breakdown conditions
  - L218-224 Townsend vs streamer dominance regime
- `KnowledgeReference/effect-of-current-sheath-initiation-on-the-radial-collapse-and-energetic-particle-accelera-b2e95b88.md`
  - L631-639 Paschen physics "fragile" for DPFs / "should no longer apply"
  - L631-635 three pressure regimes (low/medium/high)
  - L643-655 initial plasma "a few eV", `Te~4 eV` analysis assumption
  - L656-662 ionization path length `Liz(P)`, `Liz/Li = 2.4`
- `KnowledgeReference/the-dense-plasma-focus-a-versatile-dense-pinch-for-diverse-applications.md`
  - L512-541 insulator breakdown phase; ~100 J/cm^2 energy-density limit
  - L527-534 filamentary (high p) vs uniform (low p) breakdown
  - L569-573 optimum pressure ~10 mbar across devices
  - L1490-1530 preionization (Ni-63, U-238) measured yield deltas
- `KnowledgeReference/gribkov-2007-pf1000-jphysd-part2.md`
  - L62-66 first stage: surface discharge along the insulator, few-to-100 ns
- `KnowledgeReference/design-and-construction-of-a-dense-plasma-focus-device-12205ba4.md`
  - L508-514 breakdown along the insulator determines the sheath
  - L583-607 multifilamentary symmetric breakdown; surface current path
  - L632-651 inverse-pinch `F = i dL x B`; 1-microsecond uniformization
- `KnowledgeReference/fully-electromagnetic-hybrid-pic-fluid-dpf-neutron-yield-acb71fa9.md`
  - L607-614 Yee staggered grid placement (n,T nodes; B center; E,J midpoints)
  - L661-690 end-of-rundown initial densities/temperatures/drift; Table 1
  - L703-708 end-of-rundown sheath thickness 0.15-0.20 cm
  - L748-757 Eq. (34) `B_theta = mu I/(2 pi r)`; Eq. (35) external circuit
- `KnowledgeReference/sand2009-6373-b93aec67.md`
  - L317-323 DPF phases: breakdown, lift-off, run-down, pinch; Figure 1
  - L470-475 arbitrary "1 eV thin layer along the insulator" seed
  - L682-690 ALEGRA imports PIC ion/electron densities, temperatures, B-field
- `KnowledgeReference/unlimited-release-printed-september-2009-alegra-hedp-simulations-of-the-dense-plasma-focus.md`
  - L255-272 breakdown/lift-off/run-down sequence; PIC import to initiate MHD
- `KnowledgeReference/theory-and-finite-element-simulation-methodology-of-gas-discharge-plasmas.md`
  - L256-292 Eq. (6) electron source term `R_e`; coefficient definitions
  - L1187-1196 secondary-emission flux BC `Gamma_e = gamma c_p u_p`
- Source code (named, not edited): `src/dpf/first_principles/startup_bvp.py`,
  `src/dpf/first_principles/startup_breakdown.py`,
  `src/dpf/first_principles/runner.py`,
  `src/dpf/first_principles/certificate_gate.py`,
  `src/dpf/experimental/civ_breakdown.py`.
