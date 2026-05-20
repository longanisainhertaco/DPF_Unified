# WP-N5 Closure Registry Source Audit

Date: 2026-05-19
Branch: `codex/corpus`
Lane: Allowed parallel lane 4 (WP-N5 closure registry source audit)
Status: `source_backed_research_packet_not_validation`
Acceptance: `can_support_first_principles_acceptance=false` for every closure below.

## 0. Scope and Integrity Statement

This packet is a *source audit* of the eight first-principles physics closures
required before WP-N5 runtime work begins. It prepares the closure-packet
registry. It does **not** implement a closure, does **not** mark any closure
`implemented`, does **not** edit Sprint 2.2-owned files, and does **not**
promote validation, acceptance, or first-principles authority.

Integrity rules applied:

- Every physics claim cites a local `KnowledgeReference/` file with a path,
  line range, and equation identifier. No outside material is used as authority.
- Where the local corpus lacks a source, the closure is marked `blocked` and the
  missing source is listed explicitly in section 4.
- PlasmaPy is treated as an *external candidate cross-check only*, never as a
  source authority. This matches the existing policy in
  `src/dpf/first_principles/plasmapy_audit.py`.
- No expected closure value is asserted from training data.

The eight audited closures map to existing `REQUIRED_EFFECTS` keys in
`src/dpf/first_principles/closure_packet.py:37-48`:

| WP-N5 closure name        | `closure_packet.py` effect key            |
| ------------------------- | ------------------------------------------ |
| EOS / thermodynamics      | `eos_thermodynamics`                       |
| Radiation                 | `radiation_losses`                         |
| Ablation / impurity       | `impurity_electrode_ablation`              |
| Anomalous resistance      | `restrike_anomalous_resistance` (shared)   |
| Restrike                  | `restrike_anomalous_resistance` (shared)   |
| Electron inertia          | (no effect key — see F-EI below)           |
| Collision / stopping      | `electrical_thermal_transport` + new       |
| Beam-target coupling      | `beam_target_coupling`                     |

Finding F-EI: the current `REQUIRED_EFFECTS` tuple has **no electron-inertia
effect key**. Electron inertia (the generalized-Ohm `m_e dJ/dt` term and the
electron skin depth `c/omega_pe`) is a distinct closure and is currently not
registered. It must be added as a registry row (see sections 2.6, 6).

---

## 1. Source-Backed Findings (per closure)

Each closure below records: governing source equation with KR citation, units,
symbol map, validity range, and current code status. Local KR citations use an
exact `KnowledgeReference/<file>.md:<line-or-ranges>` path inside the citation.

### 1.1 EOS / Thermodynamics

Source equation (engineering ideal-gas form, in code):
`src/dpf/fluid/eos.py:32-67` — `IdealEOS`:

```
p_i = (rho / m_i) * k_B * T_i              [Pa]
p_e = Z * (rho / m_i) * k_B * T_e          [Pa]
e_i = p_i / ((gamma - 1) * rho)            [J/kg]
c_s = sqrt(gamma * (p_i + p_e) / rho)      [m/s]
```

KR support for the *closure-class decision* (not the ideal-gas formula itself):
`[KR: KnowledgeReference/unlimited-release-printed-september-2009-alegra-hedp-simulations-of-the-dense-plasma-focus.md:188,215,333-369]`.
The ALEGRA HEDP DPF study states the DPF EOS path used LANL **Sesame**
tabulated EOS for deuterium "down to a lowest meaningful density of .01 kg/m3"
(L351-352) and the analytic **QEOS** model with gaseous-deuterium parameters
(L357-360). This establishes that a credible DPF whole-shot EOS is a
tabular/QEOS closure, **not** the constant-gamma ideal gas.

Symbol map: `rho` mass density [kg/m^3]; `m_i` ion mass [kg]; `k_B` Boltzmann
constant [J/K]; `T_i`,`T_e` ion/electron temperature [K]; `Z` charge state [-];
`gamma` adiabatic index [-]; `p` pressure [Pa]; `e` specific internal energy
[J/kg]; `c_s` sound speed [m/s].

Units: SI throughout (the in-code `IdealEOS` is dimensionally consistent).

Validity range: ideal-gas `IdealEOS` is valid only for a fully ionized,
non-degenerate, optically thin gas with a fixed `Z` and fixed `gamma`. It is
**out of validity** for (a) the cold neutral fill before breakdown, (b) the
partially ionized rundown sheath, (c) the dense degenerate pinch core, and
(d) any radiation-pressure-significant regime. The ALEGRA reference's
"lowest meaningful density" floor (L352) confirms tabular EOS itself has a
documented low-density validity edge.

Current code status: `IdealEOS` exists and is dimensionally sound, but the
registry effect `eos_thermodynamics` is correctly `blocked` in
`closure_packet.py:80-85` with missing channels `qEOS_or_tabular_EOS`,
`low_density_validity`, `verification_tests`. No tabular/QEOS closure with a
KR-cited equation set exists in the corpus. **Status: blocked.**

Note: `src/dpf/fluid/eos.py:48` uses a bare `np.maximum(rho, 1e-30)` floor.
This is flagged by the `telemetry.apply_floor()` policy in the project
`CLAUDE.md` and should be routed through `telemetry.apply_floor()` when the EOS
closure is implemented. This is recorded as a code-hygiene blocker, not a
physics blocker.

### 1.2 Radiation

Source equations (NRL Plasma Formulary 2019, verified verbatim):

- Bremsstrahlung (free-free), hydrogen-like plasma:
  `[KR: KnowledgeReference/2019nrlplasma-formulary-037290d4.md:4732-4736]`
  `P_Br = 1.69e-32 * N_e * T_e^(1/2) * sum_Z[Z^2 N(Z)]   [W/cm^3]`,
  `N_e` in cm^-3, `T_e` in eV.
- Bremsstrahlung optical depth:
  `[KR: KnowledgeReference/2019nrlplasma-formulary-037290d4.md:4737-4740]`
  `tau = 5.0e-38 * N_e * N_i * Z^2 * g * L * T^(-7/2)`, `g ~= 1.2`.
- Inverse-bremsstrahlung absorption coefficient:
  `[KR: KnowledgeReference/2019nrlplasma-formulary-037290d4.md:4741-4748]`.
- Recombination (free-bound) radiation:
  `[KR: KnowledgeReference/2019nrlplasma-formulary-037290d4.md:4749-4755]`.
- Cyclotron radiation:
  `[KR: KnowledgeReference/2019nrlplasma-formulary-037290d4.md:4756-4758]`
  `P_c = 6.21e-28 * B^2 * N_e * T_e   [W/cm^3]`.

Symbol map: `N_e`,`N_i` electron/ion number density [cm^-3]; `T_e` electron
temperature [eV] in eq.(30)/(33)/(34); `Z` charge state [-]; `N(Z)` population
of charge state Z [cm^-3]; `g` Gaunt factor [-]; `L` path length [cm]; `B`
magnetic field [Gauss in NRL CGS]; `tau` optical depth [-]; `P` volumetric
power [W/cm^3].

Units: NRL eq.(30)-(34) are CGS-eV. The in-code `BREM_COEFF = 1.569e-40`
(`src/dpf/radiation/bremsstrahlung.py:53`) is the documented SI conversion of
the CGS `1.69e-32` and is internally derivation-traced in the module docstring
(L18-24). The cyclotron SI coefficient `5.35e-24` is the documented conversion
of NRL `6.21e-28` (`improved_radiation.py:13-16`).

Validity range: NRL eq.(30) is the optically *thin* bremsstrahlung loss; it
holds while `tau << 1` (eq.(31) gives `tau`). Once `tau` approaches 1 the
optically-thin loss over-counts and a transport closure (Rosseland/FLD) is
required. eq.(30) bakes in a temperature-averaged Gaunt factor; the in-code
`gaunt_factor=1.2` default and the `gaunt_factor_thermal` fit
(`improved_radiation.py`) are fit to Karzas & Latter (1961) tables which are
**not in the local KR corpus**.

Current code status: bremsstrahlung volumetric loss eq.(30) and cyclotron
eq.(34) are source-grounded against the local NRL KR. **But** the radiation
*loss-and-transport closure as a whole* is `blocked` in `closure_packet.py:138-143`
(missing `loss_model_or_bound`, `opacity_or_diffusion_decision`, `energy_ledger`).
The flux-limited-diffusion transport scaffold
(`src/dpf/radiation/transport.py:33-67`) self-declares
`rosseland_kramers_fld_source_packet_missing` — the Levermore-Pomraning limiter
mechanics exist but the Rosseland/Kramers opacity closure is **not** source-closed
by the local corpus. `line_radiation.py:59` self-declares
`unknown_provenance_empirical_fits`. `qmf_suppression.py:53` self-declares
`free_free_suppression_source_missing`. **Status: candidate** for the
bremsstrahlung/cyclotron *volumetric loss term only*; **blocked** for the
opacity/transport/line closure required for a whole-shot radiation ledger.

### 1.3 Ablation / Impurity

Source equation (engineering form, in code):
`src/dpf/atomic/ablation.py:122-192` — `ablation_source`:

```
P_ohmic = eta * J^2                  [W/m^3]
S_rho   = efficiency * P_ohmic       [kg/(m^3 s)]
```

KR support: **none for the ablation efficiency coefficient.** The module itself
declares `ABLATION_SOURCE_STATUS = "ablation_efficiency_source_packet_missing"`
(`ablation.py:59`). The cited references (Bruzzone & Aranchuk 2003; Vikhrev &
Korolev 2007; Lee & Serban 1996; `ablation.py:27-30`) are **not present in the
local `KnowledgeReference/` corpus** — they are docstring citations only. The
Ohmic-heating driver `eta * J^2` is dimensionally sound but the conversion
efficiency (`COPPER_ABLATION_EFFICIENCY = 5.0e-5 kg/J`,
`TUNGSTEN_ABLATION_EFFICIENCY = 2.0e-5 kg/J`; `ablation.py:49,54`) is an
unverified empirical constant.

Symbol map: `eta` resistivity [Ohm m]; `J` current density [A/m^2]; `P_ohmic`
volumetric Ohmic power [W/m^3]; `efficiency` ablation yield [kg/J]; `S_rho`
volumetric mass source [kg/(m^3 s)]; `m_atom` electrode atomic mass [kg].

Units: SI, dimensionally consistent.

Validity range: the linear `dm/dt = efficiency * P_surface` model is documented
in-code as valid for "moderate power densities ~10^8 to 10^11 W/m^2"
(`ablation.py:99-103`); plasma shielding reduces the effective efficiency at
higher fluence. No KR source bounds this range.

Current code status: `impurity_electrode_ablation` is `blocked` in
`closure_packet.py:144-149` (missing `ablation_source_model`,
`impurity_transport`, `electrode_material_uq`). The module is explicitly an
"impurity-source scaffold" (`ablation.py:84-86`). **Status: blocked.**

### 1.4 Anomalous Resistance

Source equation (engineering form, in code):
`src/dpf/turbulence/anomalous.py:312-329` — `_compute_eta_anom`:

```
eta_anom = alpha * m_e * omega_pe / (n_e * e^2)     [Ohm m]
omega_pe = sqrt(n_e * e^2 / (epsilon_0 * m_e))      [rad/s]
```

with threshold models: ion-acoustic (`v_d > c_s`), LHDI
(`v_d > (m_e/m_i)^(1/4) * v_ti`), Buneman (`v_d > v_te`), and CIV
(`v_bulk > v_crit`).

KR support: **partial.** The NRL formulary gives an anomalous ion-sound
collision rate `[KR: KnowledgeReference/2019nrlplasma-formulary-037290d4.md:2706-2710]`:
`nu* ~= omega_pe * W~/(kT) = 5.64e4 * n_e^(1/2) * W~/(kT) s^-1`, where `W~` is
the total energy of waves with `omega/K < v_Ti`. This NRL row gives the
*functional structure* (anomalous rate proportional to `omega_pe`) but the DPF
turbulence parameter `alpha ~ 0.01-0.1` and the threshold-model selection are
**not** source-closed by the local corpus. The module self-declares
`microinstability_source_packets_missing` (`anomalous.py:44`). The threshold
references (Buneman 1959; Sagdeev 1966; Davidson & Gladd 1975; Haines 2011;
Krall & Trivelpiece 1973; `anomalous.py:26-31`) are docstring citations and are
**not present in `KnowledgeReference/`**. The CIV `v_crit` table
(`anomalous.py:380-389`, Alfven 1954/Brenning 1992) is also not KR-cited.

Symbol map: `alpha` turbulence parameter [-]; `m_e` electron mass [kg];
`omega_pe` electron plasma frequency [rad/s]; `n_e` electron density [m^-3];
`e` elementary charge [C]; `epsilon_0` vacuum permittivity [F/m]; `v_d` electron
drift speed [m/s]; `c_s` ion sound speed [m/s]; `v_ti`,`v_te` ion/electron
thermal speed [m/s]; `v_crit` CIV threshold velocity [m/s].

Units: SI, dimensionally consistent.

Validity range: each microinstability threshold has its own regime. NRL
transport-validity criterion (6) explicitly states classical transport
coefficients are valid "only when ... anomalous transport processes owing to
microinstabilities are negligible"
`[KR: KnowledgeReference/2019nrlplasma-formulary-037290d4.md:3382-3383]` — i.e. anomalous
resistivity is exactly the regime where the *classical* closure fails, so the
two closures are mutually exclusive by validity and the runtime must record
which one is active. The `alpha` magnitude band is not KR-bounded.

Current code status: shares the `restrike_anomalous_resistance` effect, `blocked`
in `closure_packet.py:166-171`. The module is a "microinstability resistivity
scaffold" (`anomalous.py:43`). **Status: blocked** (functional form partially
NRL-supported; coefficient `alpha` and threshold selection unsourced).

### 1.5 Restrike

Source equation: **none.** There is no restrike (post-pinch current-dip
recovery) closure equation in `src/dpf/` and no restrike governing equation in
`KnowledgeReference/`. `restrike` appears only as config/preset wording
(`src/dpf/config.py`, `src/dpf/presets.py`), as comparator/waveform-phase
metadata (`src/dpf/first_principles/comparator_uq.py`,
`waveform_phase.py`), and as a `blocked` registry effect.

KR support: none located. (`grep` over `KnowledgeReference/*.md` returns no
restrike closure equation.)

Validity range: not applicable — not simulated.

Current code status: shares the `restrike_anomalous_resistance` effect, `blocked`
in `closure_packet.py:166-171` (missing `restrike_model`). **Status: blocked**
and `not_simulated`. Any current-dip or post-pinch claim must be blocked
(`closure_packet.py:256` `anomalous_resistance_or_restrike_claim_rejection_required`).

### 1.6 Electron Inertia

Source equation: **none located in `KnowledgeReference/`.** The generalized
Ohm's law electron-inertia term and the electron skin depth `c/omega_pe` are
standard, but no KR file in the corpus presents the generalized-Ohm electron-
inertia closure as a cited equation. `grep -i "electron inertia"` over
`KnowledgeReference/*.md` returns the NRL formulary files and hybrid/Hall-MHD
papers as keyword hits, but none supplies a `m_e dJ/dt` closure equation in a
form directly citable for implementation.

In-code: electron inertia appears only in *diagnostics* —
`src/dpf/diagnostics/plasma_regime.py`, `evidence_manifest.py`, and
`src/dpf/validation/kr_targets.py` reference the concept but there is no
electron-inertia *closure operator*.

Symbol map (standard, for the future packet): `m_e` electron mass [kg]; `J`
current density [A/m^2]; `n_e` electron density [m^-3]; `e` elementary charge
[C]; `omega_pe` electron plasma frequency [rad/s]; `c` speed of light [m/s];
electron skin depth `d_e = c / omega_pe` [m].

Validity range: electron inertia matters when the electron skin depth `d_e` is
resolved by the grid and on timescales `~ 1/omega_pe`; for a whole-shot MHD
deck it is typically a sub-grid term. Cannot be bounded without a KR source.

Current code status: **not registered.** No `electron_inertia` key exists in
`REQUIRED_EFFECTS` (`closure_packet.py:37-48`). **Status: blocked** and
`not_registered` — must be added to the registry (see section 6, F-EI).

### 1.7 Collision / Stopping

This closure splits into two parts: (a) thermal collision frequencies and
classical transport (collision closure), and (b) fast-ion stopping power
(stopping closure).

#### 1.7a Collision (thermal transport)

Source equations (NRL Plasma Formulary 2019, verified verbatim):

- Coulomb logarithm definition and electron-ion branch:
  `[KR: KnowledgeReference/2019nrlplasma-formulary-037290d4.md:3024-3065]` —
  `lambda = ln(r_max/r_min)`; electron-ion branch (L3045-3059):
  `lambda_ei = 23 - ln(n_e^(1/2) Z T_e^(-3/2))` for
  `T_i m_e/m_i < T_e < 10 Z^2 eV`, and
  `lambda_ei = 24 - ln(n_e^(1/2) T_e^(-1))` for `10 Z^2 eV < T_e`.
- Transverse Spitzer resistivity:
  `[KR: KnowledgeReference/2019nrlplasma-formulary-037290d4.md:2701-2704]` —
  `eta_perp = 1.03e-2 * Z * ln(Lambda) * T^(-3/2)  [Ohm cm]`.
- Weakly-ionized electron-neutral collision frequency:
  `[KR: KnowledgeReference/2019nrlplasma-formulary-037290d4.md:3384-3394]` —
  `nu_alpha = n_0 * sigma * (k T_alpha / m_alpha)^(1/2)`, with
  `sigma ~ 5e-15 cm^2`.
- Weakly-ionized conductivity `sigma_alpha = n_alpha e_alpha mu_alpha`:
  `[KR: KnowledgeReference/2019nrlplasma-formulary-037290d4.md:3410-3411]`.

Symbol map: `lambda = ln Lambda` Coulomb logarithm [-]; `n_e` electron density
(cm^-3 in NRL eq.); `Z` charge state [-]; `T_e` electron temperature (eV in
NRL eq.); `eta_perp` transverse resistivity [Ohm cm in NRL]; `n_0` neutral
density [cm^-3]; `sigma` electron-neutral cross-section [cm^2]; `mu` mobility
[m^2/(V s)].

Units: NRL rows are CGS-eV; in-code `src/dpf/collision/spitzer.py` and
`src/dpf/fields/conductivity.py` work in SI. The in-code
`spitzer_resistivity` (`spitzer.py:192-231`) applies a Braginskii `alpha(Z)`
correction; for `Z=1`, `alpha=0.5064`, giving `eta` about 0.5x the classical
value, which the docstring states matches NRL `eta ~ 5.2e-5 Z lnL Te_eV^(-3/2)`.

Validity range — **the central PlasmaPy-relevant finding.** NRL explicitly
bounds classical collisional transport
`[KR: KnowledgeReference/2019nrlplasma-formulary-037290d4.md:3379-3383]`: transport coefficients
are valid only when "(3) the Coulomb logarithm satisfies `lambda >> 1`", and
"(5) relative drifts `u = v_alpha - v_beta` ... are small compared with the
thermal velocities, i.e. `u^2 << kT_alpha/m_alpha`". NRL further states the
Coulomb-log theory "is good only to ~10% and fails when `lambda ~ 1`"
`[KR: KnowledgeReference/2019nrlplasma-formulary-037290d4.md:3036-3038]`. The in-code
`coulomb_log` (`spitzer.py:41-65`) **floors** `ln Lambda` at 2.0
("Spitzer theory invalid below this") — this is a code-side acknowledgment of
exactly the NRL `lambda ~ 1` breakdown, but it is a silent floor, not a gate
that flags the regime.

Current code status: `electrical_thermal_transport` is `candidate` in
`closure_packet.py:121-137`. The collision/transport functions in `spitzer.py`
and `conductivity.py` carry source headers routing to the NRL KR
(`spitzer.py:33-36`, `conductivity.py:14-19`) and self-declare
`source_grounded_engineering_transport_not_validation` with
`can_support_first_principles_acceptance = False`. **Status: candidate**
(NRL-source-grounded for the collision-frequency and resistivity forms; missing
an accepted validity-regime gate and sensitivity/UQ packet).

#### 1.7b Stopping (fast-ion stopping power)

Source equation: **none located as an implementable closure.** No fast-ion
stopping-power closure operator exists in `src/dpf/`. Stopping is named as a
required-but-missing channel inside the beam-target effect
(`closure_packet.py:177` `ion_distribution_transport_stopping`). No
Bethe/Bethe-Bloch or plasma stopping-power equation in `KnowledgeReference/`
was located as a citable closure for implementation.

Current code status: **blocked** and `not_simulated`. Stopping must be a
registry row in its own right because beam-target neutron authority
(section 1.8) cannot be evaluated without it.

### 1.8 Beam-Target Coupling

Source equation (Lee & Saw phenomenological form, KR-cited):
`src/dpf/diagnostics/beam_target.py:1-16` —

```
Y_bt = Cn * n_i * I_pinch^2 * z_p^2 * ln(b/r_p) * sigma(E_beam) / V_max^(1/2)
```

KR support:

- Lee & Saw beam-target form and `Cn` calibration:
  `[KR: KnowledgeReference/a-course-on-plasma-focus-numerical-experiments-s-lee-and-s-h-saw-part-1-basic-course.md:5109-5145]`
  (cited verbatim in `beam_target.py:6-16`; `E_beam = 3*V_max` from KR
  L5133-5139; `Cn` calibration from KR L5141-5144).
- DD fusion cross-section (Bosch-Hale 1992 parametric fit):
  `[KR: KnowledgeReference/bosch-hale-1992-fusion-reactivity.md:40-57]` — `sigma(E) = S(E)/(E exp(B_G/sqrt(E)))`,
  Table IV coefficients in `beam_target.py:76-90`.

Symbol map: `Y_bt` beam-target neutron yield [neutrons]; `Cn` calibrated
proportionality constant [SI, value `1.810e7`]; `n_i` ion density [m^-3];
`I_pinch` pinch current [A]; `z_p` pinch length [m]; `b`,`r_p` outer/pinch
radius [m]; `sigma` DD cross-section [m^2]; `E_beam` beam ion energy [eV];
`V_max` peak voltage [V]; `S(E)` astrophysical S-factor [keV millibarn];
`B_G` Gamow constant [keV^(1/2)].

Units: SI in code; Bosch-Hale fit internally in keV/millibarn then converted
(`_MBARN_TO_M2 = 1e-31`, `beam_target.py:89`).

Validity range: Bosch-Hale fit valid `0.5 keV < E_cm < 5000 keV`
(`beam_target.py:108`); returns 0 outside. The Lee & Saw form is an empirical
*single-constant* fit and is explicitly baseline/comparator context only
(`beam_target.py:33-37`): "Accepted neutron authority remains in
`dpf.first_principles.neutron_authority` and requires mechanism-separated,
same-scope evidence."

Current code status: `beam_target_coupling` is `blocked` in
`closure_packet.py:172-182` (missing `mechanism_separation`,
`ion_distribution_transport_stopping`, `spectrum_anisotropy_detector_response`,
`uq`); goes `candidate` only when `kinetic_yield_present=True`, and even then
`can_support_first_principles_acceptance` stays `False`
(`test_first_principles_closures.py:83-87`). The DD cross-section sub-component
is Bosch-Hale KR-cited. **Status: blocked** for the *coupling closure* (the
phenomenological Lee/Saw form is `external_candidate_not_authority`; the
mechanism-separated thermonuclear-vs-beam-target split has no closure).

---

## 2. Closure Status Table (`supported` / `candidate` / `blocked`)

`supported` here means "the closure operator is implemented AND every governing
equation, coefficient, unit, validity range, and verification test is closed by
a local `KnowledgeReference/` source". By that definition **no closure is
`supported`** — this is consistent with `closure_packet.py:266`
`can_support_first_principles_acceptance = False`.

| # | Closure                | Status     | KR source (closes the *form*)                                  | Code location                                  | Why not higher                                                                 |
| - | ---------------------- | ---------- | -------------------------------------------------------------- | ----------------------------------------------- | ------------------------------------------------------------------------------- |
| 1 | EOS / thermodynamics   | blocked    | ALEGRA HEDP (EOS *class* only): `[KR: KnowledgeReference/unlimited-release-printed-september-2009-alegra-hedp-simulations-of-the-dense-plasma-focus.md:188,351-360]` | `src/dpf/fluid/eos.py` (ideal-gas only)         | No tabular/QEOS closure equation; ideal gas out of validity for cold/dense regimes |
| 2 | Radiation              | candidate  | NRL eq.(30),(34): `[KR: KnowledgeReference/2019nrlplasma-formulary-037290d4.md:4732-4758]`  | `src/dpf/radiation/bremsstrahlung.py`, `improved_radiation.py` | Volumetric loss term only; opacity/FLD/line closure unsourced                    |
| 3 | Ablation / impurity    | blocked    | none (efficiency coefficient unsourced)                        | `src/dpf/atomic/ablation.py`                    | `efficiency` is unverified empirical constant; references not in KR              |
| 4 | Anomalous resistance   | blocked    | NRL anomalous ion-sound rate (form only): `[KR: KnowledgeReference/2019nrlplasma-formulary-037290d4.md:2704-2710]` | `src/dpf/turbulence/anomalous.py`               | `alpha` band and threshold-model selection unsourced                            |
| 5 | Restrike               | blocked    | none                                                           | none (config/preset wording only)              | Not simulated; no closure equation in code or KR                                |
| 6 | Electron inertia       | blocked    | none located                                                   | none (diagnostics reference only)              | Not registered; no closure operator; no KR closure equation                     |
| 7a| Collision (transport)  | candidate  | NRL Coulomb log + Spitzer: `[KR: KnowledgeReference/2019nrlplasma-formulary-037290d4.md:3024-3065]` + `[KR: KnowledgeReference/2019nrlplasma-formulary-037290d4.md:2701-2704]`  | `src/dpf/collision/spitzer.py`, `fields/conductivity.py` | Missing accepted validity-regime gate + sensitivity/UQ                           |
| 7b| Stopping (fast-ion)    | blocked    | none located                                                   | none                                            | No stopping-power closure operator; no KR closure equation                       |
| 8 | Beam-target coupling   | blocked    | Lee/Saw + Bosch-Hale: `[KR: KnowledgeReference/a-course-on-plasma-focus-numerical-experiments-s-lee-and-s-h-saw-part-1-basic-course.md:734,1297-1298]` + `[KR: KnowledgeReference/bosch-hale-1992-fusion-reactivity.md:40-57]`    | `src/dpf/diagnostics/beam_target.py`            | Lee/Saw form is empirical comparator only; no mechanism-separated closure        |

Status counts: `supported` 0, `candidate` 2 (radiation volumetric loss term;
collision/transport), `blocked` 7 (EOS, ablation, anomalous resistance,
restrike, electron inertia, stopping, beam-target). Row 4/5 share one registry
effect but are audited and counted separately because they are physically
distinct closures.

---

## 3. Runtime Fields Required (inputs / outputs per closure)

The runtime closure-packet registry must hand each closure operator the inputs
below and capture the outputs. All fields are fail-closed: a missing input
must block the closure, never substitute a default.

| Closure                | Required runtime inputs                                                                 | Required runtime outputs                                              |
| ---------------------- | ---------------------------------------------------------------------------------------- | ---------------------------------------------------------------------- |
| EOS / thermodynamics   | `rho` [kg/m^3], `T_i` [K], `T_e` [K], `Z` field [-], `gamma` policy, fill-species id     | `p_i`,`p_e` [Pa], `e_i`,`e_e` [J/kg], `c_s` [m/s], EOS-table-id hash    |
| Radiation              | `n_e` [m^-3], `T_e` [K], `Z` [-], `B` [T], path length `L` [m] (for `tau`), opacity-id   | `P_brem`,`P_cyc` [W/m^3], `tau` [-], radiated-energy ledger delta [J]   |
| Ablation / impurity    | boundary `eta` [Ohm m], boundary `J` [A/m^2], `T_surface` [K], `efficiency` [kg/J], mask | `S_rho` [kg/(m^3 s)], `S_n` [m^-3 s^-1], `S_mom` [N/m^3], impurity-Z    |
| Anomalous resistance   | `J` [A/m^2], `n_e` [m^-3], `T_e`,`T_i` [K], `B` [T], `v_bulk` [m/s], `alpha`, model-id   | `eta_anom` [Ohm m], active-threshold-model id, active-cell fraction     |
| Restrike               | post-pinch `I(t)` [A], `dI/dt` [A/s], column-state flag                                  | restrike onset time [s], recovered `eta` [Ohm m]                       |
| Electron inertia       | `J` [A/m^2], `dJ/dt` [A/s], `n_e` [m^-3], grid spacing `dx` [m]                          | electron-inertia EMF term [V/m], resolved-skin-depth flag              |
| Collision (transport)  | `n_e` [m^-3], `T_e`,`T_i` [K], `Z` [-], `B` [T], `n_0` neutral [m^-3], `v_d` drift [m/s] | `ln Lambda` [-], `nu_ei`,`nu_ee`,`nu_ii`,`nu_en` [s^-1], `eta`,`sigma`, `kappa_par/perp` |
| Stopping (fast-ion)    | fast-ion energy distribution `f(E)` [-], background `n_e`,`n_i` [m^-3], `T_e` [K]        | `dE/dx` stopping power [eV/m], slowing-down time [s], deposited E [J]   |
| Beam-target coupling   | `n_i` [m^-3], `I_pinch` [A], `z_p`,`b`,`r_p` [m], `V_max` [V], ion distribution          | `Y_bt` [neutrons], `sigma(E)` [m^2], mechanism-tagged yield split       |

Every closure operator must additionally emit: `closure_id`, `classification`
(see section 6), `source_refs` list of KR citations, `validity_regime_flags`,
and `can_support_first_principles_acceptance = False`.

---

## 4. Missing Parameters (no KR source)

Closures and coefficients with **no local `KnowledgeReference/` source**. Each
must be marked `blocked` until a KR extract packet is added.

| Missing item                              | Closure              | Current value / status in code                          | What a KR source must supply                                       |
| ------------------------------------------ | -------------------- | -------------------------------------------------------- | ------------------------------------------------------------------ |
| Tabular / QEOS EOS equation set            | EOS                  | absent (ideal gas only)                                  | `p(rho,T)`, `e(rho,T)`, ionization, low-density validity floor      |
| EOS `rho` low-density validity edge        | EOS                  | bare `1e-30` floor (`eos.py:48`)                         | physically motivated minimum density and EOS-table coverage         |
| Ablation efficiency `efficiency` [kg/J]    | Ablation             | `Cu 5.0e-5`, `W 2.0e-5` (`ablation.py:49,54`)            | KR-cited erosion yield with fluence/shielding dependence            |
| Anomalous turbulence parameter `alpha`     | Anomalous resistance | `0.05` default, band "0.01-0.1" (`anomalous.py:336`)     | KR-cited saturation-amplitude / `alpha` for DPF regime              |
| CIV critical velocity `v_crit` table       | Anomalous resistance | hardcoded gas table (`anomalous.py:380-389`)             | KR-cited `v_crit = sqrt(2 E_ion/m)` per fill gas                    |
| Restrike / current-dip recovery model      | Restrike             | absent                                                   | governing equation for post-pinch resistance recovery               |
| Generalized-Ohm electron-inertia term      | Electron inertia     | absent (diagnostics-only mention)                        | `m_e dJ/dt` Ohm-law closure equation + skin-depth resolution gate    |
| Rosseland / Kramers opacity closure        | Radiation transport  | scaffold self-flagged missing (`transport.py:34`)        | KR-cited opacity / radiation-diffusion closure                       |
| Line-radiation provenance                  | Radiation            | `unknown_provenance_empirical_fits` (`line_radiation.py:59`) | KR-cited line-emission model and coefficients                    |
| Free-free QMF suppression source           | Radiation            | `free_free_suppression_source_missing` (`qmf_suppression.py:53`) | KR-cited free-free suppression reference                         |
| Gaunt-factor temperature fit               | Radiation            | Karzas & Latter (1961) fit (`improved_radiation.py`)     | KR extract of Gaunt-factor tables (Karzas-Latter not in corpus)      |
| Fast-ion stopping power `dE/dx`            | Stopping             | absent                                                   | KR-cited Bethe / plasma stopping-power closure                      |
| Braginskii `alpha(Z)` / `delta_e(Z)` table | Collision/transport  | hardcoded `0.5064...` / `3.16...` (`spitzer.py:154,242`) | KR extract of Braginskii (1965) Table 1 — not in local corpus        |
| Mechanism-separated yield closure          | Beam-target          | only phenomenological Lee/Saw                            | KR-cited thermonuclear-vs-beam-target separation model               |

Note on Braginskii: `spitzer.py` cites "Braginskii, Reviews of Plasma Physics
Vol. 1 (1965), Table 1" for the `alpha(Z)` and `delta_e(Z)` correction
coefficients. Braginskii (1965) is **not in `KnowledgeReference/`**. The NRL
formulary KR file confirms the Coulomb-log and Spitzer-resistivity *forms* but
the local NRL excerpt does not tabulate the Braginskii `Z`-dependent
correction coefficients. The collision closure is therefore `candidate` for the
NRL-grounded core but its Braginskii correction factors are a `missing
parameter` until a Braginskii KR extract is added.

---

## 5. Proposed Tests and Fail-Closed Negative Controls

These tests extend `tests/test_first_principles_closures.py` and the
formulary-audit tests (`test_formulary_transport_audit.py`,
`test_formulary_radiation_audit.py`). No test asserts a closure value from
training data; expected values come from the cited KR equation or are checked
at runtime against the in-code formula.

### 5.1 Positive / structural tests

1. `test_closure_registry_has_eight_closures` — registry exposes all eight
   closures including the new `electron_inertia` and `stopping` keys.
2. `test_radiation_brem_matches_nrl_eq30` — `bremsstrahlung_power` at a fixed
   `(n_e, T_e, Z)` reproduces the NRL eq.(30) CGS value within the documented
   SI-conversion tolerance.
3. `test_collision_coulomb_log_branch_selection` — `coulomb_log` follows the
   NRL eq. branch `[KR L3045-3059]` for the `T_e < 10 Z^2 eV` and
   `T_e > 10 Z^2 eV` regimes.
4. `test_beam_target_bosch_hale_in_valid_range` — `dd_cross_section` returns 0
   outside `0.5-5000 keV` and a finite positive value inside.
5. `test_eos_ideal_gas_dimensional_consistency` — `IdealEOS` pressures/energies
   are dimensionally consistent (already partly covered).

### 5.2 Fail-closed negative controls

1. `test_eos_blocked_without_tabular_packet` — `eos_thermodynamics` stays
   `blocked` and `can_support_first_principles_acceptance=False` when no
   tabular/QEOS packet is attached.
2. `test_radiation_blocked_without_opacity_decision` — `radiation_losses` stays
   `blocked` when the opacity/diffusion decision is missing, even if the
   bremsstrahlung volumetric term is present.
3. `test_ablation_blocked_without_efficiency_source` — `impurity_electrode_ablation`
   stays `blocked` while `ablation_efficiency_source_packet_missing` holds.
4. `test_anomalous_and_restrike_claim_rejected` — any current-dip / post-pinch /
   anomalous-resistance claim is rejected
   (`closure_packet.py:256`).
5. `test_electron_inertia_registered_and_blocked` — the new `electron_inertia`
   key exists and is `blocked` with no KR source.
6. `test_stopping_blocked_blocks_beam_target` — `stopping` `blocked` forces
   `beam_target_coupling` to remain `blocked` (dependency rejection).
7. `test_closure_value_substituted_from_residual_rejected` — no closure may be
   back-derived from an energy residual (mirrors the Auluck no-closure rule).
8. `test_closure_sensitivity_uq_missing_rejected` — a closure without a
   sensitivity/UQ packet cannot be promoted
   (`closure_packet.py:257`).

### 5.3 PlasmaPy strong-coupling regime gate (negative control)

Motivation: the broad test suite emits PlasmaPy `CouplingWarning`
strong-coupling warnings. PlasmaPy raises `CouplingWarning` when the plasma
coupling parameter indicates a strongly coupled / non-ideal regime where the
weak-coupling Coulomb-logarithm expansion is invalid. The local NRL KR
**confirms** this validity edge: classical transport needs `lambda >> 1`
`[KR: KnowledgeReference/2019nrlplasma-formulary-037290d4.md:3379-3380]` and the Coulomb-log
theory "fails when `lambda ~ 1`" `[KR: KnowledgeReference/2019nrlplasma-formulary-037290d4.md:3036-3038]`. The DPF
dense pinch core can reach the strongly coupled regime, so the warning is
*physically meaningful*, not noise.

Proposed regime gate (`test_plasmapy_coupling_regime_gate`):

1. Compute the coupling indicator from the reference state (the in-code
   `coulomb_log` already floors `ln Lambda` at 2.0, `spitzer.py:65` — treat
   `ln Lambda <= ~2` or a PlasmaPy `CouplingWarning` as the "out of weak-
   coupling validity" trigger).
2. **In validity** (`ln Lambda` comfortably `> ~2-3`, no `CouplingWarning`):
   the PlasmaPy formulary cross-check is allowed to run as an *engineering
   telemetry cross-check only* (status
   `community_formula_cross_check_within_tolerance_not_authority`,
   `plasmapy_audit.py:240-247`). It still cannot promote any claim
   (`plasmapy_audit.py:130` `can_support_first_principles_acceptance=False`).
3. **Out of validity** (`CouplingWarning` raised, or `ln Lambda <= ~2`): the
   gate must mark the closure cell `bounded_out_with_source` and record the NRL
   `lambda >> 1` validity citation. The classical Spitzer/collision closure is
   *out of its validity range* in that cell; the runtime must flag it, not
   silently floor it. A strong-coupling cell that produces a transport value
   without this flag is a test failure.
4. The test must assert that a `CouplingWarning` is **not** swallowed silently:
   `closure_packet` (or the transport closure telemetry) must surface a
   `strong_coupling_out_of_validity` field when the warning fires.
5. The test must assert PlasmaPy disagreement does **not** block an engineering
   run (`closure_packet.py:228`
   `missing_or_failed_audit_blocks_engineering_run: False`) but **does** require
   review when outside tolerance (`closure_packet.py:229`
   `outside_tolerance_audit_requires_review: True`).

This gate converts the `CouplingWarning` from an unhandled warning into a
source-backed validity flag and a fail-closed negative control.

---

## 6. Exact Implementation Recommendations

### 6.1 Closure-packet registry structure

Extend `src/dpf/first_principles/closure_packet.py` (a Sprint 3+ change, **not**
a Sprint 2.2 file — `closure_packet.py` is not in the Sprint 2.2 ownership list)
so the registry is an explicit per-closure record. Proposed `dict` schema for
each registry row (one row per closure):

```
closure_registry_row = {
    "closure_id": str,                       # e.g. "eos_thermodynamics"
    "classification": str,                   # one of the five values in 6.2
    "source_equations_or_bound": [           # list of KR citations
        {"path": "KnowledgeReference/...", "lines": "Lstart-Lend",
         "equation": "eq.(N)", "role": "..."},
    ],
    "symbol_map": {symbol: {"meaning": str, "unit": str}},
    "units": "SI" | "CGS-eV-converted",
    "validity_regime": {                     # explicit, fail-closed
        "valid_when": [str, ...],
        "out_of_validity_when": [str, ...],
        "regime_flag_field": str,            # runtime field name
    },
    "runtime_inputs": [field_name, ...],     # section 3
    "runtime_outputs": [field_name, ...],    # section 3
    "implementation_reference": str | None,  # code path or None
    "verification_tests": [test_name, ...],  # section 5
    "missing_parameters": [str, ...],        # section 4
    "sensitivity_or_uq": str,                # "missing" until a packet exists
    "claim_impact": str,                     # what claim is blocked
    "review_status": "not_reviewed_for_acceptance",
    "can_support_first_principles_acceptance": False,   # always
}
```

Required registry-level additions:

- Add `electron_inertia` and `stopping` to `REQUIRED_EFFECTS`
  (`closure_packet.py:37-48`) — the registry currently has neither (F-EI).
- Keep `restrike` and `anomalous_resistance` distinguishable: either split
  `restrike_anomalous_resistance` into two effect keys, or add a
  `sub_closures` list so the two physically distinct closures are individually
  classified and individually testable.
- The registry-level `can_support_first_principles_acceptance` stays `False`
  unchanged (`closure_packet.py:266`).

### 6.2 Per-closure classification

Each registry row gets exactly one `classification` from this five-value
vocabulary (as requested):

- `active_source_backed_candidate` — operator implemented; governing form is
  closed by a local KR source; runs engineering cases; cannot support
  acceptance. Apply to: **Collision/transport** (`spitzer.py`, `conductivity.py`
  — NRL Coulomb-log + Spitzer KR-grounded); **Radiation volumetric loss term**
  (`bremsstrahlung.py` — NRL eq.(30)/(34) KR-grounded).
- `active_blocked` — operator implemented but the closure as a whole is blocked
  on a missing source/coefficient/test. Apply to: **Anomalous resistance**
  (operator exists, `alpha` and thresholds unsourced); **Ablation/impurity**
  (operator exists, `efficiency` unsourced); **Radiation transport/opacity/line**
  (FLD scaffold exists, opacity closure unsourced).
- `bounded_out_with_source` — closure is intentionally excluded from the run
  with a cited validity bound proving it is negligible or out of range. Apply
  per-cell to: **Collision/transport in a strong-coupling cell** — bounded out
  with NRL `lambda >> 1` citation `[KR: KnowledgeReference/2019nrlplasma-formulary-037290d4.md:3379-3380]` and `[KR: KnowledgeReference/2019nrlplasma-formulary-037290d4.md:3036-3038]`
  (this is the PlasmaPy regime-gate outcome). Also the candidate target for
  **Electron inertia** once a KR skin-depth bound shows `d_e` is sub-grid.
- `not_simulated_and_claim_blocking` — closure is not implemented at all and
  its absence blocks a class of claims. Apply to: **Restrike** (blocks
  current-dip/post-pinch claims); **Stopping** (blocks beam-target neutron
  authority); **Electron inertia** (until bounded out or implemented);
  **EOS tabular/QEOS** (blocks whole-shot thermodynamics/pressure authority).
- `external_candidate_not_authority` — an external/community formula or a
  phenomenological fit is used only as a cross-check or comparator baseline,
  never as authority. Apply to: **PlasmaPy formulary cross-checks**
  (`plasmapy_audit.py` — already correctly scoped); **Lee & Saw beam-target
  phenomenological form** (`beam_target.py` — explicitly "baseline/comparator
  context only", `beam_target.py:33-37`).

A closure may carry a primary classification plus per-cell `bounded_out_with_source`
(e.g. collision/transport is `active_source_backed_candidate` globally but
`bounded_out_with_source` in strong-coupling cells).

### 6.3 Wiring rules

- Each closure operator imports its source citations as module constants
  (the pattern already used: `SPITZER_SOURCE_REFERENCES`,
  `BREMSSTRAHLUNG_SOURCE_REFERENCES`, `BEAM_TARGET_SOURCE_REFERENCES`,
  `CLOSURE_SOURCE_REFS`).
- Empirical/unsourced modules (`line_radiation.py`, `ablation.py`,
  `qmf_suppression.py`, `transport.py` FLD) stay **out** of the
  first-principles import graph — this is already a regression guard in
  `test_first_principles_closures.py:4-8`. Do not regress it.
- Route every numerical floor through `telemetry.apply_floor()` per the project
  `CLAUDE.md` numerical-coding rule when implementing closures (fix the bare
  `np.maximum(rho, 1e-30)` in `eos.py:48` at implementation time).
- No closure may be marked `implemented` in any packet unless its code and
  tests are in the *same* submitted diff (parallel-deliverable rule 4).

---

## 7. Explicit "Do Not Promote" Notes

- This packet is a **source audit only**. It does not implement, validate, or
  accept any closure. `can_support_first_principles_acceptance = False` for all
  eight closures, matching `closure_packet.py:266`.
- `candidate` in section 2 means *runs engineering cases*, never *validated*.
  Per `closure_packet.py:233` candidate closures `can_support_acceptance: False`.
- The two `candidate` closures (radiation volumetric loss; collision/transport)
  are NRL-formulary-grounded for their *functional form only*. They still lack
  accepted validity-regime gates, sensitivity/UQ packets, and same-scope review.
  Do not present them as validated DPF physics.
- PlasmaPy is **not** a source authority. Any PlasmaPy formulary result is an
  `external_candidate_not_authority` cross-check; it cannot promote a claim
  (`plasmapy_audit.py:256-264`). The strong-coupling `CouplingWarning` regime
  gate is a fail-closed *negative control*, not a validation pass.
- The Lee & Saw beam-target form is `external_candidate_not_authority` /
  comparator baseline only. Accepted neutron authority remains in
  `dpf.first_principles.neutron_authority` and requires mechanism-separated,
  same-scope evidence (`beam_target.py:33-37`).
- Docstring citations in `ablation.py`, `anomalous.py`, and `improved_radiation.py`
  (Bruzzone, Vikhrev, Lee & Serban, Buneman, Sagdeev, Davidson & Gladd, Haines,
  Karzas & Latter, Brenning, Alfven) are **not** in `KnowledgeReference/`. They
  do not satisfy the local-source-truth rule and must not be cited as authority
  until a KR extract packet exists.
- No closure may be implemented from an inferred formula. A closure is promoted
  only when its exact source packet, equation, units, symbol map, validity
  range, tests, and review are all attached (parallel-deliverable rule 1;
  `closure_packet.py:234-239`).
- This packet does not edit any Sprint 2.2-owned file and adds no runtime code.

---

## 8. KnowledgeReference Citations Used (verification index)

| Citation                                                              | Used for                                                  |
| ---------------------------------------------------------------------- | --------------------------------------------------------- |
| `2019nrlplasma-formulary-037290d4.md` L3024-3083                       | Coulomb logarithm definition + e-i branch                 |
| `2019nrlplasma-formulary-037290d4.md` L3036-3038                       | "theory fails when lambda ~ 1" (PlasmaPy regime gate)     |
| `2019nrlplasma-formulary-037290d4.md` L3379-3383                       | classical transport validity criteria (3),(5),(6)        |
| `2019nrlplasma-formulary-037290d4.md` L2701-2704                       | transverse Spitzer resistivity                            |
| `2019nrlplasma-formulary-037290d4.md` L2706-2710                       | anomalous ion-sound collision rate                        |
| `2019nrlplasma-formulary-037290d4.md` L3384-3411                       | weakly-ionized e-n collisions + conductivity              |
| `2019nrlplasma-formulary-037290d4.md` L4732-4758 eq.(30)-(34)          | bremsstrahlung, optical depth, recombination, cyclotron   |
| `unlimited-release-printed-september-2009-alegra-hedp-...dpf.md` L188,L215,L333-369 | DPF EOS class: Sesame + QEOS; radiation-diffusion scope   |
| `a-course-on-plasma-focus-numerical-experiments-s-lee-and-s-h-saw-part-1-basic-course.md` L5109-5145 | Lee & Saw beam-target form + Cn calibration               |
| `bosch-hale-1992-fusion-reactivity.md`                                 | DD fusion cross-section parametric fit                    |
| `fully-electromagnetic-hybrid-pic-fluid-dpf-neutron-yield-acb71fa9.md` L431-619 | hybrid closure equations (existing `CLOSURE_SOURCE_REFS`) |

End of WP-N5 Closure Registry Source Audit.
