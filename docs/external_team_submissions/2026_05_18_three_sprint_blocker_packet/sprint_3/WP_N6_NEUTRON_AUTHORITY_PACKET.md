# WP-N6 Neutron Authority Packet

Lane: 5 (parallel lane, per
`docs/FIRST_PRINCIPLES_CODEX_AUDIT_SPRINT2_FOLLOWUP_2026_05_19.md` lines 284-290).
Date: 2026-05-19
Branch: `codex/corpus`
Status: research/interface-design packet only. No runtime feature is marked
`implemented`. No code or test is part of this submission.

## 0. Scope, authority limits, and standing rules

This packet designs the neutron-authority runtime interface and evidence map
for a same-scope target. The canonical declared scope used throughout is
`pf1000_akel_16kv_1p2torr_shot_12581` (the PF-1000/Akel shot already named in
`src/dpf/first_principles/neutron_authority.py:24` and the existing negative-
control test).

Standing rules this packet obeys:

1. Every physics claim cites a local `KnowledgeReference/` file with a line
   range and the equation/figure/table.
2. The packet does not promote validation, acceptance, or first-principles
   authority. `can_support_first_principles_acceptance` stays `false`.
3. Scalar total neutron yield is **not** mechanism authority. This is restated
   in Section 7.
4. Mechanisms with no KR source are marked `blocked` and listed in Section 4.
5. This packet does not edit any Sprint 2.2-owned file and does not run the
   periodic audit.

Pre-existing runtime code in scope (read-only review, not modified):

- `src/dpf/first_principles/neutron_authority.py` — fail-closed mechanism-
  separated packet builder. Already correct in shape; this packet specifies the
  runtime interface that feeds it.
- `src/dpf/diagnostics/neutron_yield.py` — thermonuclear Bosch-Hale scalar path.
- `src/dpf/diagnostics/beam_target.py` — Lee/Saw beam-target form + Bosch-Hale
  cross section + anisotropy helper.
- `src/dpf/diagnostics/neutron_tof.py` — synthetic nToF spectrum (no detector
  response, no scatter).
- `src/dpf/diagnostics/pic_yield.py`, `src/dpf/fields/kinetic_yield.py` —
  candidate PIC ion yield-history diagnostic.
- `tests/test_first_principles_neutron_authority.py` — current SSR-009/WP-6
  negative controls.

Code-wiring fact established by grep this session: the thermonuclear path
`neutron_yield.py` is imported only by `diagnostics/yield_tracker.py` and
`diagnostics/evidence_manifest.py`; it is **not** wired into
`src/dpf/first_principles/` runtime. So no first-principles runtime currently
emits a mechanism-separated neutron channel — consistent with the blocked
status throughout.

---

## 1. Source-backed findings per mechanism

Each finding gives: the source equation / cross-section with KR citation, units,
symbol map, validity range, and current code status. ASCII equation forms are
reproduced from the KR text; the KR files remain the authority.

### 1.1 DD fusion reaction channels and Q-values

- Source: `KnowledgeReference/2019nrlplasma-formulary-037290d4.md:3802-3814`.
  Verbatim: reaction (1a) `D + D -> 50% T(1.01 MeV) + p(3.02 MeV)`;
  reaction (1b) `D + D -> 50% He3(0.82 MeV) + n(2.45 MeV)`;
  reaction (2) `D + T -> He4(3.5 MeV) + n(14.1 MeV)`.
- Units: reaction-product kinetic energies in MeV. Branching ratios 50%/50%
  "correct for energies near the cross section peaks".
- Symbol map: only the (1b) branch produces a neutron; its mono-energetic
  birth energy is 2.45 MeV in the centre-of-mass frame.
- Validity range: branching ratios stated as valid near the cross-section peak.
- Current code status: `supported` as a constant. `neutron_tof.py:24`
  hardcodes `_E_N_CENTER = 2.45e6` eV; `neutron_yield.py:36-37` documents both
  branches. Consistent with KR.

### 1.2 Thermonuclear reactivity (Bosch-Hale Maxwellian fit)

- Source: `KnowledgeReference/bosch-hale-1992-fusion-reactivity.md:59-93`
  (Eqs. 12-14, p. 624; Table VII, p. 625).
  ASCII: `<sigma v> = C1 * theta * sqrt(xi / (mr*c^2 * T^3)) * exp(-3*xi)`;
  `theta = T / (1 - T*(C2 + T*(C4 + T*C6)) / (1 + T*(C3 + T*(C5 + T*C7))))`;
  `xi = (BG^2 / (4*theta))^(1/3)`.
- Units: `T` = ion temperature in keV; `<sigma v>` in cm^3/s; `BG` in
  sqrt(keV); `mr*c^2` in keV.
- Symbol map (D(d,n)3He branch, Table VII):
  `BG = 31.3970`, `mr*c^2 = 937814`, `C1 = 5.43360e-12`, `C2 = 5.85778e-3`,
  `C3 = 7.68222e-3`, `C4 = 0`, `C5 = -2.96400e-6`, `C6 = 0`, `C7 = 0`.
  D(d,p)T branch (`C1 = 5.65718e-12`, `C2 = 3.41267e-3`, `C3 = 1.99167e-3`,
  `C5 = 1.05060e-5`).
- Validity range: `0.2 keV <= Ti <= 100 keV`
  (`bosch-hale-1992-fusion-reactivity.md:92,106`). Fit deviation <= 0.3%
  (D(d,n)3He); absolute reactivity uncertainty 6% for D-D from R-matrix cross
  sections (`...:108-109`).
- Thermonuclear yield-rate form: `dpf/diagnostics/neutron_yield.py:6-9` uses
  `dY/dt = (1/4) * n_D^2 * <sigma v>(Ti) * V_pinch`. The 1/4 factor
  (identical-particle 1/2, neutron-branch 1/2) is `[INFERRED]` — it is standard
  but the 1/4 itself is not a verbatim KR formula; the KR source supplies only
  `<sigma v>`. Flag in Section 4.
- Current code status: reactivity fit `supported` (verbatim coefficients,
  validity clamp at 0.2/100 keV present, `neutron_yield.py:31-116`). The
  volumetric `(1/4) n_D^2 <sigma v> V` integrator is `candidate` — code exists
  but the 1/4 prefactor lacks a KR citation and the path is not wired into the
  first-principles runtime.

### 1.3 Beam-target production (Lee/Saw reduced model)

- Source: `KnowledgeReference/radiation-physics-and-chemistry-188-2021-109633.md`
  (Akel 2021), lines 195-215. Verbatim eq. (1):
  `Yn = Yb-t = Cn ni Ipinch^2 zp^2 (ln(b/rp)) sigma / U^(1/2)`.
- Units (SI, per the KR text "all quantities in SI units"): `ni` m^-3,
  `Ipinch` A, `zp`/`b`/`rp` m, `sigma` m^2, `U` V; `Yn` neutrons per shot.
- Symbol map (KR lines 199-211): `ni` = maximum pinch ion density; `Ipinch` =
  initial current through the pinch; `rp`, `zp` = final pinch radius and length;
  `b` = cathode radius; `sigma` = D-D cross section for beam energy = 3*Vmax;
  `U` = disruption-caused diode voltage; data fitting gives `U = 3 Vmax`;
  `Cn = 8.54e8` (SI), calibrated at 0.5 MA from a graphical fit across devices.
- Validity range: this is an explicitly **fitted reduced model** — KR lines
  205-215 and 858-861 state `fc` is held constant at 0.7 and the four model
  parameters are tuned per shot to fit the measured current waveform.
  Therefore the Lee/Saw form is comparator/baseline context only; it is not a
  first-principles mechanism authority.
- Mechanism premise (physical, KR-cited): "a beam of fast deuteron ions is
  produced by a vacuum diode in a thin layer close to the anode... traverses
  the pinch column of deuterium plasma to produce the fusion neutrons"
  (`...109633.md:200-204`).
- Current code status: `beam_target.py:184-260` implements eq. (1) verbatim
  with `Cn = 1.810e7` recalibrated to a different KR datum (Lee/Saw course
  L5141-5144, `Yn = 7e9` at 0.5 MA). Status `candidate` for runtime diagnostic;
  explicitly `not validation` (`BEAM_TARGET_SOURCE_STATUS` in
  `beam_target.py:55`). Note the two `Cn` values (8.54e8 from Akel 2021 vs
  1.810e7 from the Lee/Saw course) come from different KR sources and different
  calibration points — see Section 4.

### 1.4 Beam-target requires kinetic ions (mechanism-separation requirement)

- Source: `KnowledgeReference/fully-kinetic-simulations-of-dense-plasma-focus-z-pinch.md:38-43`.
  Verbatim: "Fluid simulations predict no neutrons and do not allow for
  non-thermal ions, while hybrid simulations under-predict neutron yield by
  ~100x and exhibit an ion tail that does not exceed 200 keV. Only fully
  kinetic simulations predict MeV-energy ions and experimental neutron yields."
- Source: `...z-pinch.md:152-161`: "Both thermonuclear and beam-target fusion
  can occur inside DPF plasmas... The total neutron production predicted in the
  fully kinetic calculation was 0.86e7... The hybrid simulation predicted
  3.6e4... the fluid simulation predicted no neutrons."
- Source: `KnowledgeReference/sand2009-6373-b93aec67.md:345-355,394-397`:
  "Plasma densities and temperatures... are insufficient to produce
  thermonuclear neutrons in the quantities observed... MHD codes such as
  ALEGRA can model only the thermonuclear contribution... we therefore expect
  that computed predictions of neutron yield will fall far below the
  experimental observations." SAND2009 line 511-512: "the discrepancy between
  computed and experimental neutron yields was expected due to the inability of
  MHD to model non-thermonuclear production mechanisms."
- Units / symbol map: yields are dimensionless neutron counts per shot;
  ion-tail energies in keV/MeV.
- Validity range: applies to the MHD/hybrid/kinetic model hierarchy generally;
  the LLNL device is ~180 kA, an other-scope source relative to PF-1000/Akel.
- Current code status: `blocked` for first-principles authority. The MHD core
  of DPF-Unified is a fluid code; per these sources a fluid code cannot by
  itself separate or quantify beam-target production. The candidate PIC path
  (`pic_yield.py`, `kinetic_yield.py`) is the only kinetic-ion route and it is
  `candidate` (self-declares `not_mechanism_separated`,
  `kinetic_yield.py:116-117`).

### 1.5 Ion energy distribution function

- Source: `...z-pinch.md:130-137`: "Figure 6 shows the ion energy distribution
  inside the z = 4 to 6 cm region for the hybrid and fully kinetic simulations.
  The fully kinetic simulation predicts multiple MeV ions... the kinetic
  simulation predicts ~12 keV ions and ~3 keV electrons in the hottest part of
  the pinch region." Hybrid "does not predict ions of energies greater than
  ~200 keV" (`...z-pinch.md:141-143`).
- Source: `KnowledgeReference/fully-electromagnetic-hybrid-pic-fluid-dpf-neutron-yield-acb71fa9.md:1214-1240`:
  the hybrid yield is "more sensitive to alpha, because fusion reactions are
  dominated by the high energy tail of the ion distribution"; a Te = alpha*Ti
  scan with alpha in {0.5, 2.0} changes total yield from 2.86e5 to 5.30e5 vs
  2.96e6 baseline.
- Units / symbol map: ion distribution `f(E)` over kinetic energy E
  (keV/MeV); the neutron rate weights `f(E)` by `sigma(E)` and relative speed.
- Validity range: the distribution is region-resolved (KR cites the
  z = 4-6 cm pinch region) and time-resolved; it is not a single scalar.
- Current code status: `candidate`. `pic_yield.py:81-136`
  (`pic_neutron_yield_rate`) consumes per-macroparticle velocities and weights —
  it does have access to the ion distribution. But the histogram of `f(E)` is
  not emitted as a runtime channel, and the distribution is not separated into
  a thermal core vs a beam tail. So mechanism separation is `blocked`.

### 1.6 Beam transport / stopping in the target

- Source (mechanism premise): `...109633.md:202-204` — the deuteron beam
  "traverses the pinch column of deuterium plasma to produce the fusion
  neutrons"; the Lee/Saw eq. (1) `ln(b/rp)` factor and `zp` path length encode
  a transport path length geometrically.
- No KR file in this corpus gives a quantitative deuteron stopping-power /
  range formula (Bethe stopping, plasma stopping, or beam-energy-loss ODE) for
  the DPF beam. Searched: `neutron`, `stopping`, `beam-target`, `deuteron`,
  `dE/dx`, `range`. The Lee/Saw form treats `sigma(E_beam)` at a single fixed
  energy `E_beam = 3*Vmax`; it does not integrate energy loss along the path.
- Units / symbol map: a stopping model would need `dE/dx` [J/m or keV/m] vs
  beam energy and target density; path length `L` [m].
- Validity range: not available from KR.
- Current code status: `blocked` — no KR source for a deuteron stopping model.
  The single-energy `sigma(3*Vmax)` in `beam_target.py:240-245` is a fixed-
  energy approximation, not a transported beam. Flag in Section 4.

### 1.7 Neutron energy spectrum (thermonuclear broadening + beam-target shift)

- Source: `KnowledgeReference/neutron-generation-dynamics-inside-a-ma-class-dense-plasma-focus-z-pinch-5.md:561-588`:
  "At stagnation, the spectrum is narrow around the 2.45 MeV energy expected in
  thermonuclear processes. As time evolves and the neutron generation shifts
  from thermal processes to beam-target, the energy distribution broadens,
  showing a larger amount of higher energy neutrons, up to 5 MeV... thermal
  processes will not lead to significant energy spread, whereas the presence of
  beam-target events will be marked by broader spectrum."
- Source: `...z-pinch-5.md:555-560`: "a shift of the ion beam distribution
  toward higher energy would cause the neutrons to shift to higher energies and
  would increase the neutron fluence in the direction parallel to the beam."
- Source (TOF-derived spectral fit form): `KnowledgeReference/anisotropy-of-the-emission-of-dd-fusion-neutrons-caused-by-the-plasma-focus-vessel-527cc533.md:366-378`,
  shifted-Maxwell-Boltzmann TOF signal:
  ASCII `S(L,t) ~ (L^2 / t^5) * exp(-(m/2kT) * (L/t - uCM)^2)`, eq. (4), where
  `m` = neutron mass, `k` = Boltzmann constant, `T` = temperature, `uCM` =
  centre-of-mass velocity of the expanding neutron bunch.
- Units / symbol map: spectrum `f(E_n)` over neutron energy `E_n` [eV/MeV];
  TOF form: `L` [m], `t` [s], `T` [keV], `uCM` [m/s].
- Validity range: the 2.45 MeV narrow / up-to-5 MeV broadened picture is MJ-
  class MJOLNIR (other scope vs PF-1000/Akel). The shifted-Maxwell form is a
  PF-1000-vessel-context TOF-fit form, full-energy scope.
- Current code status: `candidate`. `neutron_tof.py:27-84` produces a
  thermonuclear Gaussian (Doppler `sigma = 82.5*sqrt(Ti_keV)` keV — see Section
  4 note) and a kinematically shifted beam-target Gaussian. It is a synthetic
  generator with no KR-cited broadening law and no detector response; status
  `candidate`, mechanism-separation `blocked`.

### 1.8 Neutron anisotropy

- Source: `...527cc533.md:199-204`: anisotropy "defined as the ratio of total
  neutron yields `Yn(psi)/Yn(90 deg)`... where psi is an angle of observation
  related to the downstream direction. The neutron emission anisotropy caused
  by the PF-1000 vessel ranges from 0.30 to 1.1." KR lines 432-438: the
  experimental anisotropy coefficient `Yn(0 deg)/Yn(90 deg)` "must be completed
  by the calculated anisotropy which is caused by the discharge vessel."
- Source: `...z-pinch-5.md:592-613` (MJOLNIR): "the neutron emission from
  thermal processes will be more isotropic than the beam-target mechanism...
  higher yields are accompanied by higher anisotropy with higher yield reported
  in the forward direction suggesting an increasing beam-target contribution at
  higher yields." On-axis/off-axis ratio rises to 80-100% above off-axis at
  high yield.
- Units / symbol map: anisotropy `A = Yn(psi)/Yn(90 deg)`, dimensionless;
  `psi` = polar angle from the downstream axis [deg].
- Validity range: the 0.30-1.1 vessel-scattering range is PF-1000 at
  450-500 kJ / 3.5 Torr (other scope vs Akel 16 kV). The MJOLNIR trend is MJ-
  class. The physical statement "thermal isotropic, beam-target forward-peaked"
  is general.
- Current code status: `candidate` for runtime diagnostic. `beam_target.py:439-490`
  (`neutron_anisotropy`) returns a yield-weighted ratio, but the beam-target
  anisotropy law `A_bt = 1 + 0.3*sqrt(E_beam/100 keV)` is explicitly an
  uncited empirical model (the docstring says "approximately", alpha ~ 0.3).
  Mechanism authority `blocked`: no KR-cited intrinsic-emission anisotropy law,
  and the vessel-scattering contribution is not modelled at all.

### 1.9 Detector response — TOF scintillator

- Source: `...527cc533.md:177-180`: "TOF spectra of neutrons and hard x-rays
  filtered by a 75 mm lead were observed with probes composed of fast
  photo-multipliers and BC408 scintillators. The TOF probes were positioned at
  distances of 7, 16.3 and 58.3 m."
- Source: `KnowledgeReference/radiation-physics-and-chemistry-188-2021-109633.md:115-123`
  (Akel/PF-1000): "Three scintillation detectors were used to record signals at
  a distance of 7 m from the focus center in the downstream (at 0 deg),
  upstream (at 180 deg), and side-on (at 90 deg) directions... The mean values
  of the energy of the fusion neutrons and primary deuterons were determined
  with the time-of-flight method from the temporal difference in the amplitudes
  of the neutron signals recorded downstream and upstream."
- Source (TOF signal model): `...527cc533.md:284-301`, Kelly TOF transform,
  eqs. (1)-(2): ASCII `S(L,t) ~ (L^2/t^5) * f(L/t)` and inversion
  `f(v) ~ v^-5 * S(L/v)`. Burn-time/velocity-width relation eq. (3):
  `delta_t_TOF = delta_t_FBT + L/delta_u`.
- Source (fusion burn time vs TOF): `...527cc533.md:328-337`: fitted
  `delta_t_FBT = 45 ns`, velocity width `delta_u = 8.2e8 m/s`.
- Units / symbol map: `S(L,t)` detector signal; `L` [m] source-detector
  distance; `t` [s] TOF; `f(v)` velocity distribution; `delta_t_FBT` fusion
  burn time [s].
- Validity range: PF-1000 full energy / NNSS deuterium DPF — both other-scope
  relative to a generic same-scope runtime, but the Akel 109633.md detector
  layout IS the declared Akel scope.
- Current code status: `blocked`. `neutron_tof.py` generates a synthetic
  spectrum but applies no detector impulse response, no finite time
  resolution, no `L^2/t^5` TOF kernel, and no PMT/scintillator transfer
  function. There is no runtime detector-response model.

### 1.10 Detector response — activation counters (silver, Be/Y/Br)

- Source: `...109633.md:124-131` (Akel/PF-1000, the declared scope):
  "The total neutron yield was measured using two calibrated silver activation
  counters, which were placed outside the main experimental chamber. These
  counters were calibrated using an Am-Be source placed on the anode axis. The
  uncertainty in the neutron yield measurements were estimated as +-0.2"
  (i.e. +-0.2e9 on the ~e9 yields).
- Source: `...z-pinch-5.md:138-144` (MJOLNIR): "The neutron yield is
  characterized using three different activation reactions from Be, Y, and Br.
  The beryllium activation is measured using two separate photomultiplier
  tubes and has been absolutely calibrated at the Sandia Ion Beam Laboratory.
  LaBr detectors with yttrium caps... to assess the anisotropy."
- Units / symbol map: activation counter response maps a 4-pi-integrated
  neutron fluence to a measured count; calibration constant from an Am-Be or
  accelerator source; yield uncertainty `+-0.2e9` neutrons (Akel).
- Validity range: the +-0.2e9 figure is specifically the Akel PF-1000
  16 kV / 1.05-1.2 Torr campaign (declared scope).
- Current code status: `blocked`. No runtime activation-counter response model
  exists. The +-0.2e9 figure is a text-supported reference channel in
  `neutron_authority.py` (`yield_uncertainty_scalar`,
  `am_be_activation_calibration_text`) — reference only, not an accepted
  authority channel.

### 1.11 Activation / TOF response, scatter, and multi-pinch

- Source (scatter fraction): `...527cc533.md:205-209`: "The computation led to
  the estimate that at least 54% of the produced neutrons are scattered by the
  PF-1000 vessel and by the laboratory equipment." KR lines 280-283: "it is
  very important to separate scattered neutrons from direct ones in the TOF
  spectrum before the transformation... Without their separation the direct
  transformation in the velocity or energy distribution is disputable."
- Source (multi-pinch superposition): `...527cc533.md:379-392`, eq. (5):
  ASCII `S(L,t) = sum_i Si(Li, t - ti,0)` — when the discharge produces two or
  more z-pinches, the detector signal is the sum of partial signals with
  distinct origin times `ti,0`. Fitted PF-1000 example: two fusion burns at
  `t1,0 = 0 ns` and `t2,0 = 293 ns`, `T1 = 4.3 keV`, `T4 = 6.5 keV`,
  Doppler shifts `~370 keV` and `~70 keV` (`...527cc533.md:406-412`).
- Source (tomographic inversion): `KnowledgeReference/tomographic-reconstruction-of-the-neutron-time-energy-spectrum-from-a-dense-plasma-focus-b78f1154.md:122-133`,
  the ToF spectroscopy model, eq. (1):
  ASCII `S(X,t') = integral integral f(t,E) * delta(t - t' + X/sqrt(2E/m0)) dE dt`,
  where `X` = source-detector distance, `t'` = detection time, `f(t,E)` = the
  time-energy spectrum to be reconstructed, `m0` = neutron rest mass. The
  Catenacci method uses "detector pairs set up in a geometry that allows for
  scatter background subtraction" (`b78f1154.md:48-49`).
- Units / symbol map: `f(t,E)` neutron creation time-energy density;
  `S(X,t')` detector signal; scatter fraction dimensionless.
- Validity range: PF-1000 vessel (full energy) and NNSS DPF — other scope vs
  Akel, but the TOF-inversion math and scatter-separation requirement are
  general.
- Current code status: `blocked`. No runtime scatter-transport or TOF-
  inversion code. `beam_target.py:498-636` (`decompose_neutron_events`) is a
  peak-finder on a neutron-rate trace; it does multi-event splitting in time
  but it is not a detector-response inversion and cites Goyon as `[TRAINING]`-
  style context, not the KR multi-pinch superposition eq. (5).

### 1.12 Uncertainty quantification (UQ) hooks

- Source (closure-sensitivity UQ): `...acb71fa9.md:1226-1240`: a Te = alpha*Ti
  scan with alpha in {0.5, 2.0} gives total yields 2.86e5 / 5.30e5 vs 2.96e6
  baseline — "at least a factor of a few uncertainty in the absolute neutron
  yield" from the electron-temperature closure. The paper states the yield
  result "should be interpreted as an order-of-magnitude validation rather than
  as a precise prediction" (`acb71fa9.md:1259-1266`).
- Source (cross-section UQ): `bosch-hale-1992-fusion-reactivity.md:108-109`:
  absolute reactivity uncertainty 6% for D-D from the R-matrix cross sections.
- Source (measurement UQ): `...109633.md:130-131`: `+-0.2e9` neutron-yield
  measurement uncertainty (silver activation).
- Units / symbol map: yield uncertainty as a multiplicative factor (closure)
  and an additive band (measurement); cross-section uncertainty as a percent.
- Validity range: closure-sensitivity figure is the hybrid-PIC compact-device
  scope; the 6% is the Bosch-Hale D-D reactivity fit; +-0.2e9 is Akel.
- Current code status: `candidate`. `kinetic_neutron_yield_authority_status`
  in `kinetic_yield.py:149-194` already takes `uncertainty_evidence` and
  `temperature_authority` arguments and blocks when they are missing — the UQ
  hook exists as an interface, but no runtime UQ budget is computed or emitted.

---

## 2. Mechanism status table: supported / candidate / blocked

One row per mechanism. "Supported" = a verbatim KR equation/cross-section
exists and is implemented faithfully. "Candidate" = code exists but the
physics law is uncited/empirical, or the path is a runtime diagnostic not
wired to mechanism-separated authority. "Blocked" = no KR source for the
mechanism, or the mechanism cannot be produced by the current model class.

| # | Mechanism | Status | KR source (file : lines) | Reason for status |
|---|-----------|--------|--------------------------|-------------------|
| 1 | Thermonuclear production (Bosch-Hale reactivity) | `candidate` | `bosch-hale-1992-fusion-reactivity.md:59-93,106-109`; `2019nrlplasma-formulary-037290d4.md:3802-3814` | Reactivity fit + 2.45 MeV branch are verbatim KR. The volumetric `(1/4) n_D^2 <sigma v> V` integrator exists but the 1/4 prefactor has no KR citation and the path is not wired to first-principles runtime. |
| 2 | Beam-target production | `candidate` | `radiation-physics-and-chemistry-188-2021-109633.md:195-215` (Lee eq. 1); `fully-kinetic-...-z-pinch.md:38-43,152-161`; `sand2009-6373-b93aec67.md:345-355,394-397,511-512` | Lee eq. (1) is a fitted reduced model (comparator only). A first-principles beam-target mechanism needs kinetic ions; the fluid MHD core cannot produce it. |
| 3 | Ion energy distribution function | `blocked` | `fully-kinetic-...-z-pinch.md:130-143`; `acb71fa9.md:1214-1240` | PIC path can access per-particle velocities but does not emit `f(E)` or split thermal-core vs beam-tail. No mechanism-separated distribution channel. |
| 4 | Beam transport / stopping | `blocked` | (no KR source) | No KR file gives a deuteron stopping-power / range / energy-loss formula. `beam_target.py` uses a single fixed `sigma(3*Vmax)`. |
| 5 | Neutron energy spectrum | `candidate` | `neutron-generation-...-z-pinch-5.md:561-588`; `anisotropy-...-527cc533.md:366-378` (shifted-MB eq. 4) | Qualitative broadening picture + a TOF-fit shifted-Maxwell form are KR-cited. `neutron_tof.py` uses an uncited Doppler width and a synthetic kinematic shift; no detector response. |
| 6 | Neutron anisotropy | `candidate` | `anisotropy-...-527cc533.md:199-204,432-438`; `z-pinch-5.md:592-613` | Definition `Yn(psi)/Yn(90)` and the "thermal isotropic / beam forward-peaked" rule are KR-cited. `beam_target.py:439-490` uses an uncited `A_bt` law; vessel-scattering contribution not modelled. |
| 7 | Detector response (TOF + activation) | `blocked` | `527cc533.md:177-180,284-301,328-337`; `109633.md:115-131`; `b78f1154.md:122-133` | KR gives the TOF kernel `L^2/t^5`, the inversion `f(v)~v^-5 S(L/v)`, burn-time relation, and the activation-counter layout. No runtime detector-response or TOF-inversion code exists. |
| 8 | Scatter / activation-TOF transport | `blocked` | `527cc533.md:205-209,280-283,379-392` (eq. 5); `b78f1154.md:48-49` | KR requires direct-vs-scattered separation before any TOF inversion (>=54% scattered on PF-1000). No runtime scatter-transport model. |
| 9 | UQ (closure + cross-section + measurement) | `candidate` | `acb71fa9.md:1226-1240,1259-1266`; `bosch-hale-1992-fusion-reactivity.md:108-109`; `109633.md:130-131` | The authority interface already takes `uncertainty_evidence` and `temperature_authority`. No runtime UQ budget is computed/emitted. |

Counts: `supported` 0; `candidate` 5 (rows 1, 2, 5, 6, 9); `blocked` 4
(rows 3, 4, 7, 8).

Overall packet status (unchanged from existing `neutron_authority.py`):
`blocked_mechanism_separated_neutron_authority_not_available`. No mechanism is
fully `supported` for first-principles authority; total neutron yield must
remain non-authoritative.

---

## 3. Runtime fields required

### 3.1 Per-mechanism consumed runtime state

| Mechanism | Runtime state consumed | Centering / sampling |
|-----------|------------------------|----------------------|
| Thermonuclear | `n_D(x,t)` deuterium number density; `Ti(x,t)` ion temperature; `cell_volume(x)`; integration `dt` | cell-centered; per-cell; per-step |
| Beam-target | beam ion population: per-macroparticle position, velocity, weight, species tag; target `n_D(x,t)`; pinch geometry `zp`, `rp`, `b` | particles + cell-centered target; per-step |
| Ion distribution | per-macroparticle kinetic energy `E_i`; species tag (thermal core vs beam); region mask (pinch column) | particle-level; binned to `f(E)` histogram |
| Beam stopping | beam energy `E_beam(s)` along path `s`; target `n_D` along path; path length `L` | along-beam line sample (BLOCKED — no model) |
| Spectrum | per-mechanism birth-neutron list: birth time, birth energy, birth direction; `Ti` for thermal Doppler; `uCM` beam drift | particle/event-level emission record |
| Anisotropy | per-mechanism `Yn(psi)` angular-binned yield; intrinsic emission angular distribution; vessel-geometry mask | angular bins in `psi`; per-mechanism |
| Detector response (TOF) | birth time-energy spectrum `f(t,E)`; detector positions `X_k`; detector impulse response; lead/Al filter transmission | per-detector kernel convolution |
| Detector response (activation) | 4-pi-integrated neutron fluence; activation calibration constant; counter geometry | scalar per counter |
| Scatter transport | birth neutron distribution; vessel + room geometry; material cross sections | transport solve (BLOCKED — no model) |
| UQ | closure parameter `alpha = Te/Ti`; cross-section uncertainty %; measurement band; mechanism-separated yield channels | scan + budget aggregation |

### 3.2 The single neutron-authority runtime interface

A first-principles runtime must emit one structured `NeutronAuthorityRuntime`
record per accepted same-scope shot. It is fed into
`build_mechanism_separated_neutron_packet(...)`
(`neutron_authority.py:174`) so the packet builder can decide acceptance. Every
field is fail-closed: absent or unreviewed -> the corresponding channel stays
`missing_or_blocked`.

```
NeutronAuthorityRuntime
  declared_scope: str                       # must match the target scope, e.g.
                                            # "pf1000_akel_16kv_1p2torr_shot_12581"
  device_name: str

  # --- mechanism-separated yield histories (REQUIRED for authority) ---
  thermonuclear_yield_history:              # time series, NOT a scalar
      times_s: float[]
      rate_per_s: float[]
      cumulative: float[]
      source_ref: KRRef                     # Bosch-Hale reactivity
      prefactor_citation: KRRef | None      # 1/4 factor — currently None -> blocked
  beam_target_yield_history:
      times_s: float[]
      rate_per_s: float[]
      cumulative: float[]
      mechanism_basis: enum                 # "kinetic_ion_distribution" |
                                            # "lee_reduced_model"
      source_ref: KRRef
  mechanism_separation_status: enum         # "mechanism_separated" |
                                            # "not_mechanism_separated"

  # --- ion distribution ---
  ion_energy_distribution_history:
      times_s: float[]
      energy_bins_keV: float[]
      f_thermal_core: float[][]             # per-time histogram
      f_beam_tail: float[][]
      pinch_region_mask_hash: str

  # --- transport / stopping ---
  beam_transport_stopping_model:
      status: "blocked_no_kr_source"        # see Section 4
      path_length_m: float | None
      target_density_path: float[] | None

  # --- spectrum ---
  neutron_energy_spectrum:
      energy_bins_eV: float[]
      f_thermonuclear: float[]
      f_beam_target: float[]
      doppler_width_law_ref: KRRef | None   # currently None -> blocked
  neutron_timing_history:
      times_s: float[]
      rate_per_s: float[]
      burn_time_s: float | None

  # --- anisotropy ---
  neutron_anisotropy_angular_yield:
      psi_bins_deg: float[]
      Yn_thermonuclear: float[]
      Yn_beam_target: float[]
      intrinsic_anisotropy_law_ref: KRRef | None   # currently None -> blocked
      vessel_scatter_anisotropy: float[] | None    # currently None -> blocked

  # --- detector response ---
  detector_response_model:
      detector_positions_m: float[]
      tof_kernel_ref: KRRef | None          # L^2/t^5 kernel; None -> blocked
      impulse_response: float[] | None
      filter_transmission: float[] | None
  activation_counter_response_model:
      calibration_constant: float | None
      calibration_source: str | None        # e.g. "Am-Be"
      response_ref: KRRef | None            # None -> blocked
  direct_scattered_neutron_transport:
      scatter_fraction: float | None        # PF-1000: >=0.54 per KR
      direct_spectrum: float[] | None
      scattered_spectrum: float[] | None
      transport_ref: KRRef | None           # None -> blocked

  # --- comparator + UQ ---
  same_scope_scalar_yield:                  # comparator ONLY, not authority
      value: float
      uncertainty: float
      source_ref: KRRef
  yield_uncertainty_budget:
      cross_section_uncertainty_frac: float # 0.06 D-D per Bosch-Hale
      closure_sensitivity_factor: float     # Te=alpha*Ti scan
      measurement_band: float
  electron_temperature_yield_sensitivity_uq:
      alpha_scan: float[]
      yield_per_alpha: float[]
  output_mapping_and_comparator:
      comparator_target_id: str
      mapped_observables: str[]
  source_review_certificate:
      reviewer: str | None
      review_date: str | None
      review_status: enum                   # absent -> blocked
```

`KRRef = { path: str, lines: str, role: str }` — same shape as
`NEUTRON_AUTHORITY_SOURCE_REFS` in `neutron_authority.py:8-74`.

Interface contract: a record is acceptance-eligible only if (a)
`declared_scope` matches the target scope, (b) `mechanism_separation_status ==
"mechanism_separated"`, (c) every `*_ref` field used by a blocking channel is
non-null and points at a reviewed same-scope source, and (d)
`source_review_certificate.review_status` is a passed review. Any field left
`None` keeps its channel in `missing_acceptance_channels`. `same_scope_scalar_yield`
alone never satisfies (b) — it is comparator-only by construction.

---

## 4. Missing parameters — values/cross-sections with no KR source

| Item | Where it would be needed | KR status |
|------|--------------------------|-----------|
| Thermonuclear volumetric prefactor (the `1/4` in `dY/dt = (1/4) n_D^2 <sigma v> V`) | `neutron_yield.py:166-168` | No verbatim KR formula. KR supplies `<sigma v>` only. The 1/4 (identical-particle 1/2 x neutron-branch 1/2) is standard but `[INFERRED]`; needs a cited source for the full reaction-rate equation. |
| Deuteron stopping power / beam range in deuterium plasma | beam-target transport (Section 1.6) | No KR source. No Bethe / plasma-stopping / energy-loss ODE in this corpus. |
| Doppler broadening law for the thermonuclear neutron line | `neutron_tof.py:47-49` uses `sigma = 82.5*sqrt(Ti_keV)` keV; docstring attributes "Brysk 1973" | Not in KR. The "177*sqrt(Ti_keV) keV FWHM" / "82.5" coefficient has no KnowledgeReference file. The KR-cited alternative is the shifted-Maxwell TOF form `527cc533.md:366-378`. |
| Beam-target lab-frame kinematic neutron-energy shift law | `neutron_tof.py:64,79-84`; `beam_target.py:455-460` | Not in KR as an equation. KR (`z-pinch-5.md:555-560`, `527cc533.md:323-325`) gives the qualitative direction and a `~370 keV` measured Doppler shift, not a closed-form shift law. |
| Intrinsic beam-target emission anisotropy law `A_bt(E_beam)` | `beam_target.py:481` `A_bt = 1 + 0.3*sqrt(E_beam/100 keV)` | Not in KR. Coefficient 0.3 is uncited/empirical. KR gives only ratios and trends. |
| Vessel/room neutron-scattering response (anisotropy + spectral degradation) | direct/scattered separation (Section 1.11) | KR (`527cc533.md`) gives MCNP-computed *results* (>=54% scattered, 0.30-1.1 anisotropy range) for the PF-1000 vessel at full energy, but no transferable model and no Akel-scope numbers. |
| TOF detector impulse response / PMT transfer function | detector-response model (Section 1.9) | KR names "BC408 scintillators", "fast photo-multipliers" and the `L^2/t^5` kernel but gives no impulse-response function. |
| Activation-counter calibration constant for the Akel silver counters | activation response (Section 1.10) | KR states the counters were Am-Be calibrated and the yield uncertainty is `+-0.2e9`, but gives no calibration constant. |
| Beam-target `Cn` reconciliation | `beam_target.py:177` (`Cn = 1.810e7`) vs `109633.md:213` (`Cn = 8.54e8`) | Both KR-sourced but from different papers and different calibration points (Lee/Saw course 0.5 MA / `Yn = 7e9` vs Akel 2021 0.5 MA / graphical multi-device fit). Not a missing value, but an unreconciled conflict — the Lee form is comparator-only either way. |

These items are why the `blocked` rows in Section 2 are `blocked` and why
mechanism-separated authority cannot be granted. Per the parallel-lane rule,
none of these may be implemented from an inferred formula until the exact
source packet, equation, units, symbol map, and validity range are attached.

---

## 5. Proposed tests and fail-closed negative controls

These extend `tests/test_first_principles_neutron_authority.py`. They are
proposed, not part of this submission (no code is submitted in this packet).

### 5.1 Negative controls (must fail closed)

1. `test_scalar_yield_only_does_not_separate_mechanisms` — a
   `NeutronAuthorityRuntime` with only `same_scope_scalar_yield` populated and
   every mechanism history empty must yield
   `mechanism_separation_status != "mechanism_separated"` and
   `can_support_total_yield_acceptance == False`. (Strengthens the existing
   `test_scalar_total_yield_only_cannot_accept_neutron_authority`.)
2. `test_thermonuclear_only_does_not_grant_total_authority` — a record with a
   populated `thermonuclear_yield_history` but empty `beam_target_yield_history`
   must stay blocked: per `sand2009-6373-b93aec67.md:345-355,511-512` a
   thermonuclear-only result cannot represent total DPF yield.
3. `test_lee_reduced_model_basis_is_comparator_not_authority` — a
   `beam_target_yield_history` with `mechanism_basis == "lee_reduced_model"`
   must not count toward acceptance (Lee eq. (1) is a fitted model,
   `109633.md:205-215`).
4. `test_missing_prefactor_citation_blocks_thermonuclear` — a
   `thermonuclear_yield_history` with `prefactor_citation == None` keeps the
   thermonuclear channel `missing_or_blocked` (the `1/4` factor is uncited).
5. `test_blocked_stopping_model_blocks_beam_target_transport` — a record whose
   `beam_transport_stopping_model.status == "blocked_no_kr_source"` keeps
   beam-target authority blocked.
6. `test_missing_detector_response_blocks_authority` — already covered by
   `test_detector_response_required_before_kinetic_yield_authority`; extend to
   the full runtime record.
7. `test_missing_scatter_separation_blocks_tof_spectrum` — a
   `neutron_energy_spectrum` derived from raw detector signal without
   `direct_scattered_neutron_transport` populated must not be accepted
   (`527cc533.md:280-283`: inversion without scatter separation is "disputable").
8. `test_missing_uq_budget_blocks_authority` — covered by
   `test_uq_required_before_kinetic_yield_authority`; extend to the full record.
9. `test_te_closure_sensitivity_blocks_when_factor_of_a_few` — if
   `electron_temperature_yield_sensitivity_uq` shows a yield spread of a factor
   of a few across the alpha-scan and no Te authority is attached, block
   (`acb71fa9.md:1226-1240`).
10. `test_cross_scope_record_rejected` — covered by
    `test_cross_scope_target_cannot_accept_pf1000_akel_neutron_authority`;
    extend so a `NeutronAuthorityRuntime` with a mismatched `declared_scope` is
    rejected wholesale.
11. `test_anisotropy_without_intrinsic_law_blocked` — a populated
    `neutron_anisotropy_angular_yield` with `intrinsic_anisotropy_law_ref ==
    None` (uncited `A_bt`) keeps the anisotropy channel blocked.

### 5.2 Positive structural controls (interface shape, not physics validation)

12. `test_runtime_record_round_trips_into_packet` — a fully populated synthetic
    `NeutronAuthorityRuntime` is accepted by
    `build_mechanism_separated_neutron_packet` without raising, and the packet
    still returns `can_support_first_principles_acceptance == False` until a
    passed `source_review_certificate` is present.
13. `test_mechanism_histories_are_time_series_not_scalars` — assert
    `thermonuclear_yield_history` and `beam_target_yield_history` carry
    `times_s`/`rate_per_s` arrays of equal length > 1; a scalar must be
    rejected.
14. `test_every_blocking_channel_has_a_kr_ref_slot` — assert each blocking
    channel in `BLOCKING_NEUTRON_AUTHORITY_CHANNELS` maps to a `KRRef`-typed
    field in the runtime schema.

Key fail-closed principle for the whole suite: **scalar-yield-only must not
pass mechanism separation.** A record that agrees with the measured
`6.14e9` scalar yield (`109633.md:282-288`) but carries no separated
thermonuclear and beam-target histories must return
`mechanism_separation_status != "mechanism_separated"` and
`can_support_total_yield_acceptance == False`.

---

## 6. Exact implementation recommendations

The neutron-authority interface contract and a same-scope evidence map.
No code is changed by this packet; these are the recommendations a future
implementation diff must follow (with code + tests in the same diff, per
parallel-lane acceptance rule 4).

### 6.1 Interface contract

1. Define `NeutronAuthorityRuntime` (Section 3.2) as a frozen dataclass in
   `src/dpf/first_principles/neutron_authority.py`, alongside the existing
   `build_mechanism_separated_neutron_packet`. Every physics field carries a
   `KRRef` slot; a `None` ref keeps the channel `missing_or_blocked`.
2. `build_mechanism_separated_neutron_packet` gains an optional
   `runtime: NeutronAuthorityRuntime | None` argument. When present, the
   builder reads `mechanism_separation_status` and the per-channel `*_ref`
   slots instead of only `validation_targets`. The hardcoded
   `can_support_first_principles_acceptance = False`
   (`neutron_authority.py:259`) and the unconditional
   `missing.update(BLOCKING_NEUTRON_AUTHORITY_CHANNELS)`
   (`neutron_authority.py:200`) stay until every blocking channel has a
   reviewed same-scope ref AND a passed `source_review_certificate`.
3. Mechanism-separated histories MUST be time series. The thermonuclear and
   beam-target channels each carry `times_s` / `rate_per_s` / `cumulative`
   arrays. A scalar total is recorded only in `same_scope_scalar_yield` and is
   tagged comparator-only.
4. Keep the Lee/Saw form (`beam_target.py`) as `mechanism_basis =
   "lee_reduced_model"`. The builder must never count a `lee_reduced_model`
   basis toward acceptance — it is comparator/baseline only
   (`BEAM_TARGET_SOURCE_STATUS = "baseline_or_runtime_diagnostic_not_validation"`).
5. Thermonuclear authority is blocked until the `1/4` volumetric prefactor has
   a cited source. Until then, `prefactor_citation` is `None` and the
   thermonuclear channel stays `missing_or_blocked` even when
   `thermonuclear_yield_history` is otherwise populated.
6. Detector response, scatter transport, and beam stopping are `blocked`
   (Section 4). Their runtime fields exist as typed slots that default to
   `None`/`"blocked_no_kr_source"` so the packet fails closed and the gap is
   explicit, not silent.
7. Reuse `kinetic_neutron_yield_authority_status` (`kinetic_yield.py:149`) as
   the gate for the kinetic-ion route — it already requires kinetic yield,
   mechanism separation, detector response, UQ, and Te authority.

### 6.2 Same-scope evidence map

Acceptance is per declared scope `pf1000_akel_16kv_1p2torr_shot_12581`. A
source is `same_scope` only if it is the Akel PF-1000 16 kV / 1.05-1.2 Torr
campaign. Everything else is `other_scope` and usable only for requirements or
schema (`cross_scope_policy.can_use_other_scope_for_acceptance == False`,
`neutron_authority.py:229`).

| Channel | Same-scope evidence | Other-scope schema source | Map decision |
|---------|---------------------|---------------------------|--------------|
| Scalar yield (comparator) | `109633.md:282-288` shot 12581 `6.14e9`; `:130-131` `+-0.2e9` | — | `accepted` as comparator only |
| Detector layout (TOF 0/90/180 deg) | `109633.md:115-123` Akel 7 m scintillators | `527cc533.md:177-180` PF-1000 full energy | `text_supported` reference; no runtime model -> channel `blocked` |
| Activation counters | `109633.md:124-131` Akel silver, Am-Be | `z-pinch-5.md:138-144` MJOLNIR Be/Y/Br | `text_supported`; no runtime model -> `blocked` |
| Thermonuclear reactivity | `bosch-hale-1992-fusion-reactivity.md:59-93` (device-independent) | — | physics `supported`; volumetric integrator `candidate` (prefactor uncited) |
| Beam-target reduced model | `109633.md:195-215` Lee eq. (1), Akel | — | comparator only; never authority |
| Mechanism separation schema | — | `z-pinch-5.md:433-445,561-613` MJOLNIR; `sand2009...:345-355` | schema only; `blocked` for Akel |
| Spectrum / anisotropy / TOF inversion | — | `527cc533.md`, `b78f1154.md`, `z-pinch-5.md` (all other scope) | schema only; `blocked` for Akel |
| Ion distribution / kinetic yield | — | `fully-kinetic-...-z-pinch.md` (LLNL); `acb71fa9.md` (compact) | requirement + candidate PIC diagnostic; `blocked` for Akel authority |
| UQ | `109633.md:130-131` (`+-0.2e9`, same scope); `bosch-hale-...:108-109` (6%) | `acb71fa9.md:1226-1240` closure factor | UQ hook `candidate`; budget not computed |

Conclusion of the map: for the Akel scope, **no mechanism channel has same-
scope evidence sufficient for authority.** The only same-scope quantitative
data are the scalar yield and its `+-0.2e9` band — both comparator-only. Every
mechanism-separation, spectrum, anisotropy, detector-response, and scatter
source is other-scope and contributes schema/requirements only. The packet
therefore stays
`blocked_mechanism_separated_neutron_authority_not_available`.

---

## 7. Explicit "do not promote" notes

1. **Scalar total neutron yield is not mechanism authority.** Agreement
   between a computed scalar yield and the measured Akel `6.14e9`
   (`109633.md:282-288`) is a baseline comparison only. It does not
   demonstrate that the thermonuclear and beam-target mechanisms are correctly
   separated or individually correct. A scalar match can occur with both
   mechanisms wrong in compensating directions.
2. **The Lee/Saw beam-target formula is a fitted reduced model, not first-
   principles authority.** `109633.md:205-215` and `:858-861`: `fc` is held
   constant at 0.7 and four model parameters are tuned per shot to fit the
   measured current waveform. `beam_target.py` correctly tags it
   `baseline_or_runtime_diagnostic_not_validation`. Do not promote it.
3. **A fluid MHD result cannot represent total DPF neutron yield.**
   `sand2009-6373-b93aec67.md:345-355,394-397,511-512` and
   `fully-kinetic-...-z-pinch.md:38-43,152-161`: MHD models only the
   thermonuclear contribution; beam-target production requires kinetic ions.
   Any neutron number from the fluid core is a thermonuclear lower bound, not a
   total.
4. **The candidate PIC yield history is a runtime diagnostic, not authority.**
   `kinetic_yield.py:116-117` self-declares
   `mechanism_separation_status = "not_mechanism_separated"`;
   `acb71fa9.md:1259-1266` calls the integral yield "order-of-magnitude
   validation rather than a precise prediction" with "a factor of a few"
   closure uncertainty. It must surface only as a `candidate_*` channel.
5. **`blocked` channels must not be silently filled.** Beam stopping, intrinsic
   anisotropy law, Doppler broadening law, detector impulse response, and
   scatter transport (Section 4) have no KR source. They must not be
   implemented from training-data formulas. Their runtime slots default to
   `None` / `"blocked_no_kr_source"` so the gap stays visible.
6. **Other-scope neutron diagnostics do not transfer to the Akel scope without
   a reviewed transfer rule.** `neutron_authority.py:130-140`
   (`TRANSFER_RULE_REQUIRED_CHANNELS`) and `cross_scope_policy`
   (`...:225-230`): MJOLNIR (MJ-class), PF-1000 full energy (450-500 kJ), LLNL
   (~180 kA), and NNSS sources are other-scope. They supply requirements and
   schema only.
7. This packet does not change
   `can_support_first_principles_acceptance` — it stays `false`. It does not
   mark any runtime feature `implemented`. No validation or acceptance is
   promoted.

---

## 8. Source reference index (this packet)

All citations are local `KnowledgeReference/` files, verified by direct read
this session.

- `KnowledgeReference/bosch-hale-1992-fusion-reactivity.md` — DD reactivity fit
  (Eqs. 12-14), Table VII coefficients, validity 0.2-100 keV, 6% D-D
  uncertainty.
- `KnowledgeReference/2019nrlplasma-formulary-037290d4.md` — DD reaction
  channels and Q-values (lines 3802-3814).
- `KnowledgeReference/radiation-physics-and-chemistry-188-2021-109633.md` —
  Akel 2021 PF-1000 16 kV: Lee eq. (1), shot 12581 `6.14e9`, `+-0.2e9`
  uncertainty, silver/Am-Be activation, 0/90/180 deg scintillators, Lee is a
  fitted model.
- `KnowledgeReference/fully-kinetic-simulations-of-dense-plasma-focus-z-pinch.md`
  — fluid predicts no neutrons; hybrid under-predicts ~100x; only fully kinetic
  reaches experimental yield and MeV ions.
- `KnowledgeReference/sand2009-6373-b93aec67.md` — MHD models only the
  thermonuclear contribution; DPF yield is largely non-thermonuclear.
- `KnowledgeReference/neutron-generation-dynamics-inside-a-ma-class-dense-plasma-focus-z-pinch-5.md`
  — Goyon 2025 MJOLNIR: thermonuclear-at-stagnation vs beam-target-at-disruption
  separation, spectral broadening, forward anisotropy trend.
- `KnowledgeReference/anisotropy-of-the-emission-of-dd-fusion-neutrons-caused-by-the-plasma-focus-vessel-527cc533.md`
  — PF-1000 vessel: anisotropy definition, >=54% scattered, shifted-Maxwell TOF
  form (eq. 4), multi-pinch superposition (eq. 5), burn-time relation.
- `KnowledgeReference/tomographic-reconstruction-of-the-neutron-time-energy-spectrum-from-a-dense-plasma-focus-b78f1154.md`
  — Catenacci 2020: ToF tomographic inversion model (eq. 1), detector-pair
  scatter-background subtraction.
- `KnowledgeReference/fully-electromagnetic-hybrid-pic-fluid-dpf-neutron-yield-acb71fa9.md`
  — hybrid PIC-fluid: yield from the simulated ion distribution, Te closure
  sensitivity (factor of a few), order-of-magnitude caveat.

Provenance note: numeric values quoted (`6.14e9`, `+-0.2e9`, `0.86e7`,
`3.6e4`, `2.96e6`, `2.86e5`/`5.30e5`, `0.30-1.1`, `>=54%`, `45 ns`, `6%`) are
copied verbatim from the cited KR line ranges; none were generated from
training data. Items with no KR source are listed in Section 4 and are not
asserted as values.
