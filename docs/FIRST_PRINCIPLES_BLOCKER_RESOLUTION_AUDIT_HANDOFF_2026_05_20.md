# First-Principles DPF Blocker Resolution — Audit Handoff (2026-05-20)

- Repository: `/Users/anthonyzamora/dpf-unified`
- Branch: `codex/corpus`
- HEAD at research-completion: `022b774`
- Research method: 7 parallel sub-agent fan-out across KR + on-disk corpus + legacy DB
- Audience: the audit team verifying that every Sprint 4 named blocker has a definite resolution path (existing KR, KR promotion, existing-KR target extraction, or external acquisition with full citation)

> Supersession note 2026-05-20: this V1 handoff is retained as historical
> triage. Sprint 6 user-supplied Scholz 2001 target extraction supersedes the
> V1 `PF1000-BLK-015` verdict: insulator outer radius is now source-available
> for the 2001 24-rod PF-1000 revision and remains blocked only for runtime
> revision mapping/acceptance. Use
> `docs/FIRST_PRINCIPLES_BLOCKER_RESOLUTION_LEDGER_2026_05_20.csv` as the
> normalized current ledger.

This handoff converts every Sprint 4 named blocker into one of four resolution categories. The goal is **not** to assert physics — every recommendation is documented with the exact local path or full external citation so the audit team can verify independently. Nothing in this document promotes a first-principles, validation, or whole-shot acceptance claim.

## Resolution Category Glossary

| Category | Meaning |
| --- | --- |
| `RESOLVED_VIA_KR` | An existing KR file holds the value or formula with the correct scope; cite path + line range. |
| `KR_PROMOTION_RECOMMENDED` | The source PDF is on disk but not yet target-extracted into KR; promotion resolves the blocker. |
| `EXISTING_KR_TARGET_EXTRACTION_RECOMMENDED` | KR holds the source as `text_parity_extracted_review_needed`; specific lines/figures need target-extraction. |
| `EXTERNAL_ACQUISITION_REQUIRED` | The source is published but not in this corpus; acquire externally with the cited DOI/journal. |
| `ABSENT_FROM_LITERATURE` | The value appears unpublished (e.g., IPPLM internal drawings); only acquirable via facility request. |

## Executive Summary

**Sprint 4 status:** all P1–P5 priority closures committed (HEAD `da97ed2` before this audit pass; periodic audit 10/10 PASS at that HEAD). 21 distinct named blockers researched.

| Category | Count |
| ---: | --- |
| RESOLVED_VIA_KR | 2 (cathode-cage 200 mm radius, qualitative DPF anomalous resistivity) |
| KR_PROMOTION_RECOMMENDED (high-value, on-disk) | 3 (Bennett 2017, Braginskii 1965, Talebitaher PhD subset) |
| EXISTING_KR_TARGET_EXTRACTION_RECOMMENDED | 3 (UCSD/Beg b2e95b88:615-670; Plasma Focus Update Te/beam-target tables; Stepniewski 2004 review completion) |
| EXTERNAL_ACQUISITION_REQUIRED | 11 (named citations; see §6) |
| ABSENT_FROM_LITERATURE (likely unpublished) | 4 (PF-1000 insulator wall thickness, insulator outer radius, backplate radial extent, backplate axial thickness) |

**The single largest unblock-yield action:** promote `archive_reference_OLD/references/papers/core-dpf/schmidt-2017-kinetic-dpf-breakdown.pdf` (actually **Bennett et al. 2017, Phys. Plasmas 24, 062705** — the filename is misleading) into KR. This single promotion resolves STARTUP-BVP-CH04, STARTUP-BVP-CH08, and partially STARTUP-BVP-CH07 (3 of the 13 startup channels).

**The single largest research-direction finding:** Bernard 1977 contains the **first published direct Thomson-scattering Ti measurement of a DPF** (700 eV in the filament phase of a ~500 kA Limeil/Jülich Mather-type device). It does not map to PF-1000 but is historically significant and is the **only direct spectroscopic Ti measurement in the entire searched corpus**. The Te/Ti gap for PF-1000 specifically remains structural — confirmed against the 2021 Plasma Focus Update (220-page review) which itself notes the field has not published validated bulk-pinch Te via Thomson scattering or crystal spectrometry.

**Comparator matrix recommendation:** Option B from `docs/FIRST_PRINCIPLES_SCOPE_DECISION_MEMO_2026_05_20.md` (scope change to PF-1000 full-energy 27-40 kV) is unchanged after the Plasma Focus Update + Bernard 1977 KR ingest. Te/Ti remain TEXT-ONLY across the entire DPF corpus regardless of target device.

## 1. Geometry Blockers (PF-1000 hardware-scope)

| Blocker | Verdict | Evidence / Citation |
| --- | --- | --- |
| **PF1000-BLK-009** anode hollow-bore radius (Stepniewski 0.015 m simulation-parameter only) | partial `KR_PROMOTION_RECOMMENDED` + `EXTERNAL_ACQUISITION_REQUIRED` | Existence of bore: `KR: experimental-study-of-the-structure-of-the-plasma-current-sheath-on-the-pf-1000-facility-705bcc83.md:373` (Krauz 2012 hardware probe insertion at r=12 mm establishes lower bound). 15 mm value: `KR: doi-10-1016-j-vacuum-2004-05-019-f931cb0b.md:312-314` (Stepniewski 2004 simulation parameter, KR status `text_parity_extracted_review_needed`). Hardware-scope confirmation requires: **Miklaszewski et al. 2001 Nukleonika 46:S61-S64** or **Schmidt et al. 2002 Physica Scripta 66:168-172** (neither on disk). |
| **Cathode-cage radius conflict** (200 mm Krauz vs 160 mm Akel) | `RESOLVED_VIA_KR` — **three independent hardware sources confirm 200 mm**; 160 mm is definitively a Lee-fit parameter (category mismatch) | (1) `KR: experimental-study-...-pf-1000-facility-705bcc83.md:345-346` Krauz 2012 PCFP 54:025010 hardware measurement: "OE and copper center electrode (CE) radii are 200 mm and 115.5 mm". (2) `KR: gribkov-2007-pf1000-jphysd-part2.md:393-400` Gribkov 2007 J. Phys. D 40:1977: "12 stainless-steel rods... distributed around a 400 mm diameter circumference". (3) `KR: chunks/update-on-the-scientific-status-of-the-plasma-focus-1385adeb/pages-0026-0050.md:844-845` Plasma Focus Update 2021 §3.1.2. |
| **PF1000-BLK-015** insulator outer radius | `ABSENT_FROM_LITERATURE` | No paper in the corpus text-extracts the alumina sleeve outer diameter. Insulator length is documented (85 mm Krauz; 113 mm Gribkov/Scholz; 114 mm Update); outer radius bounded by anode radius 115.5 mm only. Acquisition: IPPLM facility request (Paduch, Miklaszewski) or ICDMP workshop reports. |
| **PF1000-BLK-016** insulator wall thickness | `ABSENT_FROM_LITERATURE` | No corpus source. PF-24 wall thickness 4 mm appears in Update §3.1.3 but is a different device. Acquisition path same as BLK-015. |
| **PF1000-BLK-017** backplate radial extent | `ABSENT_FROM_LITERATURE` | Krauz 2012:352 mentions "back plate of the OE" structurally but no dimension. Chamber outer diameter 1400 mm and OE on 400 mm diameter bound the geometry but do not specify the backplate flange. Acquisition: IPPLM facility request. |
| **PF1000-BLK-018** backplate axial thickness | `ABSENT_FROM_LITERATURE` | Same as BLK-017. Acquisition: IPPLM facility request. |

**Notable finding:** PF-1000 has had **multiple electrode configurations** over its operating history. The `final-stages-of-the-plasma-column-evolution-in-the-plasma-focus-pf1000-device-plasma-scien-fa128cfd.md` extract gives cathode diameter 368 mm (= 184 mm radius), anode 240 mm diameter / 450 mm active length, insulator 113 mm — distinct from the Krauz 2012 configuration. This multi-configuration variability must be documented in any future runtime that selects a constructor.

## 2. Startup BVP Blockers (13 channels)

### 2.1 D2 gas-physics channels (Townsend α, Paschen A/B, SEE γ, preionization)

| Blocker | Verdict | Evidence / Acquisition Target |
| --- | --- | --- |
| **STARTUP-BVP-CH02** D2 Townsend α(E/p) + Paschen A/B | `EXTERNAL_ACQUISITION_REQUIRED` | No tabulated D2 α(E/p) or Paschen A/B anywhere on disk. Bennett 2017 uses Monte Carlo σ(E) directly (not Paschen approximation). MSR PhD references Paschen qualitatively without A/B. Acquisition: **Raizer, *Gas Discharge Physics*, 2nd ed., Springer 1991** (Section 4 tables) OR **NIST LXCat Phelps/Morgan database** (H₂ ≈ D₂ by mass scaling). |
| **STARTUP-BVP-CH05** SEE γ for Cu/pyrex/alumina/SS | `EXTERNAL_ACQUISITION_REQUIRED` | NRL "secondary emission" is OML dust-charging (irrelevant). Zero KR hits for γ on relevant materials. Acquisition: **Hagstrum 1956 Phys. Rev. 104:317** (Cu metal SEE); **Vaughan formula** for ceramics; **CRC Handbook electron emission section**. |
| **STARTUP-BVP-CH03** preionization seed model | partial `KR_PROMOTION_RECOMMENDED` (Bennett 2017) + `EXTERNAL_ACQUISITION_REQUIRED` (cosmic-ray rate) | Bennett 2017 (on disk, not in KR) gives **seed density 10⁷ cm⁻³ at 5.5 Torr** directly (§II) — sufficient for DPF startup BVP without separate cosmic-ray rate. Also confirms photoionization negligible (1.2% by 125 ns). |
| **STARTUP-BVP-CH10** D2 ionization/recombination | partial `RESOLVED_VIA_KR` (atomic Lotz/NRL only) + `EXTERNAL_ACQUISITION_REQUIRED` (D₂ molecular) | KR Lotz 1967 (`an-empirical-formula-for-the-electron-impact-ionization-cross-section-b5fde85c.md`) covers H I (atomic; applicable to D by isotopic identity). NRL Formulary lines 4589-4630 has hydrogenic S(Z), αr, α₃. Missing: D₂ molecular dissociation/ionization. Acquisition: **Voronov 1997 ADNDT 65:1** (DOI: 10.1006/adnd.1997.0732); **Janev-Smith 1993 IAEA-TECDOC-697**; **Open ADAS adf11** files. |

### 2.2 Sheath / flashover / initial-fields channels

| Blocker | Verdict | Evidence / Citation |
| --- | --- | --- |
| **STARTUP-BVP-CH04** flashover closure (delay / voltage / striation timescale) | `KR_PROMOTION_RECOMMENDED` | `archive_reference_OLD/references/papers/core-dpf/schmidt-2017-kinetic-dpf-breakdown.pdf` — **filename mislabel: this is actually Bennett et al. 2017, Phys. Plasmas 24, 062705**. Sections I-III contain the full flashover closure: three pressure-regime taxonomy (low p volumetric, medium p surface flashover, high p radial filament), ionization mean free path criterion λ_ioniz ~ L_i, breakdown delay ~20 ns, Coulomb collision Eq. 1, plasma density 10¹⁵ cm⁻³ along insulator by 100 ns. Priority: HIGH. |
| **STARTUP-BVP-CH07** surface plasma E-field set | partial `EXISTING_KR_TARGET_EXTRACTION_RECOMMENDED` + `KR_PROMOTION_RECOMMENDED` | Existing: `KR: effect-of-current-sheath-initiation-...-b2e95b88.md:615-660` — extract three pressure-regime boundaries (0.75 / 3.75 Torr), Te ~4 eV during breakdown, L_iz/L_i = 2.4 criterion. Promotion: Bennett 2017 (same as CH04) shows breakdown-phase E-field spatial distribution + space-charge limited emission 250 kV/cm (bulk) / 10 kV/cm (knife-edge). |
| **STARTUP-BVP-CH08** initial E/J distributions | `KR_PROMOTION_RECOMMENDED` (Bennett 2017) | Same source as CH04. Bennett 2017 §III provides ion density contours at t = 100 ns / 500 ns / 1 µs (Fig. 4), T_e ~3.5-4 eV along insulator (Fig. 7), 71% current fraction in sheath by 500 ns. Sole PIC source for DPF breakdown-phase E/J in the corpus. |
| **STARTUP-BVP-CH12** sheath lift-off / current-sheath formation | `EXISTING_KR_TARGET_EXTRACTION_RECOMMENDED` | `KR: effect-of-current-sheath-initiation-...-b2e95b88.md:615-670` contains DPF-generic mass-sweep factor **massf ≈ 0.4·p₀^(-1/2)** (Fig. 10 of source) — the empirical detachment criterion. Scope caveat: derived on UCSD 4.6 kJ Mather; needs explicit `wrong_scope_method_context` annotation on the closure. |
| **STARTUP-BVP-CH06** photoemission | `EXTERNAL_ACQUISITION_REQUIRED` | Zero hits across KR + on-disk for DPF insulator photoemission. Bennett 2017 explicitly rules photoemission as 1.2% effect and neglects it. Acquisition: **Yordanov et al. 2003 Vacuum 76:365** (Bennett 2017 Ref. [21]) — most likely source for surface photoemission quantification. |

**Action for the audit team — verify by promoting Bennett 2017:** the filename `schmidt-2017-kinetic-dpf-breakdown.pdf` belongs to Bennett et al. 2017, Phys. Plasmas 24, 062705. A targeted KR promotion + target extraction of §II-III resolves 3 of 13 startup channels (CH04, CH08, partial CH07) and gives an additional preionization datum (CH03). Recommend renaming the on-disk file to `bennett-2017-kinetic-dpf-breakdown.pdf` before promotion.

## 3. Transport / Closures Blockers

### 3.1 Direct Braginskii + D2 transport

| Blocker | Verdict | Evidence / Citation |
| --- | --- | --- |
| **CLOSURE-BLK-BRAG-001** Braginskii 1965 Z-dependent coefficients | `KR_PROMOTION_RECOMMENDED` | `archive_reference_OLD/references/papers/mhd-numerics/braginskii_1965.pdf` exists on disk. Confirmed: **Table 2 (p. 251)** has Z = 1, 2, 3, 4, ∞ columns with full α/β/γ/δ coefficient families (resistivity ∥/⊥/∧, thermo-electric β, heat conduction γ, ion/electron viscosity); **Eqs. 4.30-4.45** (pp. 249-253) bind table entries to transport closures. Suggested KR slug: `braginskii-1965-transport-processes-plasma`. Provenance form: `[KR: braginskii-1965-transport-processes-plasma.md §Table2 p.251]`. Priority: HIGH. |
| **CLOSURE-BLK-D2-EN-001** D2 electron-neutral momentum-transfer cross-section | `EXTERNAL_ACQUISITION_REQUIRED` | No D2 cross-section data in any KR or on-disk plasma-physics/textbooks. Acquisition: **Itikawa & Mason 2005 JPCRD 34:1** (open access via NIST JPCRD; DOI: 10.1063/1.1799251). Alternate: **LXCat Biagi or BSR database**. |
| **CLOSURE-BLK-ION-001** D2 ionization/recombination | `EXTERNAL_ACQUISITION_REQUIRED` | KR Lotz 1967 covers atomic H I (D-applicable). Missing: D₂ molecular. Acquisition: **Janev-Smith 1993 IAEA-TECDOC-697**; **Janev et al. 1987 "Elementary Processes in Hydrogen-Helium Plasmas" Springer**; **Open ADAS** charge-exchange + recombination rates. |

### 3.2 Anomalous resistivity + restrike

| Blocker | Verdict | Evidence / Citation |
| --- | --- | --- |
| **CLOSURE-BLK-ANOM-001** DPF-regime anomalous resistivity | partial `RESOLVED_VIA_KR` (qualitative DPF-LHDI) + `EXTERNAL_ACQUISITION_REQUIRED` (quantitative η*) | **DPF qualitative confirmed in KR**: `comparisons-of-dense-plasma-focus-kinetic-simulations-with-experimental-measurements.md` — LLNL fully-kinetic simulations show η_anomalous ~ 3 orders of magnitude above Spitzer (56 mΩ-cm vs ~20 µΩ-cm), with 4.6-18 GHz EM fluctuations matching the lower-hybrid band for B = 10-40 T. Direct quote: *"anomalous resistivity in the plasma arises due to a kinetic instability near the lower hybrid frequency."* Quantitative η* formula acquisition: **Davidson & Gladd 1975 Phys. Fluids 18:1327** (DOI: 10.1063/1.861021); **Bruzzone & Bernal 2001 Nukleonika 46:59-61**; **Bruzzone 2001 Nukleonika 46:S3-S7** (cited Ref. [34] in Auluck 2022 Theory Part 1 — Auluck cites it but does not develop the formula in-text). |
| **CLOSURE-BLK-REST-001** DPF restrike governing equation | `ABSENT_FROM_LITERATURE` (no equation exists anywhere) | KR + Auluck 8-paper series + Plasma Focus Update searched: restrike appears as **phenomenological circuit fit** (Lee/Saw, Beresnyak 2022 "*restrikes result in a sudden decrease in inductance by introducing a return-current path in parallel with the load*") but **no governing differential equation or first-principles physics model exists in any source**. The LLNL kinetic simulation paper explicitly identifies this as an open gap: *"Future work will address the fidelity of the simulations after the pinch, so that post-pinch features such as restrikes and multiple pinches can be resolved."* Closest available: **Barbaglia et al. 2010 Plasma Phys. Control. Fusion 52:032001** ("Multiple pinch formations in small plasma-focus devices") — multiple-pinch ≠ restrike strictly but is the nearest characterized phenomenon. Acquisition: Barbaglia 2010 (best available) or Sagdeev-Galeev 1969 *Nonlinear Plasma Theory* (foundational anomalous transport, generic). |

**Notable finding (Auluck 8-paper survey):** Of the eight Auluck 2021-2024 papers on disk, **none** develops a quantitative anomalous resistivity formula or restrike equation in-text. They cite the relevant external papers (Davidson-Gladd, Bruzzone) but extend them in different directions (poloidal flux, filamentation, propulsion). The Auluck corpus is therefore **not** the path to quantitative DPF transport closure — external acquisition is required.

## 4. Neutron Mechanism Blockers

| Blocker | Verdict | Evidence / Citation |
| --- | --- | --- |
| **NEUTRON-BLK-001** fast-ion distribution function | `EXTERNAL_ACQUISITION_REQUIRED` (with partial `KR_PROMOTION_RECOMMENDED` for NX2-scope) | Talebitaher PhD p.103: *"the real deuteron angular and energy distributions within the pinch column are not known"* and uses mono-energetic 30° cone for NX2 only. Promote Talebitaher pp.100-119 for NX2-specific cone model (eq. 3-5 differential cross-section Legendre polynomial; Table 8 NX2 anisotropy values). Acquisition for PF-1000: **Gribkov et al. 2007 J. Phys. D 40:3592**; **Schmidt et al. 2012 PRL 109:205003** (fully kinetic PIC). |
| **NEUTRON-BLK-002** deuteron stopping power | `EXTERNAL_ACQUISITION_REQUIRED` | AD1079881 (Schmidt LLNL) uses "cold matter ion stopping powers" without tabulated values. Acquisition: **ICRU Report 49 (1993)** "Stopping Powers and Ranges for Protons and Alpha Particles" (deuteron interpolation); **Andersen-Ziegler 1977** "Stopping and Range of Ions in Matter" Pergamon; **SRIM-2013** tabulations. |
| **NEUTRON-BLK-003** beam-target yield | dependency-blocked on 001 + 002 | Y_bt = ∫ f(E) · [dE/dx]⁻¹ · σ(E) dE · n_target. σ(E) is RESOLVED via Bosch-Hale KR. Resolves automatically once 001 + 002 acquired. |
| **NEUTRON-BLK-004** Brysk 1973 Doppler broadening | `EXTERNAL_ACQUISITION_REQUIRED` | Confirmed absent from KR + disk + downloaded_books_papers. Acquisition: **Brysk, H. 1973 Plasma Phys. 15:1282** — FWHM = 82.5 √(Ti[keV]) keV. |
| **NEUTRON-BLK-005** same-scope angular anisotropy (Akel 16 kV) | `EXTERNAL_ACQUISITION_REQUIRED` | No measurement at 16 kV / 170 kJ scope exists in corpus (Krasa 450-500 kJ; Talebitaher NX2 1.6 kJ; Plasma Focus Update has no 16 kV anisotropy entry for PF-1000). Acquisition: search Nukleonika / J. Fusion Energy / Phys. Plasmas for Akel-authored PF-1000 16 kV anisotropy papers post-2010. |
| **Thermonuclear 1/4 volumetric prefactor** | `EXTERNAL_ACQUISITION_REQUIRED` | Bosch-Hale KR covers ⟨σv⟩ only; the n_d² / 4 identical-particle pair-counting derivation is not stated verbatim in any KR file. Acquisition: **Glasstone & Lovberg 1960 "Controlled Thermonuclear Reactions" Ch. 1**; or **McNally 1979 ORNL/TM-6914**. |

**Notable Bernard 1977 finding (historical baseline):** `KR: the-dense-plasma-focus-a-high-intensity-neutron-source-f0a3910d.md` (just ingested) contains:
- **Direct Thomson-scattering Ti = 700 eV** during filament phase on the ~500 kA Limeil/Jülich Mather-type device (lines 455-461; method described 977-1033). **First direct spectroscopic Ti for any DPF in the corpus**.
- I(t) + V(t) waveforms for the Frascati 1 MJ machine at 20/25/33 kV (Fig. 21, lines 1546-1547).
- 3-direction neutron TOF at 0°, 45°, 90° same discharge (Fig. 17, lines 1185-1193) — early anisotropy.
- HXR waveform + spectrum to 350 keV (Fig. 14, lines 793-796).
- Holographic interferometry density maps (lines 963-968).

This does not map to PF-1000 but is historically significant. If the comparator scope is ever broadened to historical devices, Bernard 1977 supplies the only direct spectroscopic Ti measurement in the searched corpus.

## 5. Same-Scope Comparator Decision — Confirmation After New KR

The 2021 Plasma Focus Update (220 pages) and Bernard 1977 (28 pages) were re-surveyed against the existing comparator matrix in `docs/FIRST_PRINCIPLES_SCOPE_DECISION_MEMO_2026_05_20.md`.

**Result: no cell flips. Option B (PF-1000 full-energy 27-40 kV) is reaffirmed.**

- Akel 16 kV scope: **no new SUPPORTED cells**. Plasma Focus Update mentions PF-1000 at 16 kV / 300 kJ only for column-breakup imaging, not waveforms or temperatures.
- Full-energy 27-40 kV scope: **one candidate** for Te (Zaloga 2018 J. Phys. Conf. Ser. 959:012003 filter-ratio at PF-1000U: 7.5 keV local hotspot, cited in Update p.~37-40). Three caveats keep it TEXT-ONLY:
  1. KR status `text_parity_extracted_review_needed` — figure-caption value not yet target-extracted.
  2. Filter-ratio Te is method-dependent (assumes optically thin bremsstrahlung).
  3. Plasma Focus Update itself flags a separate filter-ratio result as possibly mis-evaluated.
- **Te/Ti gap is structural** across the entire 2021 review of the field. No validated Thomson-scattering or crystal-spectrometry pinch-phase Te for PF-1000 at any voltage through at least 2021.

## 6. External Acquisition List (Prioritized for the Audit Team)

| Priority | Citation | Resolves |
| --- | --- | --- |
| **P1** | **Raizer, Yu. P. (1991).** *Gas Discharge Physics*, 2nd ed. Springer. §4 | STARTUP-BVP-CH02 (D2 Townsend α + Paschen A/B) |
| **P1** | **NIST LXCat database** (lxcat.net, Phelps/Morgan/Biagi datasets) | STARTUP-BVP-CH02, CLOSURE-BLK-D2-EN-001 |
| **P1** | **Brysk, H. (1973).** *Plasma Phys.* 15:1282 | NEUTRON-BLK-004 (Doppler broadening) |
| **P1** | **ICRU Report 49 (1993).** "Stopping Powers and Ranges for Protons and Alpha Particles." Bethesda, MD | NEUTRON-BLK-002 (deuteron stopping) |
| **P1** | **Davidson, R.C.; Gladd, N.T. (1975).** *Phys. Fluids* 18:1327. DOI: 10.1063/1.861021 | CLOSURE-BLK-ANOM-001 (quantitative LHDI η*) |
| **P1** | **Bruzzone, H.; Bernal, L. (2001).** *Nukleonika* 46:59-61; **Bruzzone, H. (2001).** *Nukleonika* 46:S3-S7 | CLOSURE-BLK-ANOM-001 (DPF-scope) |
| **P2** | **Voronov, G.S. (1997).** *At. Data Nucl. Data Tables* 65:1. DOI: 10.1006/adnd.1997.0732 | CLOSURE-BLK-ION-001 (ionization rates) |
| **P2** | **Janev, R.K.; Smith, J.J. (1993).** IAEA-TECDOC-697 | CLOSURE-BLK-ION-001 (ionization + recombination) |
| **P2** | **Itikawa, Y.; Mason, N. (2005).** *J. Phys. Chem. Ref. Data* 34:1. DOI: 10.1063/1.1799251 | CLOSURE-BLK-D2-EN-001 (D2 cross sections) |
| **P2** | **Gribkov et al. (2007).** *J. Phys. D: Appl. Phys.* 40:3592 | NEUTRON-BLK-001 (PF-1000 fast-ion distribution) |
| **P2** | **Schmidt et al. (2012).** *Phys. Rev. Lett.* 109:205003 | NEUTRON-BLK-001 (kinetic PIC benchmark) |
| **P2** | **Hagstrum, H.D. (1956).** *Phys. Rev.* 104:317 | STARTUP-BVP-CH05 (SEE γ Cu) |
| **P2** | **Miklaszewski et al. (2001).** *Nukleonika* 46:S61-S64; **Schmidt et al. (2002).** *Physica Scripta* 66:168-172 | PF1000-BLK-009 (hardware-scope hollow bore confirmation) |
| **P3** | **Barbaglia et al. (2010).** *Plasma Phys. Control. Fusion* 52:032001 | CLOSURE-BLK-REST-001 (multiple-pinch as restrike proxy) |
| **P3** | **Janev et al. (1987).** *Elementary Processes in Hydrogen-Helium Plasmas*, Springer | CLOSURE-BLK-ION-001 |
| **P3** | **Yordanov et al. (2003).** *Vacuum* 76:365 | STARTUP-BVP-CH06 (photoemission) |
| **P3** | **Glasstone & Lovberg (1960).** *Controlled Thermonuclear Reactions*, Ch. 1 | Thermonuclear 1/4 prefactor |
| **P3** | **Bernard et al. (1977).** *Nucl. Instrum. Methods* 145:191 | NEUTRON-BLK-001 (historical) |
| **P3** | **Sagdeev & Galeev (1969).** *Nonlinear Plasma Theory*, Benjamin | CLOSURE-BLK-ANOM-001 / -REST-001 (foundational) |
| **P4** | **IPPLM facility engineering drawings** (request via M. Paduch / R. Miklaszewski; or IAEA CRP 11940/11941; or ICDMP workshop proceedings) | PF1000-BLK-015, -016, -017, -018 (likely unpublished elsewhere) |

## 7. On-Disk KR Promotions the Audit Team Should Execute

These resolve blockers with **no external acquisition** — the PDF is already in the corpus.

| Action | Source | Resolves |
| --- | --- | --- |
| **Promote and target-extract** `archive_reference_OLD/references/papers/core-dpf/schmidt-2017-kinetic-dpf-breakdown.pdf` (rename to `bennett-2017-kinetic-dpf-breakdown.pdf` first — **this is Bennett et al. 2017 Phys. Plasmas 24, 062705**, not a Schmidt paper) | Bennett 2017 §I-III | STARTUP-BVP-CH04, CH08, partial CH07; preionization seed datum for CH03 |
| **Promote and target-extract** `archive_reference_OLD/references/papers/mhd-numerics/braginskii_1965.pdf` Table 2 (p.251) + Eqs. 4.30-4.45 (pp.249-253) | Braginskii 1965 | CLOSURE-BLK-BRAG-001 (Z-dependent transport coefficients) |
| **Promote subset** `downloaded_books_papers/Research Papers/2026-05-16-user-validated-theses/PhD2012AlirezaTalebitaher.pdf:pp.100-119` (NX2-scope cone model; eqs. 3-5 to 3-8; Table 8 anisotropy) — explicitly labeled `wrong_scope_NX2_not_pf1000` in any runtime use | Talebitaher 2012 PhD | NEUTRON-BLK-001 partial (NX2 scope only) |

## 8. Existing KR Target Extractions the Audit Team Should Execute

These KR files already exist as `text_parity_extracted_review_needed`; specific lines need target extraction.

| KR file | Lines | Targets |
| --- | --- | --- |
| `effect-of-current-sheath-initiation-...-b2e95b88.md` | 615-670 | mass-sweep factor **massf = 0.4 · p₀^(-1/2)** (Fig. 10); Paschen-regime boundaries (0.75 / 3.75 Torr); L_iz/L_i = 2.4 criterion; Te ≈ 4 eV breakdown temperature → resolves part of CH07 + CH12 |
| `chunks/update-on-the-scientific-status-of-the-plasma-focus-1385adeb/pages-0026-0050.md` | 512-517 + 822-850 | Te = 7.5 keV filter-ratio (Zaloga 2018 — caveat method-dependent); PF-1000 §3.1.2 geometry summary |
| `chunks/update-on-the-scientific-status-of-the-plasma-focus-1385adeb/pages-0126-0150.md` | beam-target figures + tables | Beam-target mechanism summary (320 keV mean, 500 keV max deuteron energies referenced) |
| `doi-10-1016-j-vacuum-2004-05-019-f931cb0b.md` | 306-314 | Stepniewski 0.015 m hollow bore — needs HARDWARE-SCOPE review verdict (currently sim-parameter only) |
| `the-dense-plasma-focus-a-high-intensity-neutron-source-f0a3910d.md` (Bernard 1977) | 455-461, 977-1033 | **Thomson Ti = 700 eV** measurement (historical, ~500 kA Mather, filament phase) — first direct spectroscopic Ti in corpus |

## 9. What the Audit Team Should Verify

For each row in §§ 1-4, the audit team should:

1. **Read the cited KR file at the cited lines** — confirm the content matches the claim.
2. **For `RESOLVED_VIA_KR` entries** — confirm the scope matches (e.g., cathode-cage 200 mm has three hardware-scope citations, all from PF-1000).
3. **For `KR_PROMOTION_RECOMMENDED` entries** — verify the PDF is on disk at the cited path; spot-check page contents match the verdict.
4. **For `EXISTING_KR_TARGET_EXTRACTION_RECOMMENDED` entries** — confirm the KR file is currently `text_parity_extracted_review_needed` and the cited lines contain the named targets.
5. **For `EXTERNAL_ACQUISITION_REQUIRED` entries** — confirm the cited DOI/journal/volume/page is real and the paper is genuinely absent from disk (`find / -iname '<author><year>*' -type f 2>/dev/null`).
6. **For `ABSENT_FROM_LITERATURE` entries** — confirm via independent KR + disk + DB search that no publication contains the value at the required scope.

## 10. Acceptance Boundary (Sprint 4 → Sprint 5)

Sprint 5 may proceed if and only if the audit team:

- Approves the comparator scope decision (Option B: PF-1000 full-energy 27-40 kV) or proposes an alternative scope with a documented matrix.
- Triages the 11 external acquisitions into MUST (block Sprint 5), SHOULD (parallel acquisition), and DEFER (Sprint 6+).
- Executes (or schedules) the three high-value on-disk KR promotions in §7.

No physics is asserted by this handoff. All runtime `can_support_first_principles_acceptance` flags remain `false`. No validation certificate is implied or unlocked by these findings.

---

**Prepared by:** automated 7-agent research fan-out + lead synthesis, 2026-05-20.
**Verifiable basis:** every claim in this document is backed by either a KR `[path:lines]` citation, an on-disk PDF path, or a full external citation (author, year, source, vol:page, DOI).
