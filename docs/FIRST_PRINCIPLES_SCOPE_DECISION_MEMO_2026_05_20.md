# First-Principles DPF Same-Scope Comparator Decision Memo (2026-05-20)

## Context

The current V&V certificate target is **PF-1000 at Akel 16 kV**, a reduced-energy operating point
(~170 kJ stored, I_peak ~1 MA, D2 at modest pressure). Akel (2021) published I(t) with pinch-dip and
Yn at this operating point. All other PF-1000 diagnostic data in this corpus — optical spectroscopy,
interferometric density, HXR/SXR waveforms, neutron TOF spectrum, and anisotropy — are documented
exclusively at 27–40 kV (~1 MA and above, Gribkov/Scholz/Krauz/Malir era). The spectroscopy paper
(Skladnik-Sadowska et al. 2011, KR: `optical-spectroscopy...md`) explicitly states the PF-1000 operated
at **21–27 kV, 290–480 kJ** during those measurements; it is free-stream spectroscopy (ne from Stark
broadening at z = 30 cm after the pinch), not pinch-phase Te/Ti. No paper in this corpus reports
pin-phase Te, Ti, ne, X-ray yield, neutron spectrum, or angular anisotropy for PF-1000 at 16 kV.

**Decision required:** (A) acquire same-scope Akel 16 kV data for the missing seven channels, or
(B) propose a different demonstrator that has higher channel coverage at its published operating point.

---

## Comparator Matrix

| Device / Shot | I(t) | V(t) | Te | Ti | ne | X-ray | Yn | Spectrum | Anisotropy |
|---|---|---|---|---|---|---|---|---|---|
| **PF-1000 Akel 16 kV** | SUPPORTED `KR: characteristics-of-closed-currents...md` (Lee-model fit to Akel 2021 I_peak);<br>on-disk: `akel-2021-pf1000-neutron-yield.pdf` | ABSENT | ABSENT | ABSENT | ABSENT | ABSENT | SUPPORTED `akel-2021-pf1000-neutron-yield.pdf` (Yn ~3×10^10) | ABSENT | ABSENT |
| **PF-1000 full-energy 27–40 kV (Gribkov/Scholz 2007)** | SUPPORTED `KR: scholz-2007-pf1000-part2-jphysd.md` pp.200–230 (I(t) at 27 kV, I_min ~1.85 MA) | SUPPORTED `KR: scholz-2007-pf1000-part2-jphysd.md` pp.200–230 (V(t) waveform at 27 kV, Fig. 2(b)) | TEXT-ONLY `KR: scholz-2007-pf1000-part2-jphysd.md` (pinch target ≤1 keV cited, no direct Te measurement) | TEXT-ONLY `KR: scholz-2007-pf1000-part2-jphysd.md` (≤1 keV plasma, inferred from model not spectroscopy) | SUPPORTED `KR: malir-2024-interferometry-dpf.md` (16-frame Mach-Zehnder interferometry, electron density maps for 2 shots) | SUPPORTED `KR: scholz-2006-pf1000-mega-joule.md` pp.1–6 (SXR PIN diode >4 keV, HXR 100 keV; Gribkov 2007 Part I: SXR-PMT 3–8 keV, HXR 8–30 keV, PIN 0.8–4 keV) | SUPPORTED `KR: scholz-2006-pf1000-mega-joule.md` (Yn ~10^10–10^11; best ~3.5×10^11) | SUPPORTED `KR: fusion-neutron-detector-for-tof...md` (TOF detector calibrated at PF-1000, 2.45 MeV at 1 MA) | SUPPORTED `KR: scholz-2007-pf1000-part2-jphysd.md` pp.380–460 (5 silver-activation counters at 0°, 30°, 60°, 90°, 150°; Y0°/Y90° ~1.8) |
| **PF-1000 Krasa 2008 (450–500 kJ, 3.5 Torr D2)** | TEXT-ONLY `KR: anisotropy-of-the-emission-of-dd-fusion-neutrons...md` p.2 (mentions shots, no waveform) | ABSENT | ABSENT | ABSENT | ABSENT | ABSENT | TEXT-ONLY `KR: anisotropy-of-the-emission-of-dd-fusion-neutrons...md` p.2 (Yn ≈3.5×10^11 quoted as maximum) | TEXT-ONLY `KR: anisotropy-of-the-emission-of-dd-fusion-neutrons...md` pp.3–4 (TOF energy-group spectra discussed; direct digitized values absent from KR extract) | SUPPORTED `KR: anisotropy-of-the-emission-of-dd-fusion-neutrons...md` pp.2–5 (TLD/Bonner-sphere multi-position measurement; vessel-scatter MCNP computation) |
| **MJOLNIR (LLNL, 2 MJ, 3.8 MA class)** | SUPPORTED `KR: neutron-generation-dynamics-inside-a-ma-class-dense-plasma-focus-z-pinch-5.md` p.3 (Rogowski coil, head-current trace at 0.75 MJ) | SUPPORTED (same KR, p.3 — "Rogowski coils to measure voltage between anode and cathode" cited as additional diagnostics; Fig. 2 shows current including current-dip region) | TEXT-ONLY `KR: neutron-generation-dynamics-inside-a-ma-class-dense-plasma-focus-z-pinch-5.md` pp.6–7 (T_st estimated from 1D shock theory, ~21 keV predicted; not measured spectroscopically) | TEXT-ONLY (same, shock-model estimate only) | TEXT-ONLY (same; "fiber-coupled photodiodes, step-wedge filter" mentioned; no published ne profile) | SUPPORTED `KR: neutron-generation-dynamics-inside-a-ma-class-dense-plasma-focus-z-pinch-5.md` p.3 (x-ray step-wedge filter; scintillator-PMT for HXR/SXR; Fig. 6 X-ray peaks timing) | SUPPORTED (same; Yn up to 8×10^11 with Be activation, absolutely calibrated at Sandia) | SUPPORTED (same; TOF at 2.2 m and 6.6 m; Fig. 7 neutron energy spectrum from MHD-kinetic simulation with qualitative experimental confirmation) | SUPPORTED (same pp. 595–614; LaBr at 10° and 70°; on-axis/off-axis ratio up to 1.8×) |
| **Faeton-I (100 kV, ~750 kJ, Iran/Malaysia 2025)** | SUPPORTED `KR: faeton-i-investigation-of-plasma-dynamics-and-radiation-output-of-a-100-kv-plasma-focus-device.md` (Lee-code fit to fcr, I_peak reported per shot; Table 3) | SUPPORTED (same KR; V_peak spike to 194 kV measured per shot; dynamics-induced pre-stagnation voltage waveform) | ABSENT | ABSENT | ABSENT | TEXT-ONLY (same KR p.1; "PMT-scintillators detected gamma photons above 3 MeV at 40 m" — qualitative only) | SUPPORTED (same; Yn up to 8×10^10 per shot; 2.5×10^10 typical) | SUPPORTED (same; PMT-scintillator TOF at 40 m; neutron energy (2.5±0.3) MeV extracted) | TEXT-ONLY (same; anisotropy factor 1.6 forward/on-axis cited; no multi-position quantitative table) |
| **NX2 (NTU Singapore, 3 kJ, 300 kA)** | TEXT-ONLY `KR: the-dense-plasma-focus-a-versatile-dense-pinch-for-diverse-applications.md` lines 1159–1165 (mentioned as optimized SXR source; no waveform in KR) | ABSENT | ABSENT | ABSENT | ABSENT | SUPPORTED (same review, lines 1159–1165; SXR 140 J into 4π in Ne; Ar K-shell SXR characterized) | ABSENT (not a deuterium-optimized neutron device; D2 neutrons are incidental) | ABSENT | ABSENT |
| **LPP-FF1 / FF-2B (LPPFusion, ~0.6 kJ pB11 device)** | TEXT-ONLY `KR: focus-fusion-overview-of-progress-towards-p-b11-fusion-with-the-dense-plasma-focus.md` p.8 (device described; no I(t) waveform published) | ABSENT | TEXT-ONLY (same KR; "confined ion energies >200 keV" claimed; no independent spectroscopic Te) | ABSENT | ABSENT | TEXT-ONLY (same KR; "highest wall-plug efficiency" claimed; no published absolute X-ray yield with calibration) | ABSENT (p-B11 fuel; DD neutron yield absent) | ABSENT | ABSENT |
| **Bayesian/spectroscopy device (CCHEN Chile, 400 J, H2)** | SUPPORTED `KR: bayesian-inference-of-spectrometric-data-and-validation-with-numerical-simulations-of-plas-9ff01860.md` p.2 (current-sheath electrical signals shown) | SUPPORTED (same KR p.2 — breakdown voltage peak visible in waveform) | SUPPORTED (same KR; T_e = 4–20 eV in rundown phase from Stark broadening of H-alpha) | ABSENT (rundown only; no pinch-phase Ti) | SUPPORTED (same; electron density from Stark broadening, rundown phase ne) | ABSENT | ABSENT | ABSENT | ABSENT |
| **PF-1000U / ICDMP upgrades (2020–2025)** | ABSENT (no separate KR extract for PF-1000U upgrade shots; Malir 2024 uses standard PF-1000 at ~23 kV, 350 kJ) | ABSENT | ABSENT | ABSENT | SUPPORTED `KR: malir-2024-interferometry-dpf.md` (16-frame interferometry; electron density maps for shots 13 317 and 13 328 at ~23 kV, 350 kJ) | ABSENT | ABSENT | ABSENT | ABSENT |

---

## Channel-by-channel gap analysis

**I(t) — discharge current waveform with pinch-dip.** Available at 16 kV (Akel 2021, DB record 273) and at all full-energy PF-1000 shots (Gribkov/Scholz 2007, Malir 2024, Krauz 2012). MJOLNIR and Faeton-I both publish I(t). Gap is specific to Akel 16 kV where no voltage waveform or diagnostic suite accompanies the published current data.

**V(t) — tube voltage waveform.** Published at PF-1000 full-energy (Gribkov/Scholz 2007, Fig. 2(b) at 27 kV). MJOLNIR confirms a V(t) channel exists; Faeton-I has detailed pre-stagnation voltage spikes. Genuinely absent for Akel 16 kV — Akel 2021 is a Lee-model fit paper and does not present V(t) data.

**Te — electron temperature.** No time-resolved experimental Te exists in this corpus for any PF-1000 operating point; Gribkov/Scholz cite the pinch as "≤1 keV" from model estimates. MJOLNIR reports T_st from 1D shock theory (~21 keV predicted, not measured by spectroscopy). Faeton-I and Bayesian/CCHEN devices provide spectroscopic T_e only in the rundown phase (4–20 eV) — not pinch-phase Te. This channel is ABSENT corpus-wide for the pinch phase.

**Ti — ion temperature.** No independent Ti measurement exists in any paper in this corpus. All "temperatures" cited for the DPF pinch are model-derived or inferred indirectly from neutron spectra. MJOLNIR's shock-theory estimate merges Ti and Te. ABSENT corpus-wide from direct measurement.

**ne — electron number density.** Best coverage is Malir 2024 (16-frame laser interferometry, electron density maps, 2 shots at PF-1000 ~23 kV 350 kJ) and Kubes et al. 2009/2012 (16-beam interferometry at PF-1000 1 MA). Stark broadening at CCHEN 400 J device. ABSENT at Akel 16 kV specifically.

**X-ray output.** Best coverage at PF-1000 full-energy: SXR (PIN diode, PMT) and HXR (100 keV) waveforms with timing (Gribkov/Scholz 2007; Scholz 2006). MJOLNIR has X-ray step-wedge + scintillator-PMT timing; Faeton-I has gamma above 3 MeV at 40 m (qualitative). SXR yield, spectral energy, and absolute calibration are TEXT-ONLY or absent at 16 kV.

**Neutron yield Yn.** Covered: Akel 16 kV (Yn ~3×10^10), all full-energy PF-1000 (~10^11), MJOLNIR (~8×10^11), Faeton-I (up to 8×10^10). Best absolute calibration is MJOLNIR (Be activation, Sandia-calibrated) and Gribkov/Scholz (silver + indium + bubble detectors).

**Neutron energy spectrum.** TOF detector calibrated and tested at PF-1000 (Klir 2011). MJOLNIR provides time-resolved spectrum from simulation with experimental confirmation. Faeton-I extracts (2.5±0.3) MeV from PMT-scintillator TOF. PF-1000 Krasa 2008 discusses TOF energy groups qualitatively but digitized values are not target-extracted in KR. Absent at 16 kV.

**Neutron angular anisotropy.** Best coverage: PF-1000 full-energy (Gribkov/Scholz 2007: 5 silver-activation counters at 0°, 30°, 60°, 90°, 150°; Y0°/Y90° ~1.8); Krasa 2008 (TLD/Bonner-sphere 4-position measurement + MCNP scatter computation); MJOLNIR (LaBr at 10°, 45°, 70°; ratio up to 1.8); Faeton-I (factor 1.6 forward, qualitative citation). ABSENT at 16 kV.

---

## Option A — Acquire same-scope Akel 16 kV data

**What specifically must be acquired:**
- V(t) tube voltage waveform at 16 kV with pinch-dip (not available from Akel 2021)
- Pinch-phase Te from Thomson scattering or crystal spectrometry at 16 kV
- Pinch-phase Ti from neutron spectroscopy (TOF broadening) at 16 kV
- ne from laser interferometry (Mach-Zehnder or multi-frame) at 16 kV
- SXR/HXR absolute yield with PIN diode + calibrated scintillator at 16 kV
- Neutron energy spectrum (TOF) at 16 kV
- Neutron angular anisotropy (≥3 activation counters) at 16 kV

**Feasibility assessment:** PF-1000 is operated by IPPLM Warsaw under the ICDMP (International Centre for Dense Magnetized Plasmas) framework. The facility regularly runs collaborative campaigns and has published at lower voltage (Cikhardtova 2015 at 23 kV, 350 kJ). Running at 16 kV is technically feasible but would require a dedicated low-energy campaign — 16 kV is well below the facility's nominal operating range of 27–40 kV, and the diagnostic suite (interferometer trigger, neutron TOF distances, scintillator timing) would need reoptimization for the lower signal levels (Yn ~3×10^10 vs ~10^11). This is not a shot-request task; it requires a 3–6 month campaign proposal submitted to IPPLM/ICDMP. It cannot be satisfied from the existing corpus.

---

## Option B — Propose alternate demonstrator

The **PF-1000 at full-energy 27–40 kV (Gribkov/Scholz 2007 era, ~810 kJ, ~2 MA)** is the strongest alternate demonstrator in this corpus.

**Channel coverage:**
- I(t): SUPPORTED (Rogowski coil, waveform published)
- V(t): SUPPORTED (oscilloscope trace published at 27 kV)
- Te: TEXT-ONLY (model estimate ≤1 keV; no direct spectroscopic measurement in corpus)
- Ti: TEXT-ONLY (same caveat)
- ne: SUPPORTED (Malir 2024 16-frame interferometry; Kubes 2009/2012 interferometry)
- X-ray: SUPPORTED (SXR + HXR waveforms, PIN + PMT, multiple energy bands)
- Yn: SUPPORTED (silver + indium + bubble detectors, ~10^10–10^11)
- Spectrum: SUPPORTED (Klir 2011 TOF detector calibrated at PF-1000; Krasa 2008 TOF discussion)
- Anisotropy: SUPPORTED (5-point angular distribution, Y0°/Y90° ~1.8, Gribkov/Scholz 2007)

**Still missing:** Directly measured (not model-derived) pinch-phase Te and Ti. This is a structural gap for the full-energy PF-1000 as well — the corpus contains no peer-reviewed spectroscopic pinch-phase temperature measurement for PF-1000 at any voltage.

The second strongest candidate is **MJOLNIR (LLNL, 2 MJ, 3–4 MA)** with similar channel coverage but: (a) it is a different machine class (2 MJ vs 1 kJ–1 MJ DPF scope); (b) Te/Ti are simulated-only; (c) spectral and anisotropy data are partially from simulation with qualitative experimental confirmation rather than fully independent experimental measurement.

**Faeton-I** is a strong candidate for I(t), V(t), Yn, and energy spectrum, but lacks ne, Te, Ti, X-ray absolute yield calibration, and full angular anisotropy. Not sufficient as a standalone demonstrator.

---

## Recommendation

**Option B — full-energy PF-1000 (27–40 kV, Gribkov/Scholz 2007 era)**, with explicit acknowledgment of the Te/Ti gap.

**Justification:** The full-energy PF-1000 dataset in this corpus provides 7 of 9 required channels from peer-reviewed sources (I(t), V(t), ne, X-ray, Yn, neutron TOF spectrum, and angular anisotropy) and is the only single DPF device with multi-channel coverage across all radiation diagnostics in this corpus. Te and Ti remain absent as direct measurements for the DPF pinch phase at any device, making them a structural gap in the field — not a gap specific to this choice of target. Acquiring the Akel 16 kV data (Option A) would require a dedicated IPPLM campaign of 3–6 months minimum and would still leave Te and Ti in TEXT-ONLY status without new diagnostic investment. Option B enables V&V work to proceed now using the existing corpus; Te/Ti can be flagged as "model-validated, not spectroscopically validated" pending future campaign data.

**Caveat:** The code must be validated at the full-energy PF-1000 operating point, which differs substantially from the 16 kV Akel target (I_peak ~2 MA vs ~1 MA, energy ~810 kJ vs ~170 kJ). If the original Akel 16 kV scope was chosen for reasons of computational tractability (lower current, simpler dynamics), a change of demonstrator must be documented as a scope change in the V&V certificate with explicit acknowledgment that the 16 kV results are extrapolated from the validated 27–40 kV regime.
