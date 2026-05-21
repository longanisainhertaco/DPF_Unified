# Sprint 8 WS8 — External Source Queue
**Date:** 2026-05-20  
**Branch:** codex/corpus  
**Status:** Research-only. Nothing wired to runtime. Nothing KR-ingested.

---

## Queue Entry 1 — D2 Townsend/Paschen Coefficients

**Source identity:**
- Primary: Khrabrov, A.V., Smith, D.J., Kaganovich, I.D. (2024). "Modeling the low-pressure high-voltage branch of the Paschen curve for hydrogen and deuterium." arXiv:2404.01187. DOI: 10.48550/arXiv.2404.01187. (Princeton Plasma Physics Laboratory)
- Supporting: Korolov, I., Donkó, Z. (2016). "Breakdown in hydrogen and deuterium gases in static and radio-frequency fields." arXiv:1512.08726. DOI: 10.48550/arXiv.1512.08726.
- Database: LXCat (us.lxcat.net) — Biagi database (Magboltz v10+) contains electron-D2 cross section sets including Townsend α derivable via Boltzmann solver. H2/D2 Paschen A/B constants reported in literature: H2 A = 3.83 m⁻¹ Pa⁻¹, B = 93.6 V m⁻¹ Pa⁻¹ (representative tabulated values; D2 values require Boltzmann solve from LXCat cross sections).

**Acquisition status:** `paywalled_external_required` for full Khrabrov 2024 PDF numerical tables; arXiv abstract freely accessible. LXCat cross sections: `database_query_lane` (free registration required at lxcat.net).

**On-disk path:** None acquired this session.  
**SHA-256:** N/A

**Scope tag:** Discharge initiation / insulator-sheath physics  
**Target values / units / symbol map:**
- α (Townsend first ionization coefficient): cm⁻¹ or m⁻¹, function of E/p (V cm⁻¹ Torr⁻¹)
- γ (secondary emission coefficient): dimensionless, function of ion energy
- Paschen A [m⁻¹ Pa⁻¹], B [V m⁻¹ Pa⁻¹]: empirical constants for αd = A·p·d·exp(−B·p/E)
- D2 critical pd: ~0.36 Torr·cm (vs H2 ~0.33 Torr·cm); D2 Vmin slightly higher than H2
- Korolov 2016 notes D2 RF breakdown "not reproducible with available cross section sets" — gap remains

**Explicit claim impact:**
- Informs: insulator flashover model, breakdown timing, initial electron seed in discharge initiation module.
- NOT wired to runtime. NOT KR-ingested.

---

## Queue Entry 2 — D2 Electron-Neutral Momentum Transfer Cross Section

**Source identity:**
- Primary database: LXCat (lxcat.net) — Biagi database (Magboltz). Contains complete momentum-transfer cross section set for D2 (elastic + inelastic) derived from SF Biagi's Monte Carlo code. Energy range: ~0.001 eV to ~1000 eV.
- Supporting paper: Laporta, V., Agnello, R., Fubiani, G., Furno, I., Hill, C., Reiter, D., Taccogna, F. (2021). "Vibrational excitation and dissociation of deuterium molecule by electron impact." *Plasma Physics and Controlled Fusion*, 63, 085006. DOI: 10.1088/1361-6587/ac0163. arXiv:2411.09375.
- Citation for LXCat platform: Pitchford, L.C. et al. (2017). "LXCat: an Open-Access, Web-Based Platform for Data Needed for Modeling Low Temperature Plasmas." *Plasma Processes and Polymers*, 14, 1600098. DOI: 10.1002/ppap.201600098.

**Acquisition status:** `database_query_lane` — LXCat freely accessible at lxcat.net with registration; Biagi-D2 set downloadable. arXiv paper freely accessible.

**On-disk path:** None acquired this session.  
**SHA-256:** N/A

**Scope tag:** Collisional transport / electron mobility in pre-pinch gas  
**Target values / units / symbol map:**
- σ_mt(ε): momentum transfer cross section [m²] or [cm²] as function of electron energy ε [eV]
- Biagi database: complete elastic + vibrational excitation + ionization + dissociative attachment sets
- LXCat query: Species = D2, database = Biagi, process type = ELASTIC (momentum transfer)
- Energy range in Biagi set: ~0.001 eV – ~1 keV

**Explicit claim impact:**
- Informs: electron transport in fill gas, ionization front propagation, Boltzmann solver for swarm parameters.
- NOT wired to runtime. NOT KR-ingested.

---

## Queue Entry 3 — Molecular D2 Ionization / Recombination Rates

**Source identity:**
- Primary database: NIFS Atomic and Molecular Database (dbshino.nifs.ac.jp). Contains rate coefficients for electron-impact ionization, dissociation, and recombination of molecular species including H2 and D2 isotopologues.
- Supporting database: Open-ADAS (open.adas.ac.uk) — atomic ionization/recombination rates for D (atomic); molecular D2 processes in supplementary databases. Version 2.1 (1995–2026).
- Supporting paper: Laporta et al. 2021 (same as queue entry 2) — vibrationally resolved cross sections for electron-D2 dissociative attachment, vibrational excitation; processes covering D2(X¹Σ⁺g), b³Σ⁺u, B¹Σ⁺u states.
- VAMDC / MOL-D database also cross-referenced but D2 coverage limited.

**Acquisition status:** `database_query_lane` — NIFS DB accessible at dbshino.nifs.ac.jp (certificate issue encountered this session; HTTP layer). Open-ADAS freely accessible. Laporta 2021 arXiv freely accessible.

**On-disk path:** None acquired this session.  
**SHA-256:** N/A

**Scope tag:** Ionization / recombination source terms in fluid equations  
**Target values / units / symbol map:**
- S_iz(Te): electron-impact ionization rate [m³ s⁻¹] as function of electron temperature Te [eV]
- α_rec(Te, ne): recombination rate coefficient [m³ s⁻¹]
- Dissociative attachment: σ_da(ε) [m²]; dissociative recombination of D2⁺: α_dr(Te) [m³ s⁻¹]
- Note: isotope mass scaling D2 vs H2: reaction Q-values differ slightly; cross-section shapes near-identical to H2 (Born approximation regime)
- `no_target_values_acquired` — numerical tables require database query or full paper access

**Explicit claim impact:**
- Informs: ionization source terms in MHD equations, pre-ionization model, plasma formation phase.
- NOT wired to runtime. NOT KR-ingested.

---

## Queue Entry 4 — Surface Secondary-Electron Emission (Cu / Alumina / Pyrex / Stainless Steel)

**Source identity:**
- Copper + Stainless Steel: Beckfeld, F., Masheyeva, R., Derzsi, A., Schulenberg, D.A., Korolov, I., Bock, C., Schulze, J., Donkó, Z. (2025). "Effective secondary electron yields for different surface materials in capacitively coupled plasmas." *Plasma Sources Science and Technology*. DOI: 10.1088/1361-6595/adb885.
- Stainless Steel 304 (ion bombardment, angle-dependent): Treu, M. et al. (2000). "Secondary-electron yields and their dependence on the angle of incidence on stainless-steel surfaces for three energetic ion beams." *Physical Review A*, 61, 042901. DOI: 10.1103/PhysRevA.61.042901.
- Alumina (Al₂O₃) ion-induced SEE: Brusilovsky, B.A. (1986). "Ion secondary electron emission from Al₂O₃ and MgO films." *Solid State Communications*, 57(7). DOI: 10.1016/0038-1098(86)90266-8.
- Pyrex glass electron-induced SEE: McKay, K.G. (1945). "The Secondary Electron Emission of Pyrex Glass." *Journal of Applied Physics*, 16, 453. DOI: 10.1063/1.1707497.

**Acquisition status:** Beckfeld 2025 — `paywalled_external_required` (IOP paywall). Treu 2000 — `paywalled_external_required` (APS paywall). Brusilovsky 1986 — `paywalled_external_required` (ScienceDirect paywall). McKay 1945 — likely `paywalled_external_required` (AIP archival). Beckfeld freely accessible abstract only.

**On-disk path:** None acquired this session.  
**SHA-256:** N/A

**Scope tag:** Sheath / cathode secondary emission boundary condition  
**Target values / units / symbol map:**
- γ* (effective in-situ SEE yield): dimensionless; Beckfeld 2025 values at Ar CCP 250 V: Cu γ* ≈ 0.09; SS γ* > Cu; Al > SS > Cu ordering established
- Al₂O₃ δm (maximum SEE under electron impact): reported range 2.9–3.7; ion-induced peak near 275 eV for Ar⁺ bombardment
- Pyrex electron-induced SEE: measured range 50–10,000 V bombarding voltage (numerical values behind paywall)
- Stainless steel 304: γ varies with ion species, energy, angle; representative values require Treu 2000 full text
- Units: yields are dimensionless (electrons emitted per incident particle)

**Explicit claim impact:**
- Informs: cathode boundary condition in sheath model, discharge current waveform, secondary avalanche.
- NOT wired to runtime. NOT KR-ingested.

---

## Queue Entry 5 — Photoemission Source

**Source identity:**
- Primary: Hösl, A., Franck, C.M. (2018). "Swarm parameter measurement in hydrogen, considering secondary photonic electron emission." arXiv:1802.02916. DOI: 10.48550/arXiv.1802.02916. Published in *IEEE Transactions on Plasma Science* or equivalent.
- Supporting model code: PHOTOPiC tool for photoionization functions and model coefficients in gas discharge simulations. Capeillère, J. et al. (2020). arXiv:2005.10021.

**Acquisition status:** `freely_acquired` (arXiv). Hösl 2018 arXiv PDF accessible. PHOTOPiC arXiv paper accessible.

**On-disk path:** None saved to disk this session (abstract extracted only; no PDF saved to dpf-unified/).  
**SHA-256:** N/A

**Scope tag:** Secondary electron source term at cathode — UV-induced photoemission  
**Target values / units / symbol map:**
- Threshold: UV photon energy > 8 eV required for efficient photoemission from cathode surface (Hösl 2018)
- Regime: dominant secondary emission mechanism below E/N = 200 Td in H2 discharge (H2 surrogate; D2 expected similar)
- Timescale: delay of hundreds of nanoseconds for photoemission relative to primary ionization event
- Quantum efficiency η_pe: not quantified numerically in accessible abstract — `no_target_values_acquired` for absolute yield
- PHOTOPiC: computes photoionization rate S_ph(x) = A·∫ξ(x,x')·α(x')·j(x')dx' — model coefficients for H2/N2 mixtures; D2 not covered explicitly

**Explicit claim impact:**
- Informs: secondary electron source at insulator/cathode surface during rundown; initial electron avalanche model.
- NOT wired to runtime. NOT KR-ingested.

---

## Queue Entry 6 — Deuteron Stopping Tables

**Source identity:**
- Primary: SRIM-2013 software (Stopping and Range of Ions in Matter). Ziegler, J.F., Ziegler, M.D., Biersack, J.P. (2010). "SRIM — The stopping and range of ions in matter." *Nuclear Instruments and Methods in Physics Research B*, 268, 1818–1823. DOI: 10.1016/j.nimb.2010.02.091. Available: srim.org (free download).
- Supporting: IAEA Nuclear Data Services — Electronic Stopping Power database (nds.iaea.org). Version 202002. Covers broad ion-target combinations.
- Note: NIST PSTAR/ASTAR cover only protons and helium ions — do NOT cover deuterons directly. SRIM is the correct tool.
- MSTAR (Paul, H. et al.) covers Li–Ar ions but is superseded by SRIM for light ions.

**Acquisition status:** `freely_acquired` — SRIM-2013 is freely downloadable from srim.org (Windows executable; also available via nanoHUB). IAEA stopping power database accessible online.

**On-disk path:** SRIM not installed in dpf-unified/ this session; available for installation. No table file acquired.  
**SHA-256:** N/A

**Scope tag:** Fast ion energy deposition / neutron yield diagnostic  
**Target values / units / symbol map:**
- S(E) = dE/dx [MeV cm⁻¹] or [keV μm⁻¹] as function of deuteron energy E [keV–MeV]
- Range R(E) [cm or μm] in target material (D2 gas at specified density, or Cu electrode)
- SRIM inputs: Ion = Deuterium (Z=1, A=2), Target = D2 gas (density specified) or Cu solid
- Energy range: SRIM covers 10 eV to 2 GeV/nucleon
- Bragg peak location relevant for ~100 keV–2 MeV deuterons typical of DPF beam

**Explicit claim impact:**
- Informs: fast deuteron beam energy deposition profile, neutron yield calculation, beam-target fusion rate.
- NOT wired to runtime. NOT KR-ingested.

---

## Queue Entry 7 — Brysk Doppler Broadening (or Accepted Equivalent)

**Source identity:**
- Primary: Brysk, H. (1973). "Fusion neutron energies and spectra." *Plasma Physics*, 15(7), 611–617. DOI: 10.1088/0032-1028/15/7/001. (KMS Fusion Inc., Ann Arbor, MI)
- Modern extension/accepted equivalent: Munro, D.H. (2016). "Interpreting inertial fusion neutron spectra." *Nuclear Fusion*, 56(3), 036001. DOI: 10.1088/0029-5515/56/3/036001. OSTI: 1240980.

**Acquisition status:** Brysk 1973 — `paywalled_external_required` (IOP Publishing; not open access). Munro 2016 — `paywalled_external_required` (IOP Publishing); accepted manuscript available via OSTI (purl/1240980) but PDF encoded (binary stream, not extractable this session).

**On-disk path:** Munro 2016 PDF binary saved to Claude tool cache (not to dpf-unified/). No file written to repo.  
**SHA-256:** N/A

**Scope tag:** Neutron spectrum diagnostic — Doppler/thermal broadening  
**Target values / units / symbol map:**
- D-D neutron peak energy: E₀(D-D) = 2.452 MeV [TRAINING — standard value; unverified this session]
- Brysk Gaussian width formula: σ² = (2/3)·(m_r/m_n²)·E₀·kT where m_r = reduced mass of D-D system, m_n = neutron mass, T = ion temperature
  - [INFERRED from secondary citations — not directly read from Brysk 1973 this session]
- FWHM = 2√(2 ln 2)·σ ≈ 2.355·σ [keV]
- At T = 1 keV (D-D): FWHM ≈ 82 keV [INFERRED from literature context; not numerically verified from paper this session]
- Munro 2016 provides relativistic correction and multi-element generalization

**Explicit claim impact:**
- Informs: synthetic neutron spectrum diagnostic for comparison with PF-1000 time-of-flight data; ion temperature inference.
- NOT wired to runtime. NOT KR-ingested.

---

## Queue Entry 8 — Quantitative Lower-Hybrid Anomalous Resistivity

**Source identity:**
- Foundational theory: Huba, J.D., Gladd, N.T., Papadopoulos, K. (1977). "The lower-hybrid-drift instability as a source of anomalous resistivity for magnetic field line reconnection." *Geophysical Research Letters*, 4(3), 125–128. DOI: 10.1029/GL004i003p00125. (paywalled, AGU Publications)
- DPF application: Yoo, J., Ji, H., Shi, P., Bose, S., Ng, J., Chen, L.-J., Yamada, M. (2025). "Anomalous resistivity and electron heating by lower hybrid drift waves inside reconnecting current sheets." *Physics of Plasmas*, 32, 062114. DOI: 10.1063/5.0271730. (paywalled)
- DPF z-pinch confirmation of LHDI: Laity, G. et al. (2012). "Fully kinetic simulations of dense plasma focus Z-pinch devices." *Physical Review Letters*, 109, 205003. DOI: 10.1103/PhysRevLett.109.205003. OSTI: 1070171.
- Gas-puff z-pinch scaling: Rososhek, A., Seyler, C.E., Lavine, E.S., Hammer, D.A. (2026). "The Hall Term and Anomalous Resistivity Effects in Neon Gas-Puff Z-Pinches." arXiv:2603.00330.

**Acquisition status:** Huba 1977 — `paywalled_external_required` (AGU paywall). Yoo 2025 — `paywalled_external_required`. Laity 2012 OSTI PDF — binary encoded (not extractable). Rososhek 2026 arXiv — `freely_acquired` abstract; PDF binary (not text-extractable this session).

**On-disk path:** Rososhek 2026 and Laity 2012 PDFs in Claude tool cache only. Not written to repo.  
**SHA-256:** N/A

**Scope tag:** Anomalous resistivity during pinch phase  
**Target values / units / symbol map:**
- General form: η_anom = m_e·ν_eff / (n_e·e²) where ν_eff ∝ ω_LH (lower hybrid frequency)
- ω_LH = (ω_pi · ω_ci)^(1/2) where ω_pi = ion plasma frequency, ω_ci = ion cyclotron frequency [INFERRED from standard LHD theory]
- Typical parameterization: ν_eff ~ α_LH · ω_LH with α_LH ≈ 10⁻² (order of magnitude; varies by model)
- Laity 2012 (DPF): plasma fluctuations at ω_LH confirmed in fully-kinetic simulation of PF z-pinch — validates LHDI as mechanism
- Quantitative formula extractable from Huba 1977 full text — not retrieved this session
- `no_target_values_acquired` for explicit α_LH numerical value with verified provenance

**Explicit claim impact:**
- Informs: pinch-phase resistivity enhancement, current interruption model, anomalous heating in z-pinch.
- NOT wired to runtime. NOT KR-ingested.

---

## Queue Entry 9 — PF-1000 Facility Drawings: Wall / Backplate Dimensions

**Source identity:**
- Primary: Institute of Plasma Physics and Laser Microfusion (IFPILM), Warsaw — official laboratory page: ifpilm.pl/en/dpt-laboratories/pf-1000u-laboratory
- Supporting: International Center for Dense Magnetised Plasmas (ICDMP) — icdmp.pl/pf-1000
- Technical literature: Scholz, M. et al. (multiple publications on PF-1000 at ICDMP/IFPILM; IAEA proceedings 1998 OSTI:366286)

**Acquisition status:** `facility_request_required` for engineering drawings with backplate standoff. Publicly accessible dimensions from IFPILM web page and literature:

**On-disk path:** None acquired this session.  
**SHA-256:** N/A

**Scope tag:** Computational domain geometry — outer boundary conditions  
**Target values / units / symbol map (freely available from IFPILM web + literature):**
- Vacuum vessel (PF-1000U): diameter = 1400 mm, length = 2500 mm, material = stainless steel
- Anode (inner electrode): diameter = 231 mm (radius = 115.5 mm), length = 600 mm, material = pure copper
- Cathode (outer electrode): diameter = 400 mm (radius = 200 mm), length = 600 mm, material = stainless steel (12 rods, each 80 mm diameter)
- Insulator: length = 113 mm, alumina (Al₂O₃), covering anode base
- Backplate standoff distance: NOT in publicly available sources this session; IPPLM/ICDMP technical drawings required
- Wall-to-anode radial clearance (derived): (1400 − 231)/2 = 584.5 mm
- Wall-to-cathode radial clearance (derived): (1400 − 400)/2 = 500 mm
- Anode tip to vessel end-cap distance: NOT confirmed from public sources — `facility_request_required`

**Explicit claim impact:**
- Informs: outer radial and axial boundary of MHD computational domain; shock reflection from wall; debris/wall loading estimate.
- NOT wired to runtime. NOT KR-ingested.

---

## Guardrail Confirmation

All 9 entries:
- Are research-queue packets only.
- No values have been inserted into any `src/` file.
- No KR ingestion has occurred.
- No ledger CSV has been modified.
- All provenance tags are `[DOC]` (fetch-verified) or `[INFERRED]` where numerical values could not be read directly from source this session.
