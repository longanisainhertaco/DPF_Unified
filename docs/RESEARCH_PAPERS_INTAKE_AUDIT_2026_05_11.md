# Research Papers Intake Audit

Generated: 2026-05-11

Input folder: `/Users/anthonyzamora/dpf-unified/downloaded_books_papers/Research Papers`

Source-of-truth guardrail: files in this folder are acquisition/intake artifacts only. They are not scientific evidence until reviewed into `KnowledgeReference/`, hashed, and mapped to source/target records.

Post-promotion update 2026-05-11: `scripts/promote_research_papers_to_kr.py --apply` promoted 54 unique local PDFs into `KnowledgeReference/` markdown/JSON text-parity records, skipped 7 already represented source-level records, and deleted 16 exact byte-for-byte duplicate intake files. The intake folder now has 61 PDF-like files with 61 unique SHA-256 payloads. Promotion status is `text_parity_extracted_review_needed`; figures, tables, plotted curves, and numeric validation targets still require separate review/target extraction.

Supplemental intake update 2026-05-11: 30 user-supplied PDFs were copied into
`2026-05-11-user-ingest/` and the promotion script was rerun. The current
promotion manifest is `docs/RESEARCH_PAPERS_KR_PROMOTION_2026_05_11.md` /
`.json`: 91 unique intake PDFs scanned, 32 new KR text-parity records promoted,
59 already represented records skipped, 0 failures, and 0 duplicate deletions.
The new records remain `text_parity_extracted_review_needed`; title cleanup,
figure/table review, target extraction, and uncertainty extraction remain
separate blockers.

## Summary

- Files scanned: 77
- PDF files: 76
- Non-PDF/incomplete-download files: 1
- Unique SHA-256 payloads: 61
- Exact duplicate hash groups: 13
- Files with repo/KR/status matches: 2
- Intake files not matched to current repo/KR/status text: 75
- Acquisition workbook targets manually confirmed in this folder: 1 of 91
- Rejected fuzzy title matches still missing: 3 (`ACQ-003`, `ACQ-023`, `ACQ-079`)

## Exact Duplicate Groups

| sha12 | count | paths | title |
| --- | --- | --- | --- |
| 3f0ebd080206 | 2 | AD1098588_AccelerationofProtonsandDeuteronsUpto35MevandGenerationof1.pdf<br>Wave 5/AD1098588_AccelerationofProtonsandDeuteronsUpto35MevandGenerationof1.pdf | Accelerationof Protonsand Deuterons Upto35Mevand Generationof1 |
| d48f6741c06b | 2 | AD1123736_CharacterizationofElectronBeamsfromabDensebbPlasmabF.pdf<br>Wave 5/AD1123736_CharacterizationofElectronBeamsfromaDensePlasmaFocus.pdf | Characterization of Electron Beams from a Dense Plasma Focus |
| 3f5ffcf29a11 | 2 | Wave2/AD1194307_EnablingReduced-OrderCollisional-RadiativeModelingforReactivePla.pdf<br>Wave2/AD1194307_EnablingReduced-OrderCollisional-RadiativeModelingforReactiveb.pdf | Ena ling Reduced Order Collisional Radiative Modelingfor Reactive Pla |
| 365ec9f0b28d | 2 | Wave4/033302_1_5.0311453.pdf<br>Wave4/033302_1_5.0311453.pdf.crdownload | Formation and dynamics of Z-pinch plasma in a coaxial plasma gun |
| 8b83e6b55436 | 3 | AD1076777_OptimizationofDensePlasmaFocusbDPFbNeutronSourcesviaEx.pdf<br>AD1076777_OptimizationofbDensebbPlasmabFocusDPFNeutronSources.pdf<br>Wave 5/AD1076777_OptimizationofbDensebbPlasmabbFocusbDPFNeutron.pdf | Optimization of Dense Plasma Focus (DPF) Neutron Sources via Experiments and Kinetic Modeling |
| 2c016fb54004 | 2 | AD1079881_OptimizationofbDensebbPlasmabFocusDPFNeutronSources.pdf<br>Wave4/AD1079881_OptimizationofDensebPlasmabFocusDPFNeutronSourcesviaEx.pdf | Optimization of Dense Plasma Focus (DPF) Neutron Sources via Experiments and Kinetic Modeling |
| 153394e55da3 | 3 | AD1206892_Particle-in-CellModelingofOmegaExperimentsonAblationofbPlasm.pdf<br>AD1206892_bParticleb-in-bCellbModelingofOmegaExperimentsonAblati.pdf<br>Wave 5/AD1206892_Particle-in-CellModelingofOmegaExperimentsonAblationofbPlasm.pdf | Particle-in-Cell Modeling of Omega Experiments on Ablation of Plasmas |
| 62b3fa352371 | 2 | AD1302183_ReducingtheComputationalComplexityofImplicitSchemesintheModel.pdf<br>Wave4/AD1302183_ReducingtheComputationalComplexityofImplicitSchemesintheModel.pdf | Reducing the computational complexity of implicit schemes in the modeling of kinetic inelastic collisions in a partially ionized plasma |
| b5b77f274da5 | 3 | Wave 5/AD1194691_bSimulatingbaPulsed-Power-DrivenPlasmawithIdealbMHDb.pdf<br>Wave2/AD1194691_SimulatingaPulsed-Power-DrivenPlasmawithIdealbMHDb.pdf<br>Wave2/AD1194691_bSimulatingbaPulsed-Power-DrivenbPlasmabwithIdealMHD.pdf | Simulating a pulsed-power-driven plasma with ideal MHD |
| 0358c24d9e71 | 2 | AD1100306_SimulationsofabDensebbPlasmabFocusonaHigh-Impedance.pdf<br>Wave 5/AD1100306_bSimulationsbofaDensePlasmaFocusonaHigh-ImpedanceGenerat.pdf | Simulations of a Dense Plasma Focus on a High-Impedance Generator |
| 7c486e76c3d7 | 2 | AD1095975_SpatialDistributionofIonEmissioninGas-PuffZ-PinchesandbDens.pdf<br>AD1095975_SpatialDistributionofIonEmissioninGas-PuffbZb-bPinches.pdf | Spatial Distri utionof Ion Emissionin Gas Puff Z Pinchesand Dens |
| e15a4b416ec0 | 2 | Wave 5/AD1300646_StudiesofbPlasmabSheathPhysicsusingContinuumKineticbSim.pdf<br>Wave2/AD1300646_StudiesofbPlasmabSheathPhysicsusingContinuumKineticbSim.pdf | Studies of Plasma Sheath Physics using Continuum Kinetic Simulations of Plasmas |
| 548301391e2a | 2 | Wave2/AD1105890_TowardpredictivemodelingofExBbplasmabdischarges.pdf<br>Wave4/AD1105890_TowardpredictivemodelingofExBbplasmabdischarges.pdf | Towardpredictivemodelingof Ex B plasma discharges |

## Acquisition Targets Found In This Folder

Manual review accepted only one current acquisition target from `docs/SOURCE_ACQUISITION_TEAM_HANDOFF_2026_05_11.xlsx`.

| ID | priority | target | matching file | current status |
| --- | --- | --- | --- | --- |
| ACQ-038 | P2-kinetic | A. Schmidt et al., "Fully Kinetic Simulations of MegaJoule-Scale Dense Plasma Focus" | Wave 5/1169854.pdf | Found in intake; not yet KR-promoted/reviewed. |

Rejected fuzzy matches:

| target | false match | reason |
| --- | --- | --- |
| ACQ-003 Freidberg, `Ideal MHD` | Wave 5/AD1194691_bSimulatingbaPulsed-Power-DrivenPlasmawithIdealbMHDb.pdf | This is Beresnyak's pulsed-power MHD paper, not Freidberg's book. |
| ACQ-079 Schmidt/Tang/Welch 2012 PRL, `Fully kinetic simulations of dense plasma focus Z pinches` | Wave 5/1169854.pdf | This file is Schmidt et al. 2014 Physics of Plasmas, not the 2012 PRL article. |
| ACQ-023 Bernard et al. plasma-focus status review | AD1345078_EffectofCurrentSheathInitiationontheRadialCollapseandEnerget.pdf | This is an unrelated current-sheath report, not the Bernard status review. |

## Current Acquisition Targets Still Not Found In This Folder

Manual correction: `ACQ-003`, `ACQ-023`, and `ACQ-079` are also still missing despite being rejected from the generated fuzzy-match pass above.

| ID | priority | type | target | authors/leads |
| --- | --- | --- | --- | --- |
| ACQ-001 | P1-method | Book / monograph | Principles of Plasma Diagnostics, 2nd ed. | I. H. Hutchinson |
| ACQ-002 | P1-method | Book / monograph | Riemann Solvers and Numerical Methods for Fluid Dynamics, 3rd ed. | Eleuterio F. Toro |
| ACQ-004 | P2-method | Book / monograph | Magnetohydrodynamics of Laboratory and Astrophysical Plasmas | Hans Goedbloed; Rony Keppens; Stefaan Poedts |
| ACQ-005 | P2-method | Book / monograph | Plasma Physics via Computer Simulation | C. K. Birdsall; A. B. Langdon |
| ACQ-006 | P3-method | Book / monograph | Principles of Plasma Spectroscopy | Hans R. Griem |
| ACQ-007 | P3-method | Book / monograph | Radiative Processes in Astrophysics | George B. Rybicki; Alan P. Lightman |
| ACQ-008 | Optional | Book / monograph | The Physics of Inertial Fusion | Stefano Atzeni; Juergen Meyer-ter-Vehn |
| ACQ-009 | P1-blocker | Paper | Fusion neutron detector for time-of-flight measurements in z-pinch and plasma focus experiments | D. Klir et al. |
| ACQ-010 | P2-blocker | Paper | Measurements of fast ions and neutrons emitted from PF-1000 plasma focus device | M. Sadowski; M. Scholz; PF-1000 team |
| ACQ-011 | P2-blocker | Paper | Tomographic Reconstruction of the Neutron Time-Energy Spectrum from a Dense Plasma Focus | A. Catenacci et al. |
| ACQ-012 | P2-blocker | Paper | Plasma focus neutron energy and anisotropy measurements using zirconium-beryllium pair activation detectors | D. Springham et al. |
| ACQ-013 | P2-blocker | Paper | A new concept of fusion neutron monitoring for PF-1000 device | S. Jednorog et al. |
| ACQ-014 | P2-secondary | Paper | Temporal distribution of linear densities of plasma column in plasma focus discharge | E. Cikhardtova et al. |
| ACQ-015 | P2-neutron-method | Paper | Silver activation counter: Detector with large dynamic range for measurement of fast-neutron bursts | K. Rezac et al. |
| ACQ-016 | P2-neutron-method | Paper | Improvement of time-of-flight methods for reconstruction of neutron energy spectra from D(d,n)3He fusion reactions | K. Rezac; D. Klir; P. Kubes; J. Kravarik |
| ACQ-017 | P2-neutron-mechanism | Paper | Search for thermonuclear neutrons in a mega-ampere plasma focus | D. Klir; P. Kubes; PF-1000 Team |
| ACQ-018 | P1-current-sheath | Paper | Experimental study of the structure of the plasma-current sheath on the PF-1000 facility | V. Krauz et al. |
| ACQ-019 | P1-phase | Paper | Scenario of pinch evolution in a plasma focus discharge | P. Kubes; D. Klir; PF-1000 Team |
| ACQ-020 | P2-current-energy | Paper | Current flow and energy balance during the evolution of instabilities in the plasma focus | J. Kortanek; P. Kubes; PF-1000 Team |
| ACQ-021 | P2-program-review | Paper / review | Progress in MJ plasma focus research at IPPLM | M. Scholz et al. |
| ACQ-022 | P2-review | Paper / review | Update on the Scientific Status of Plasma Focus | A. Auluck et al. |
| ACQ-024 | P1-phase-spatial | Paper | Sixteen-frame interferometer for a study of a pinch dynamics in PF-1000 device | E. Zielinska; M. Paduch; M. Scholz |
| ACQ-025 | P1-phase-neutron | Paper | Interferometric Study of Pinch Phase in Plasma-Focus Discharge at the Time of Neutron Production | P. Kubes et al. |
| ACQ-026 | P1-field-neutron | Paper | Correlation of magnetic probe and neutron signals with interferometry figures on the plasma focus discharge | P. Kubes et al. |
| ACQ-027 | P1-current-sheath | Paper | Study of the fine structure of the plasma current sheath and magnetic fields in the axial region of the PF-1000 facility | K. N. Mitrofanov et al. |
| ACQ-028 | P1-current-energy | Paper | Dynamics of implosion phase of modified plasma focus studied via laser interferometry and electrical measurements | J. Malir et al. |
| ACQ-029 | P1-neutron-anisotropy | Paper | Anisotropy of the emission of DD-fusion neutrons caused by the plasma-focus vessel | J. Krasa et al. |
| ACQ-030 | P1-neutron-response | Paper | Radioindium and determination of neutron radial asymmetry for the PF-1000 plasma focus device | S. Jednorog et al. |
| ACQ-031 | P2-neutron-mechanism | Paper | Experimental evidence of thermonuclear neutrons in a modified plasma focus | D. Klir et al. |
| ACQ-032 | P2-neutron-spectrum | Paper | Determination of Deuteron Energy Distribution From Neutron Diagnostics in a Plasma-Focus Device | P. Kubes et al. |
| ACQ-033 | P2-energy-coupling | Paper | Energy Transformations in Column of Plasma Focus Discharges with Megaampere Currents | P. Kubes et al. |
| ACQ-034 | P2-field-map | Paper | Mapping of azimuthal B-fields in Z-pinch plasmas using Z-pinch-driven ion deflectometry | V. Munzar et al. |
| ACQ-035 | P2-temperature | Paper | Optical emission spectroscopy of plasma streams in PF-1000 experiments | K. Jakubowska et al. |
| ACQ-036 | P2-temperature | Paper | Optical spectroscopy of free-propagating plasma and its interaction with tungsten targets in PF-1000 facility | E. Skladnik-Sadowska et al. |
| ACQ-037 | P2-model-form | Paper | MHD numerical modelling of the plasma focus phenomena | W. Stepniewski |
| ACQ-039 | P2-radiation | Paper | Conditions for Radiative Cooling and Collapse in the Plasma Focus Illustrated With Numerical Experiments on PF1000 | S. Lee et al. |
| ACQ-040 | P1-module-atomic | Paper | Electron-impact ionization cross-sections and ionization rate coefficients for atoms and ions | W. Lotz |
| ACQ-041 | P1-module-atomic | Paper | An empirical formula for the electron-impact ionization cross-section | W. Lotz |
| ACQ-042 | P1-module-atomic | Paper | Radiative Recombination of Hydrogenic Ions | M. J. Seaton |
| ACQ-043 | P1-module-atomic | Paper | Coronal ionization equilibrium and dielectronic recombination context | A. Burgess; M. J. Seaton |
| ACQ-044 | P1-module-atomic | Data sheet / authoritative database | NIST Atomic Spectra Database / SRD 78 ionization potential data | NIST ASD / SRD 78 |
| ACQ-045 | P1-module-radiation | Paper / radiation table | Steady-state radiative cooling rates for low-density high-temperature plasma | D. E. Post et al. |
| ACQ-046 | P1-module-radiation | Data sheet / manual | ADAS User Manual and collisional-radiative data documentation | H. P. Summers / ADAS Project |
| ACQ-047 | P1-module-radiation | Data sheet / paper | CHIANTI atomic database radiative loss and ionization documentation | G. Del Zanna et al.; K. P. Dere et al. |
| ACQ-048 | P1-module-radiation | Paper / radiation table | Calculation and experimental test of the cooling factor of tungsten | T. Puetterich et al. |
| ACQ-049 | P2-module-ablation | Paper | Dense plasma focus electrode ablation and erosion source lead | H. Bruzzone; L. Aranchuk |
| ACQ-050 | P2-module-ablation | Paper | Plasma focus modelling with ablation source terms | S. Lee; A. Serban |
| ACQ-051 | P2-module-ablation | Review / paper | Plasma erosion of materials under pulsed high-current discharges | V. Vikhrev; V. Korolev |
| ACQ-052 | P2-module-transport | Paper | Dissipation of currents in ionized media | O. Buneman |
| ACQ-053 | P2-module-transport | Paper | Anomalous transport properties associated with the lower-hybrid-drift instability | R. C. Davidson; N. T. Gladd |
| ACQ-054 | P2-module-transport | Paper | Effects of finite plasma beta on the lower-hybrid-drift instability | R. C. Davidson; N. T. Gladd; C. S. Wu; J. D. Huba |
| ACQ-055 | P2-module-transport | Review / book | Dense plasma focus review and plasma instability textbook support | M. G. Haines; N. A. Krall; A. W. Trivelpiece |
| ACQ-056 | P2-module-scaling | Paper | Neutron scaling laws from numerical experiments | S. Lee; S. H. Saw |
| ACQ-057 | P2-module-scaling | Paper | Current scaling of plasma focus neutron yield | S. Lee; S. H. Saw |
| ACQ-058 | P2-module-scaling | Paper | Small dense plasma focus scaling studies | L. Soto et al. |
| ACQ-059 | P1-module-pb11 | Paper / table | The thermonuclear fusion rate coefficient for p-11B reactions | W. M. Nevins; R. Swain |
| ACQ-060 | P1-module-pb11 | Paper | Fundamental limitations on plasma fusion systems not in thermodynamic equilibrium | T. H. Rider |
| ACQ-061 | P1-module-pb11 | Paper | A review of p-11B fusion reactivity and aneutronic fusion data | M. Sikora; H. R. Weller |
| ACQ-062 | P1-module-pb11 | Paper / data | p-11B S-factor and cross-section measurements | H. W. Becker et al. |
| ACQ-063 | P2-module-thomson | Paper | Electron density fluctuations in a plasma | E. E. Salpeter |
| ACQ-064 | P2-module-thomson | Paper | Effect of magnetic field on electron density fluctuations in a plasma | E. E. Salpeter |
| ACQ-065 | P2-module-thomson | Book / chapter | Plasma Scattering of Electromagnetic Radiation | D. H. Froula; S. H. Glenzer; N. C. Luhmann Jr.; J. Sheffield |
| ACQ-066 | P2-module-xray | Paper | Dense plasma focus X-ray diagnostics and detector/filter response source lead | Shan et al. |
| ACQ-067 | P2-module-xray | Data sheet / paper | Pinhole camera and X-ray diagnostic response metadata | NIST / RSI source lead |
| ACQ-068 | P2-module-stability | Paper | Hydromagnetic instability criteria for plasma interfaces | M. D. Kruskal; M. Schwarzschild |
| ACQ-069 | P2-module-stability | Book / paper | Hydromagnetic stability of plasma | B. B. Kadomtsev |
| ACQ-070 | P2-module-stability | Paper | Finite-resistivity instabilities of a sheet pinch | H. P. Furth; J. Killeen; M. N. Rosenbluth |
| ACQ-071 | P2-module-stability | Book | Magnetic Reconnection in Plasmas | D. Biskamp |
| ACQ-072 | P2-module-stability | Paper | Sheared flow stabilization of the m=1 kink mode in Z pinches | U. Shumlak; C. W. Hartman |
| ACQ-073 | P2-module-stability | Paper | Sheared flow stabilization of Z-pinch plasmas | U. Shumlak et al. |
| ACQ-074 | P2-module-civ | Review / paper | Critical ionization velocity review and laboratory experiment leads | N. Brenning |
| ACQ-075 | P2-module-civ | Paper | Experiments on the critical ionization velocity | L. Danielsson |
| ACQ-076 | P2-module-civ | Book / data | Gas discharge coefficients and Paschen breakdown sources | Meek & Craggs; Lieberman & Lichtenberg |
| ACQ-077 | P2-module-pic | Paper | Theory of cumulative small-angle collisions in plasmas | K. Nanbu |
| ACQ-078 | P2-module-pic | Paper | A new method for calculating Coulomb collisions in plasma simulations | F. Perez et al. |
| ACQ-080 | P2-module-sheath | Book / review | Plasma sheath and Bohm criterion source material | Lieberman & Lichtenberg; Riemann/Bohm review candidates |
| ACQ-081 | P3-module-verification | Paper | The formation of a blast wave by a very intense explosion. I. Theoretical discussion | G. I. Taylor |
| ACQ-082 | P3-module-verification | Historical paper/book | Sedov/Taylor-von Neumann/Lin/Sakurai cylindrical blast-wave sources | L. I. Sedov; J. von Neumann; C. C. Lin; A. Sakurai |
| ACQ-083 | P2-module-backend | Paper | Athena: A new code for astrophysical MHD | J. M. Stone et al. |
| ACQ-084 | P2-module-backend | Paper | A simple unsplit Godunov method for multidimensional MHD | J. M. Stone; T. A. Gardiner |
| ACQ-085 | P2-module-backend | Paper | Athena++: a performance-portable astrophysical MHD code | J. M. Stone et al. |
| ACQ-086 | P2-module-backend | Paper | AthenaK method paper and performance-portable backend lead | AthenaK authors |
| ACQ-087 | P1-data-provenance | Paper / model-card / code | Walrus: A Foundation Model for the Simulation of Multiphysics Shocks | WALRUS authors |
| ACQ-088 | P1-data-provenance | Paper / dataset format | The Well: a Large-Scale Collection of Diverse Physics Simulations for Machine Learning | Ohana et al. |
| ACQ-089 | P1-data-provenance | Dataset / data sheet | The Well MHD_64 and MHD_256 public datasets | Polymathic AI / The Well |
| ACQ-090 | P2-data-provenance | Paper / dataset | The Catalogue for Astrophysical Turbulence Simulations (CATS) | B. Burkhart et al. |
| ACQ-091 | P2-data-provenance | Guidance / standard | NASA CFD Verification and Validation Tutorial | NASA |

## Files Already Represented Somewhere In Repo/KR/Status Text

Manual correction: the generated match table below is conservative and missed at least two exact title/DOI matches already promoted in `KnowledgeReference/`:

- `AD1194691` / "Simulating a pulsed-power-driven plasma with ideal MHD" is represented by `KnowledgeReference/beresnyak_2022_pulsed_power_ideal_mhd.md` and `.json`; three exact duplicate intake copies remain.
- `AD1100306` / "Simulations of a Dense Plasma Focus on a High-Impedance Generator" is represented by `KnowledgeReference/beresnyak_2018_dpf_hawk_simulations.md` and `.json`; two exact duplicate intake copies remain.
- `AD1116543_2019NRLPlasmaFormulary.pdf` overlaps existing `KnowledgeReference/plasma-formulary.md` formulary coverage, but the intake PDF edition has not been checked as the exact promoted source.

| path | title | accession | sha12 | relevance | matched repo files |
| --- | --- | --- | --- | --- | --- |
| AD1079881_OptimizationofbDensebbPlasmabFocusDPFNeutronSources.pdf | Optimization of Dense Plasma Focus (DPF) Neutron Sources via Experiments and Kinetic Modeling | AD1079881 | 2c016fb54004 | high | docs/SCIENTIFIC_CLOSURE_SOURCE_QUEUE.md |
| Wave4/AD1079881_OptimizationofDensebPlasmabFocusDPFNeutronSourcesviaEx.pdf | Optimization of Dense Plasma Focus (DPF) Neutron Sources via Experiments and Kinetic Modeling | AD1079881 | 2c016fb54004 | high | docs/SCIENTIFIC_CLOSURE_SOURCE_QUEUE.md |

## Intake Files Not Yet Matched To Repo/KR

Manual correction: `AD1194691`, `AD1100306`, and `AD1116543` appear in this generated section because matching was file/text based. Treat the first two as already KR-represented duplicates, and treat the NRL Formulary PDF as edition-review-needed.

| path | title | accession | pages | sha12 | relevance |
| --- | --- | --- | --- | --- | --- |
| Wave 5/1169854.pdf |  | 1169854 | 15 | 3f439245a587 | high |
| Wave2/ADA635937_2DMHDComputerModelingofbDensebbPlasmabbFocusbAc.pdf | 2D MHD COMPUTER MODELING OF DENSE PLASMA FOCUS ACCELERATORS | ADA635937 | 6 | 74d0df14b491 | high |
| ADA433824_AdvancementsinDensePlasmaFocusbDPFbforSpacePropulsion.pdf | Advancements in Dense Plasma Focus (DPF) for Space Propulsion | ADA433824 | 9 | 6e45eca49152 | high |
| ADA454652_AnInvestigationofBremsstrahlungReflectioninaDensePlasmaFocus.pdf | An Investigation of Bremsstrahlung Reflection in a Dense Plasma Focus | ADA454652 | 9 | f17533f2f1c9 | high |
| Wave2/ADA610142_Characterizationofa500JbDensebbPlasmabbFocusbfor.pdf | CHARACTERIZATION OF A 500J DENSE PLASMA FOCUS FOR PRODUCING SOFT X-RAYS | ADA610142 | 5 | db2349b11e58 | high |
| AD1123736_CharacterizationofElectronBeamsfromabDensebbPlasmabF.pdf | Characterization of Electron Beams from a Dense Plasma Focus | AD1123736 | 19 | d48f6741c06b | high |
| Wave 5/AD1123736_CharacterizationofElectronBeamsfromaDensePlasmaFocus.pdf | Characterization of Electron Beams from a Dense Plasma Focus | AD1123736 | 19 | d48f6741c06b | high |
| AD1345078_EffectofCurrentSheathInitiationontheRadialCollapseandEnerget.pdf | Effect of current sheath initiation on the radial collapse and energetic particle acceleration in 10 kJ Dense Plasma Focus | AD1345078 | 22 | b2e95b882b6a | high |
| AD1076777_OptimizationofDensePlasmaFocusbDPFbNeutronSourcesviaEx.pdf | Optimization of Dense Plasma Focus (DPF) Neutron Sources via Experiments and Kinetic Modeling | AD1076777 | 8 | 8b83e6b55436 | high |
| AD1076777_OptimizationofbDensebbPlasmabFocusDPFNeutronSources.pdf | Optimization of Dense Plasma Focus (DPF) Neutron Sources via Experiments and Kinetic Modeling | AD1076777 | 8 | 8b83e6b55436 | high |
| Wave 5/AD1076777_OptimizationofbDensebbPlasmabbFocusbDPFNeutron.pdf | Optimization of Dense Plasma Focus (DPF) Neutron Sources via Experiments and Kinetic Modeling | AD1076777 | 8 | 8b83e6b55436 | high |
| Wave2/ADA635195_PFMA-1A1-Hz150-kJPulsedPowerSystemforPlasmaFocusGeneration.pdf | PFMA-1: A 1-Hz, 150-kJ PULSED POWER SYSTEM FOR PLASMA FOCUS GENERATION | ADA635195 | 6 | 9590eb6507db | high |
| AD1100306_SimulationsofabDensebbPlasmabFocusonaHigh-Impedance.pdf | Simulations of a Dense Plasma Focus on a High-Impedance Generator | AD1100306 | 5 | 0358c24d9e71 | high |
| Wave 5/AD1100306_bSimulationsbofaDensePlasmaFocusonaHigh-ImpedanceGenerat.pdf | Simulations of a Dense Plasma Focus on a High-Impedance Generator | AD1100306 | 5 | 0358c24d9e71 | high |
| Wave4/033302_1_5.0311453.pdf.crdownload | 033302 1 5.0311453 | 033302_1_5.0311453 |  | 365ec9f0b28d | low |
| Wave2/AD1334827_AHybridModelforMultiscaleLaserbPlasmabbSimulationsbw.pdf | AHy rid Modelfor Multiscale Laser Plasma Simulations w | AD1334827 | 20 | 69b33ebe2e88 | low |
| AD1338534_AssemblingaDeep-HistoryBinaryCorpus (1).pdf | Assem linga Deep History Binary Corpus (1) | AD1338534 | 13 | c01698b17c7e | low |
| Wave2/DSIAC-2195387_ClassicalTrajectoryMonteCarlobSimulationbofbPlasmabFu.pdf | Classical Trajectory Monte Carlo Simulation of Plasma Fu | DSIAC-2195387 | 9 | 31246535f2e0 | low |
| Wave4/AD1230244_DevelopingMethodsofControlofSelf-OrganizedbPlasmabStructur.pdf | Developing Methodsof Controlof Self Organized Plasma Structur | AD1230244 | 17 | 6d5767901062 | low |
| Wave2/ADA462260_DeviceDemonstration.pdf | Device Demonstration | ADA462260 | 190 | 72fd2b544e20 | low |
| ADA599854_ElectrodynamicPropertiesofbDensebSemiclassicalbPlasmab.pdf | Electrodynamic Propertiesof Dense Semiclassical Plasma | ADA599854 | 3 | 9a31f386d8a5 | low |
| Wave2/AD1337397_Electronemissionbphysicsbbsimulationsb.pdf | Electronemission physics simulations | AD1337397 | 16 | 4c71fbe527f7 | low |
| Wave2/AD1331701_ExperimentModelingandbSimulationbofAdvancedMaterials-b.pdf | Experiment Modelingand Simulation of Advanced Materials | AD1331701 | 18 | dc4703eb8af1 | low |
| Wave2/AD1338034_ExperimentModelingandbSimulationbofAdvancedMaterials-b.pdf | Experiment Modelingand Simulation of Advanced Materials | AD1338034 | 18 | ea1817c6d504 | low |
| ADA589175_HierarchicalReconstructionwithUptoSecondDegreeRemainderforSol.pdf | Hierarchical Reconstructionwith Upto Second Degree Remainderfor Sol | ADA589175 | 17 | 577e3e5b899e | low |
| ADA589371_LocalDiscontinuousGalerkinMethodsfortheGeneralizedZakharovSyst.pdf | Local Discontinuous Galerkin Methodsforthe Generalized Zakharov Syst | ADA589371 | 25 | fde62bccc571 | low |
| ADA598221_Magneto-Rayleigh-TaylorInstabilityExperimentsonabDensebZ-Pi.pdf | Magneto Rayleigh Taylor Insta ility Experimentsona Dense Z Pi | ADA598221 | 3 | b55fe7702231 | low |
| AD1302801_MeasurementsandApplicationsofStronglyCorrelatedbPlasmasbGe.pdf | Measurementsand Applicationsof Strongly Correlated Plasmas Ge | AD1302801 | 150 | 41f88be94ca2 | low |
| Wave 5/AD1001263_ModelingofInelasticCollisionsinaMultifluidPlasmaExcitationan.pdf | Modelingof Inelastic Collisionsina Multifluid Plasma Excitationan | AD1001263 | 40 | c90bc64ecd07 | low |
| Wave2/AD1036184_Multi-scaleandmulti-bphysicsbbsimulationsbusingthemult.pdf | Multi scaleandmulti physics simulations usingthemult | AD1036184 | 32 | 3bf9895a5579 | low |
| Wave2/AD1326156_PREDICTIVEANDPRACTICALbSIMULATIONSbOFbPLASMAbSYSTEMSA.pdf | PREDICTIVEANDPRACTICAL SIMULATIONS OF PLASMA SYSTEMSA | AD1326156 | 25 | f40a112610c9 | low |
| Wave2/AD1097132_Physics-Based-AdaptivebPlasmabModelforHigh-FidelityNumerical.pdf | Physics Based Adaptive Plasma Modelfor High Fidelity Numerical | AD1097132 | 25 | 68a8c2e87812 | low |
| Wave 5/AD1330276_PhysicsandApplicationsofDustyPlasmasThePerspectives2023.pdf | Physicsand Applicationsof Dusty Plasmas The Perspectives2023 | AD1330276 | 54 | 070a82829ee9 | low |
| AD1302183_ReducingtheComputationalComplexityofImplicitSchemesintheModel.pdf | Reducing the computational complexity of implicit schemes in the modeling of kinetic inelastic collisions in a partially ionized plasma | AD1302183 | 25 | 62b3fa352371 | low |
| Wave4/AD1302183_ReducingtheComputationalComplexityofImplicitSchemesintheModel.pdf | Reducing the computational complexity of implicit schemes in the modeling of kinetic inelastic collisions in a partially ionized plasma | AD1302183 | 25 | 62b3fa352371 | low |
| AD1148035_RubricsforChargeConservingCurrentMappinginFiniteElementElectr.pdf | Ru ricsfor Charge Conserving Current Mappingin Finite Element Electr | AD1148035 | 16 | 7373140c775c | low |
| Wave2/AD1142423_TheRigid-BeamModelforbSimulatingbbPlasmasbGeneratedby.pdf | The Rigid Beam Modelfor Simulating Plasmas Generated y | AD1142423 | 11 | 8201d8a7fca1 | low |
| Wave 5/AD1340820_Thermo-HydrodynamicsofaStronglyCoupledPlasma.pdf | Thermo Hydrodynamicsofa Strongly Coupled Plasma | AD1340820 | 35 | d2b9cecde827 | low |
| Wave2/AD1105890_TowardpredictivemodelingofExBbplasmabdischarges.pdf | Towardpredictivemodelingof Ex B plasma discharges | AD1105890 | 19 | 548301391e2a | low |
| Wave4/AD1105890_TowardpredictivemodelingofExBbplasmabdischarges.pdf | Towardpredictivemodelingof Ex B plasma discharges | AD1105890 | 19 | 548301391e2a | low |
| Wave4/HDIAC-2191925_bWeightedbbEssentiallybbNonb-bOscillatorybbSc.pdf | Weighted Essentially Non-Oscillatory Schemes | HDIAC-2191925 | 32 | d14050ec8071 | low |
| Wave 5/AD1116543_2019NRLPlasmaFormulary.pdf | 2019NRLPlasma Formulary | AD1116543 | 72 | 037290d47c8f | medium |
| Wave2/AD1057898_3-DbALEGRAbSimulationsofMagneticFieldShieldingEffectsofC.pdf | 3 D ALEGRA Simulationsof Magnetic Field Shielding Effectsof C | AD1057898 | 17 | 4e04c8807de9 | medium |
| Wave2/AD1096774_ALE3D-bMHDbModelingofModifiedSqueeze5MagneticFluxCompress.pdf | ALE3D MHD Modelingof Modified Squeeze5Magnetic Flux Compress | AD1096774 | 22 | 7022ab0decd6 | medium |
| Wave2/AD1038890_ALEGRA-bMHDbSimulationsforMagnetizationofanEllipsoidalIncl.pdf | ALEGRA MHD Simulationsfor Magnetizationofan Ellipsoidal Incl | AD1038890 | 26 | e2b69c609f22 | medium |
| AD1098588_AccelerationofProtonsandDeuteronsUpto35MevandGenerationof1.pdf | Accelerationof Protonsand Deuterons Upto35Mevand Generationof1 | AD1098588 | 12 | 3f0ebd080206 | medium |
| Wave 5/AD1098588_AccelerationofProtonsandDeuteronsUpto35MevandGenerationof1.pdf | Accelerationof Protonsand Deuterons Upto35Mevand Generationof1 | AD1098588 | 12 | 3f0ebd080206 | medium |
| Wave4/Borges An improved weighted essentially non-oscillatory scheme for hyperbolic conservation laws.pdf | An Improved Weighted Essentially Non-Oscillatory Scheme for Hyperbolic Conservation Laws |  | 16 | 18aac89e24e6 | medium |
| AD1321713_DirectedHighEnergyRadiationandParticleBeamsGeneratedUsingExtr.pdf | Directed High Energy Radiationand Particle Beams Generated Using Extr | AD1321713 | 20 | f2c2e2318641 | medium |
| AD1099126_EffectsofaPreembeddedAxialMagneticFieldontheCurrentDistribut.pdf | Effects of a Preembedded Axial Magnetic Field on the Current Distribution in a Z-Pinch Implosion | AD1099126 | 5 | fed533116bb5 | medium |
| Wave2/AD1194307_EnablingReduced-OrderCollisional-RadiativeModelingforReactiveb.pdf | Ena ling Reduced Order Collisional Radiative Modelingfor Reactive | AD1194307 | 41 | 3f5ffcf29a11 | medium |
| Wave2/AD1194307_EnablingReduced-OrderCollisional-RadiativeModelingforReactivePla.pdf | Ena ling Reduced Order Collisional Radiative Modelingfor Reactive Pla | AD1194307 | 41 | 3f5ffcf29a11 | medium |
| Wave4/033302_1_5.0311453.pdf | Formation and dynamics of Z-pinch plasma in a coaxial plasma gun | 033302_1_5.0311453 | 13 | 365ec9f0b28d | medium |
| AD1101079_GuestEditorialSpecialIssueonZPinchbPlasmasb.pdf | Guest Editorial Special Issueon ZPinch Plasmas | AD1101079 | 2 | b1c47de6159d | medium |
| Wave 5/AD1095966_IonAccelerationandNeutronProductioninHybridGasPuffZPincheso.pdf | Ion Accelerationand Neutron Productionin Hy rid Gas Puff ZPincheso | AD1095966 | 11 | e1335f508ed0 | medium |
| ADA598893_KineticComputationalModelofExplosiveEmissionCenterInitiationby.pdf | Kinetic Computational Modelof Explosive Emission Center Initiation y | ADA598893 | 5 | 6429b871e056 | medium |
| Wave 5/AD1096219_LocalMeasurementsoftheSpatialMagneticFieldDistributioninaZ-P.pdf | Local Measurementsofthe Spatial Magnetic Field Distri utionina Z P | AD1096219 | 11 | 7a665849ebdf | medium |
| ADA628985_MAGIC3DElectromagneticFDTD-PICCodebDensebbPlasmabModel.pdf | MAGIC3DElectromagnetic FDTD PICCode Dense Plasma Model | ADA628985 | 5 | 12af3670b975 | medium |
| AD1180075_MitigationofMagneto-Rayleigh-TaylorInstabilityGrowthinaTriple-N.pdf | Mitigationof Magneto Rayleigh Taylor Insta ility Growthina Triple N | AD1180075 | 6 | 0f97b437ead2 | medium |
| AD1321188_NovelCollisionalbParticleb-In-bCellbCPICMethodsforKi.pdf | Novel Collisional Particle In Cell CPICMethodsfor Ki | AD1321188 | 6 | 8a90e551c6e5 | medium |
| Wave 5/AD1100651_ObservationsoftheMagneto-Rayleigh-TaylorInstabilityandShockDyna.pdf | O servationsofthe Magneto Rayleigh Taylor Insta ilityand Shock Dyna | AD1100651 | 10 | 912060223b28 | medium |
| ADA606395_bParticlebinbCellbSimulationofPlasmaThrusters.pdf | Particle in Cell Simulationof Plasma Thrusters | ADA606395 | 7 | 778ab66ca7e9 | medium |
| AD1206892_Particle-in-CellModelingofOmegaExperimentsonAblationofbPlasm.pdf | Particle-in-Cell Modeling of Omega Experiments on Ablation of Plasmas | AD1206892 | 9 | 153394e55da3 | medium |
| AD1206892_bParticleb-in-bCellbModelingofOmegaExperimentsonAblati.pdf | Particle-in-Cell Modeling of Omega Experiments on Ablation of Plasmas | AD1206892 | 9 | 153394e55da3 | medium |
| Wave 5/AD1206892_Particle-in-CellModelingofOmegaExperimentsonAblationofbPlasm.pdf | Particle-in-Cell Modeling of Omega Experiments on Ablation of Plasmas | AD1206892 | 9 | 153394e55da3 | medium |
| AD1156411_RobustMaxwellSolversForLargeScalebParticleb-In-bCellb.pdf | Ro ust Maxwell Solvers For Large Scale Particle In Cell | AD1156411 | 138 | cffb92861d5b | medium |
| Wave2/ADA636210_bSnowplowbModelingofaLong-Conduction-TimePlasmaOpeningSwit.pdf | SNOWPLOW MODELING OF A LONG-CONDUCTION-TIME PLASMA OPENING SWITCH | ADA636210 | 5 | 7b686fb62c50 | medium |
| Wave 5/AD1194691_bSimulatingbaPulsed-Power-DrivenPlasmawithIdealbMHDb.pdf | Simulating a pulsed-power-driven plasma with ideal MHD | AD1194691 | 11 | b5b77f274da5 | medium |
| Wave2/AD1194691_SimulatingaPulsed-Power-DrivenPlasmawithIdealbMHDb.pdf | Simulating a pulsed-power-driven plasma with ideal MHD | AD1194691 | 11 | b5b77f274da5 | medium |
| Wave2/AD1194691_bSimulatingbaPulsed-Power-DrivenbPlasmabwithIdealMHD.pdf | Simulating a pulsed-power-driven plasma with ideal MHD | AD1194691 | 11 | b5b77f274da5 | medium |
| AD1095975_SpatialDistributionofIonEmissioninGas-PuffbZb-bPinches.pdf | Spatial Distri utionof Ion Emissionin Gas Puff Z Pinches | AD1095975 | 14 | 7c486e76c3d7 | medium |
| AD1095975_SpatialDistributionofIonEmissioninGas-PuffZ-PinchesandbDens.pdf | Spatial Distri utionof Ion Emissionin Gas Puff Z Pinchesand Dens | AD1095975 | 14 | 7c486e76c3d7 | medium |
| Wave 5/AD1300646_StudiesofbPlasmabSheathPhysicsusingContinuumKineticbSim.pdf | Studies of Plasma Sheath Physics using Continuum Kinetic Simulations of Plasmas | AD1300646 | 6 | e15a4b416ec0 | medium |
| Wave2/AD1300646_StudiesofbPlasmabSheathPhysicsusingContinuumKineticbSim.pdf | Studies of Plasma Sheath Physics using Continuum Kinetic Simulations of Plasmas | AD1300646 | 6 | e15a4b416ec0 | medium |
| AD1187201_TheMFiXbParticlebinbCellbMethodMFiXPICTheoryGuide.pdf | The MFi X Particle in Cell Method MFi XPICTheory Guide | AD1187201 | 32 | 8adceb05e282 | medium |

## Full Inventory

| path | title | accession | pages | sha12 | relevance | status |
| --- | --- | --- | --- | --- | --- | --- |
| AD1076777_OptimizationofDensePlasmaFocusbDPFbNeutronSourcesviaEx.pdf | Optimization of Dense Plasma Focus (DPF) Neutron Sources via Experiments and Kinetic Modeling | AD1076777 | 8 | 8b83e6b55436 | high | intake_unreviewed_not_KR |
| AD1076777_OptimizationofbDensebbPlasmabFocusDPFNeutronSources.pdf | Optimization of Dense Plasma Focus (DPF) Neutron Sources via Experiments and Kinetic Modeling | AD1076777 | 8 | 8b83e6b55436 | high | intake_unreviewed_not_KR |
| AD1079881_OptimizationofbDensebbPlasmabFocusDPFNeutronSources.pdf | Optimization of Dense Plasma Focus (DPF) Neutron Sources via Experiments and Kinetic Modeling | AD1079881 | 8 | 2c016fb54004 | high | already_in_repo_or_KR |
| AD1095975_SpatialDistributionofIonEmissioninGas-PuffZ-PinchesandbDens.pdf | Spatial Distri utionof Ion Emissionin Gas Puff Z Pinchesand Dens | AD1095975 | 14 | 7c486e76c3d7 | medium | intake_unreviewed_not_KR |
| AD1095975_SpatialDistributionofIonEmissioninGas-PuffbZb-bPinches.pdf | Spatial Distri utionof Ion Emissionin Gas Puff Z Pinches | AD1095975 | 14 | 7c486e76c3d7 | medium | intake_unreviewed_not_KR |
| AD1098588_AccelerationofProtonsandDeuteronsUpto35MevandGenerationof1.pdf | Accelerationof Protonsand Deuterons Upto35Mevand Generationof1 | AD1098588 | 12 | 3f0ebd080206 | medium | intake_unreviewed_not_KR |
| AD1099126_EffectsofaPreembeddedAxialMagneticFieldontheCurrentDistribut.pdf | Effects of a Preembedded Axial Magnetic Field on the Current Distribution in a Z-Pinch Implosion | AD1099126 | 5 | fed533116bb5 | medium | intake_unreviewed_not_KR |
| AD1100306_SimulationsofabDensebbPlasmabFocusonaHigh-Impedance.pdf | Simulations of a Dense Plasma Focus on a High-Impedance Generator | AD1100306 | 5 | 0358c24d9e71 | high | intake_unreviewed_not_KR |
| AD1101079_GuestEditorialSpecialIssueonZPinchbPlasmasb.pdf | Guest Editorial Special Issueon ZPinch Plasmas | AD1101079 | 2 | b1c47de6159d | medium | intake_unreviewed_not_KR |
| AD1123736_CharacterizationofElectronBeamsfromabDensebbPlasmabF.pdf | Characterization of Electron Beams from a Dense Plasma Focus | AD1123736 | 19 | d48f6741c06b | high | intake_unreviewed_not_KR |
| AD1148035_RubricsforChargeConservingCurrentMappinginFiniteElementElectr.pdf | Ru ricsfor Charge Conserving Current Mappingin Finite Element Electr | AD1148035 | 16 | 7373140c775c | low | intake_unreviewed_not_KR |
| AD1156411_RobustMaxwellSolversForLargeScalebParticleb-In-bCellb.pdf | Ro ust Maxwell Solvers For Large Scale Particle In Cell | AD1156411 | 138 | cffb92861d5b | medium | intake_unreviewed_not_KR |
| AD1180075_MitigationofMagneto-Rayleigh-TaylorInstabilityGrowthinaTriple-N.pdf | Mitigationof Magneto Rayleigh Taylor Insta ility Growthina Triple N | AD1180075 | 6 | 0f97b437ead2 | medium | intake_unreviewed_not_KR |
| AD1187201_TheMFiXbParticlebinbCellbMethodMFiXPICTheoryGuide.pdf | The MFi X Particle in Cell Method MFi XPICTheory Guide | AD1187201 | 32 | 8adceb05e282 | medium | intake_unreviewed_not_KR |
| AD1206892_Particle-in-CellModelingofOmegaExperimentsonAblationofbPlasm.pdf | Particle-in-Cell Modeling of Omega Experiments on Ablation of Plasmas | AD1206892 | 9 | 153394e55da3 | medium | intake_unreviewed_not_KR |
| AD1206892_bParticleb-in-bCellbModelingofOmegaExperimentsonAblati.pdf | Particle-in-Cell Modeling of Omega Experiments on Ablation of Plasmas | AD1206892 | 9 | 153394e55da3 | medium | intake_unreviewed_not_KR |
| AD1302183_ReducingtheComputationalComplexityofImplicitSchemesintheModel.pdf | Reducing the computational complexity of implicit schemes in the modeling of kinetic inelastic collisions in a partially ionized plasma | AD1302183 | 25 | 62b3fa352371 | low | intake_unreviewed_not_KR |
| AD1302801_MeasurementsandApplicationsofStronglyCorrelatedbPlasmasbGe.pdf | Measurementsand Applicationsof Strongly Correlated Plasmas Ge | AD1302801 | 150 | 41f88be94ca2 | low | intake_unreviewed_not_KR |
| AD1321188_NovelCollisionalbParticleb-In-bCellbCPICMethodsforKi.pdf | Novel Collisional Particle In Cell CPICMethodsfor Ki | AD1321188 | 6 | 8a90e551c6e5 | medium | intake_unreviewed_not_KR |
| AD1321713_DirectedHighEnergyRadiationandParticleBeamsGeneratedUsingExtr.pdf | Directed High Energy Radiationand Particle Beams Generated Using Extr | AD1321713 | 20 | f2c2e2318641 | medium | intake_unreviewed_not_KR |
| AD1338534_AssemblingaDeep-HistoryBinaryCorpus (1).pdf | Assem linga Deep History Binary Corpus (1) | AD1338534 | 13 | c01698b17c7e | low | intake_unreviewed_not_KR |
| AD1345078_EffectofCurrentSheathInitiationontheRadialCollapseandEnerget.pdf | Effect of current sheath initiation on the radial collapse and energetic particle acceleration in 10 kJ Dense Plasma Focus | AD1345078 | 22 | b2e95b882b6a | high | intake_unreviewed_not_KR |
| ADA433824_AdvancementsinDensePlasmaFocusbDPFbforSpacePropulsion.pdf | Advancements in Dense Plasma Focus (DPF) for Space Propulsion | ADA433824 | 9 | 6e45eca49152 | high | intake_unreviewed_not_KR |
| ADA454652_AnInvestigationofBremsstrahlungReflectioninaDensePlasmaFocus.pdf | An Investigation of Bremsstrahlung Reflection in a Dense Plasma Focus | ADA454652 | 9 | f17533f2f1c9 | high | intake_unreviewed_not_KR |
| ADA589175_HierarchicalReconstructionwithUptoSecondDegreeRemainderforSol.pdf | Hierarchical Reconstructionwith Upto Second Degree Remainderfor Sol | ADA589175 | 17 | 577e3e5b899e | low | intake_unreviewed_not_KR |
| ADA589371_LocalDiscontinuousGalerkinMethodsfortheGeneralizedZakharovSyst.pdf | Local Discontinuous Galerkin Methodsforthe Generalized Zakharov Syst | ADA589371 | 25 | fde62bccc571 | low | intake_unreviewed_not_KR |
| ADA598221_Magneto-Rayleigh-TaylorInstabilityExperimentsonabDensebZ-Pi.pdf | Magneto Rayleigh Taylor Insta ility Experimentsona Dense Z Pi | ADA598221 | 3 | b55fe7702231 | low | intake_unreviewed_not_KR |
| ADA598893_KineticComputationalModelofExplosiveEmissionCenterInitiationby.pdf | Kinetic Computational Modelof Explosive Emission Center Initiation y | ADA598893 | 5 | 6429b871e056 | medium | intake_unreviewed_not_KR |
| ADA599854_ElectrodynamicPropertiesofbDensebSemiclassicalbPlasmab.pdf | Electrodynamic Propertiesof Dense Semiclassical Plasma | ADA599854 | 3 | 9a31f386d8a5 | low | intake_unreviewed_not_KR |
| ADA606395_bParticlebinbCellbSimulationofPlasmaThrusters.pdf | Particle in Cell Simulationof Plasma Thrusters | ADA606395 | 7 | 778ab66ca7e9 | medium | intake_unreviewed_not_KR |
| ADA628985_MAGIC3DElectromagneticFDTD-PICCodebDensebbPlasmabModel.pdf | MAGIC3DElectromagnetic FDTD PICCode Dense Plasma Model | ADA628985 | 5 | 12af3670b975 | medium | intake_unreviewed_not_KR |
| Wave 5/1169854.pdf |  | 1169854 | 15 | 3f439245a587 | high | intake_unreviewed_not_KR |
| Wave 5/AD1001263_ModelingofInelasticCollisionsinaMultifluidPlasmaExcitationan.pdf | Modelingof Inelastic Collisionsina Multifluid Plasma Excitationan | AD1001263 | 40 | c90bc64ecd07 | low | intake_unreviewed_not_KR |
| Wave 5/AD1076777_OptimizationofbDensebbPlasmabbFocusbDPFNeutron.pdf | Optimization of Dense Plasma Focus (DPF) Neutron Sources via Experiments and Kinetic Modeling | AD1076777 | 8 | 8b83e6b55436 | high | intake_unreviewed_not_KR |
| Wave 5/AD1095966_IonAccelerationandNeutronProductioninHybridGasPuffZPincheso.pdf | Ion Accelerationand Neutron Productionin Hy rid Gas Puff ZPincheso | AD1095966 | 11 | e1335f508ed0 | medium | intake_unreviewed_not_KR |
| Wave 5/AD1096219_LocalMeasurementsoftheSpatialMagneticFieldDistributioninaZ-P.pdf | Local Measurementsofthe Spatial Magnetic Field Distri utionina Z P | AD1096219 | 11 | 7a665849ebdf | medium | intake_unreviewed_not_KR |
| Wave 5/AD1098588_AccelerationofProtonsandDeuteronsUpto35MevandGenerationof1.pdf | Accelerationof Protonsand Deuterons Upto35Mevand Generationof1 | AD1098588 | 12 | 3f0ebd080206 | medium | intake_unreviewed_not_KR |
| Wave 5/AD1100306_bSimulationsbofaDensePlasmaFocusonaHigh-ImpedanceGenerat.pdf | Simulations of a Dense Plasma Focus on a High-Impedance Generator | AD1100306 | 5 | 0358c24d9e71 | high | intake_unreviewed_not_KR |
| Wave 5/AD1100651_ObservationsoftheMagneto-Rayleigh-TaylorInstabilityandShockDyna.pdf | O servationsofthe Magneto Rayleigh Taylor Insta ilityand Shock Dyna | AD1100651 | 10 | 912060223b28 | medium | intake_unreviewed_not_KR |
| Wave 5/AD1116543_2019NRLPlasmaFormulary.pdf | 2019NRLPlasma Formulary | AD1116543 | 72 | 037290d47c8f | medium | intake_unreviewed_not_KR |
| Wave 5/AD1123736_CharacterizationofElectronBeamsfromaDensePlasmaFocus.pdf | Characterization of Electron Beams from a Dense Plasma Focus | AD1123736 | 19 | d48f6741c06b | high | intake_unreviewed_not_KR |
| Wave 5/AD1194691_bSimulatingbaPulsed-Power-DrivenPlasmawithIdealbMHDb.pdf | Simulating a pulsed-power-driven plasma with ideal MHD | AD1194691 | 11 | b5b77f274da5 | medium | intake_unreviewed_not_KR |
| Wave 5/AD1206892_Particle-in-CellModelingofOmegaExperimentsonAblationofbPlasm.pdf | Particle-in-Cell Modeling of Omega Experiments on Ablation of Plasmas | AD1206892 | 9 | 153394e55da3 | medium | intake_unreviewed_not_KR |
| Wave 5/AD1300646_StudiesofbPlasmabSheathPhysicsusingContinuumKineticbSim.pdf | Studies of Plasma Sheath Physics using Continuum Kinetic Simulations of Plasmas | AD1300646 | 6 | e15a4b416ec0 | medium | intake_unreviewed_not_KR |
| Wave 5/AD1330276_PhysicsandApplicationsofDustyPlasmasThePerspectives2023.pdf | Physicsand Applicationsof Dusty Plasmas The Perspectives2023 | AD1330276 | 54 | 070a82829ee9 | low | intake_unreviewed_not_KR |
| Wave 5/AD1340820_Thermo-HydrodynamicsofaStronglyCoupledPlasma.pdf | Thermo Hydrodynamicsofa Strongly Coupled Plasma | AD1340820 | 35 | d2b9cecde827 | low | intake_unreviewed_not_KR |
| Wave2/AD1036184_Multi-scaleandmulti-bphysicsbbsimulationsbusingthemult.pdf | Multi scaleandmulti physics simulations usingthemult | AD1036184 | 32 | 3bf9895a5579 | low | intake_unreviewed_not_KR |
| Wave2/AD1038890_ALEGRA-bMHDbSimulationsforMagnetizationofanEllipsoidalIncl.pdf | ALEGRA MHD Simulationsfor Magnetizationofan Ellipsoidal Incl | AD1038890 | 26 | e2b69c609f22 | medium | intake_unreviewed_not_KR |
| Wave2/AD1057898_3-DbALEGRAbSimulationsofMagneticFieldShieldingEffectsofC.pdf | 3 D ALEGRA Simulationsof Magnetic Field Shielding Effectsof C | AD1057898 | 17 | 4e04c8807de9 | medium | intake_unreviewed_not_KR |
| Wave2/AD1096774_ALE3D-bMHDbModelingofModifiedSqueeze5MagneticFluxCompress.pdf | ALE3D MHD Modelingof Modified Squeeze5Magnetic Flux Compress | AD1096774 | 22 | 7022ab0decd6 | medium | intake_unreviewed_not_KR |
| Wave2/AD1097132_Physics-Based-AdaptivebPlasmabModelforHigh-FidelityNumerical.pdf | Physics Based Adaptive Plasma Modelfor High Fidelity Numerical | AD1097132 | 25 | 68a8c2e87812 | low | intake_unreviewed_not_KR |
| Wave2/AD1105890_TowardpredictivemodelingofExBbplasmabdischarges.pdf | Towardpredictivemodelingof Ex B plasma discharges | AD1105890 | 19 | 548301391e2a | low | intake_unreviewed_not_KR |
| Wave2/AD1142423_TheRigid-BeamModelforbSimulatingbbPlasmasbGeneratedby.pdf | The Rigid Beam Modelfor Simulating Plasmas Generated y | AD1142423 | 11 | 8201d8a7fca1 | low | intake_unreviewed_not_KR |
| Wave2/AD1194307_EnablingReduced-OrderCollisional-RadiativeModelingforReactivePla.pdf | Ena ling Reduced Order Collisional Radiative Modelingfor Reactive Pla | AD1194307 | 41 | 3f5ffcf29a11 | medium | intake_unreviewed_not_KR |
| Wave2/AD1194307_EnablingReduced-OrderCollisional-RadiativeModelingforReactiveb.pdf | Ena ling Reduced Order Collisional Radiative Modelingfor Reactive | AD1194307 | 41 | 3f5ffcf29a11 | medium | intake_unreviewed_not_KR |
| Wave2/AD1194691_SimulatingaPulsed-Power-DrivenPlasmawithIdealbMHDb.pdf | Simulating a pulsed-power-driven plasma with ideal MHD | AD1194691 | 11 | b5b77f274da5 | medium | intake_unreviewed_not_KR |
| Wave2/AD1194691_bSimulatingbaPulsed-Power-DrivenbPlasmabwithIdealMHD.pdf | Simulating a pulsed-power-driven plasma with ideal MHD | AD1194691 | 11 | b5b77f274da5 | medium | intake_unreviewed_not_KR |
| Wave2/AD1300646_StudiesofbPlasmabSheathPhysicsusingContinuumKineticbSim.pdf | Studies of Plasma Sheath Physics using Continuum Kinetic Simulations of Plasmas | AD1300646 | 6 | e15a4b416ec0 | medium | intake_unreviewed_not_KR |
| Wave2/AD1326156_PREDICTIVEANDPRACTICALbSIMULATIONSbOFbPLASMAbSYSTEMSA.pdf | PREDICTIVEANDPRACTICAL SIMULATIONS OF PLASMA SYSTEMSA | AD1326156 | 25 | f40a112610c9 | low | intake_unreviewed_not_KR |
| Wave2/AD1331701_ExperimentModelingandbSimulationbofAdvancedMaterials-b.pdf | Experiment Modelingand Simulation of Advanced Materials | AD1331701 | 18 | dc4703eb8af1 | low | intake_unreviewed_not_KR |
| Wave2/AD1334827_AHybridModelforMultiscaleLaserbPlasmabbSimulationsbw.pdf | AHy rid Modelfor Multiscale Laser Plasma Simulations w | AD1334827 | 20 | 69b33ebe2e88 | low | intake_unreviewed_not_KR |
| Wave2/AD1337397_Electronemissionbphysicsbbsimulationsb.pdf | Electronemission physics simulations | AD1337397 | 16 | 4c71fbe527f7 | low | intake_unreviewed_not_KR |
| Wave2/AD1338034_ExperimentModelingandbSimulationbofAdvancedMaterials-b.pdf | Experiment Modelingand Simulation of Advanced Materials | AD1338034 | 18 | ea1817c6d504 | low | intake_unreviewed_not_KR |
| Wave2/ADA462260_DeviceDemonstration.pdf | Device Demonstration | ADA462260 | 190 | 72fd2b544e20 | low | intake_unreviewed_not_KR |
| Wave2/ADA610142_Characterizationofa500JbDensebbPlasmabbFocusbfor.pdf | CHARACTERIZATION OF A 500J DENSE PLASMA FOCUS FOR PRODUCING SOFT X-RAYS | ADA610142 | 5 | db2349b11e58 | high | intake_unreviewed_not_KR |
| Wave2/ADA635195_PFMA-1A1-Hz150-kJPulsedPowerSystemforPlasmaFocusGeneration.pdf | PFMA-1: A 1-Hz, 150-kJ PULSED POWER SYSTEM FOR PLASMA FOCUS GENERATION | ADA635195 | 6 | 9590eb6507db | high | intake_unreviewed_not_KR |
| Wave2/ADA635937_2DMHDComputerModelingofbDensebbPlasmabbFocusbAc.pdf | 2D MHD COMPUTER MODELING OF DENSE PLASMA FOCUS ACCELERATORS | ADA635937 | 6 | 74d0df14b491 | high | intake_unreviewed_not_KR |
| Wave2/ADA636210_bSnowplowbModelingofaLong-Conduction-TimePlasmaOpeningSwit.pdf | SNOWPLOW MODELING OF A LONG-CONDUCTION-TIME PLASMA OPENING SWITCH | ADA636210 | 5 | 7b686fb62c50 | medium | intake_unreviewed_not_KR |
| Wave2/DSIAC-2195387_ClassicalTrajectoryMonteCarlobSimulationbofbPlasmabFu.pdf | Classical Trajectory Monte Carlo Simulation of Plasma Fu | DSIAC-2195387 | 9 | 31246535f2e0 | low | intake_unreviewed_not_KR |
| Wave4/033302_1_5.0311453.pdf | Formation and dynamics of Z-pinch plasma in a coaxial plasma gun | 033302_1_5.0311453 | 13 | 365ec9f0b28d | medium | intake_unreviewed_not_KR |
| Wave4/033302_1_5.0311453.pdf.crdownload | 033302 1 5.0311453 | 033302_1_5.0311453 |  | 365ec9f0b28d | low | intake_unreviewed_not_KR |
| Wave4/AD1079881_OptimizationofDensebPlasmabFocusDPFNeutronSourcesviaEx.pdf | Optimization of Dense Plasma Focus (DPF) Neutron Sources via Experiments and Kinetic Modeling | AD1079881 | 8 | 2c016fb54004 | high | already_in_repo_or_KR |
| Wave4/AD1105890_TowardpredictivemodelingofExBbplasmabdischarges.pdf | Towardpredictivemodelingof Ex B plasma discharges | AD1105890 | 19 | 548301391e2a | low | intake_unreviewed_not_KR |
| Wave4/AD1230244_DevelopingMethodsofControlofSelf-OrganizedbPlasmabStructur.pdf | Developing Methodsof Controlof Self Organized Plasma Structur | AD1230244 | 17 | 6d5767901062 | low | intake_unreviewed_not_KR |
| Wave4/AD1302183_ReducingtheComputationalComplexityofImplicitSchemesintheModel.pdf | Reducing the computational complexity of implicit schemes in the modeling of kinetic inelastic collisions in a partially ionized plasma | AD1302183 | 25 | 62b3fa352371 | low | intake_unreviewed_not_KR |
| Wave4/Borges An improved weighted essentially non-oscillatory scheme for hyperbolic conservation laws.pdf | An Improved Weighted Essentially Non-Oscillatory Scheme for Hyperbolic Conservation Laws |  | 16 | 18aac89e24e6 | medium | intake_unreviewed_not_KR |
| Wave4/HDIAC-2191925_bWeightedbbEssentiallybbNonb-bOscillatorybbSc.pdf | Weighted Essentially Non-Oscillatory Schemes | HDIAC-2191925 | 32 | d14050ec8071 | low | intake_unreviewed_not_KR |

## Notes

- `033302_1_5.0311453.pdf.crdownload` has the same SHA-256 payload as `033302_1_5.0311453.pdf`; it appears to be a completed duplicate with an unfinished browser suffix.
- Title matching is conservative and should be followed by manual KR intake review before moving any item into source-backed status.
- Duplicate status here is based on exact SHA-256 payloads and likely title matches; no files were deleted or moved.
