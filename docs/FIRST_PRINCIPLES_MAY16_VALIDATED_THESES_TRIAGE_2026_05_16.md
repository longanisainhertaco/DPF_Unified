# May 16 Verified Thesis/PDF Triage For First-Principles DPF

Date: 2026-05-16

Scope: usefulness review for seven user-supplied verified documents:

- `/Users/anthonyzamora/Downloads/arwinderphdthesis.pdf`
- `/Users/anthonyzamora/Downloads/PhD2012AlirezaTalebitaher.pdf`
- `/Users/anthonyzamora/Downloads/sawsorheoh.pdf`
- `/Users/anthonyzamora/Downloads/A SerbanPhD1995.pdf`
- `/Users/anthonyzamora/Downloads/MSR PhD thesis.pdf`
- `/Users/anthonyzamora/Downloads/PhD2010VermaRishi.pdf`
- `/Users/anthonyzamora/Downloads/s41598-022-19764-7.pdf`

User source status: the user states these documents are verified as valid,
including the defended theses. They have now been promoted into local
`KnowledgeReference/` text-or-OCR records. The promotion ledger is
`docs/USER_VALIDATED_THESES_KR_PROMOTION_2026_05_16.md` / `.json`.
This does not promote any target, curve, table, formula, uncertainty,
simulation output, or whole-shot claim to accepted first-principles evidence.

Intermediate extraction files are under
`tmp/pdfs/may16_verified_batch/`. The Saw thesis is scanned; an OCR sidecar and
searchable PDF were created at:

- `tmp/pdfs/may16_verified_batch/sawsorheoh_ocr.txt`
- `tmp/pdfs/may16_verified_batch/sawsorheoh_ocr.pdf`

Code binding: non-promoting structured source packets are exposed through
`src/dpf/first_principles/source_targets.py::may16_validated_thesis_source_targets`.

## Verdict

The batch is useful for first-principles development and is now available as
local `KnowledgeReference/` source text, but it does not let the project run or
accept a full first-principles whole-shot DPF simulation yet.

The strongest additions are:

- FP-5 startup/rundown observables and current-step/pinch-compression context;
- FP-11 spatial density, fusion-source image, pinch geometry, and electron
  temperature target candidates;
- FP-12 mechanism-separated neutron evidence candidates through deuteron
  spectra, anisotropy, neutron energy, detector response, and source imaging;
- FP-13 detector response and Bayesian/spectroscopic UQ method candidates;
- FP-15 second-scope/generalization material across NX2, miniature repetitive
  focus devices, Serban's 3 kJ focus, and 44-machine Lee-model baseline maps.

The batch closes source availability for this document set. It does not close:

- typed table/figure/equation extraction with units and uncertainty;
- PF-1000/Akel same-scope startup, fields, temperature, neutron, and detector
  response packets;
- accepted comparator/UQ and certificate gates.

## Document Triage

| Source | SHA-256 | Useful gates | First-principles use | Non-authority boundary |
| --- | --- | --- | --- | --- |
| Arwinder Singh, `Comparative Study of Plasma Focus Machines`, PhD thesis, 2015 | `2c7a8f4bd3b4d000638e4a7bd612a63d87cf1e179ee13365bd0dded40524b08c` | FP-10, FP-15 | 44-machine deck/waveform map and Lee baseline registry. KR: `KnowledgeReference/comparative-study-of-plasma-focus-machines-2c7a8f4b.md`. | Lee 5/6-phase fits are comparator/baseline only. |
| Alireza Talebitaher, `Coded Aperture Imaging of Nuclear Fusion in the Plasma Focus Device`, PhD thesis, 2012 | `9b79429f0cc5b2b8a12e8e13c0331a61a354694bbe551eb51891a80b1d674af2` | FP-11, FP-12, FP-13, FP-15 | NX2 fusion-source imaging, CR-39 proton diagnostics, Be activation detector response, anisotropy. KR: `KnowledgeReference/coded-aperture-imaging-of-nuclear-fusion-in-the-plasma-focus-device-9b79429f.md`. | Not PF-1000/Akel same-scope neutron authority. |
| Saw Sor Heoh, `Experimental Studies of a Current-Stepped Z-Pinch`, PhD thesis, 1990 | `ad6e93b2d85363348874702c8ff55abd73ee2037eb2e5de464853c0cbb82d096` | FP-5, FP-6, FP-8 | Current-step driver, radial compression, shock-jump/Saha/gamma-varying model context. KR: `KnowledgeReference/experimental-studies-of-a-current-stepped-z-pinch-ad6e93b2.md`. | Z-pinch method/reference material, not DPF whole-shot validation. |
| Adrian Serban, `Anode Geometry and Focus Characteristics`, PhD thesis, 1995 | `5a19c05d03b4daf92dc6cdbcb53aecbd07a52db9939db82ff3f10136321fbdf1` | FP-5, FP-8, FP-11, FP-12, FP-15 | Anode geometry, sheath velocity, pinch lifetime, focus impedance, soft x-ray/neutron diagnostics. KR: `KnowledgeReference/anode-geometry-and-focus-characteristics-5a19c05d.md`. | Not same-scope PF-1000/Akel evidence. |
| Muhammad Shahid Rafique, `Compression Dynamics and Radiation Emission from a Deuterium Plasma Focus`, PhD thesis, 2000 | `1eb27545f8fbaa8798278109af2a1242eb655209617270db5b832cf6278507f5` | FP-11, FP-12, FP-13, FP-15 | Deuteron spectra, neutron energy/anisotropy, shadowgraph pinch dynamics, instability growth. KR: `KnowledgeReference/compression-dynamics-and-radiation-emission-from-a-deuterium-plasma-focus-1eb27545.md`. | Requires digitized spectra, geometry, and UQ before neutron-authority use. |
| Rishi Verma, `Construction and Optimization of Low Energy (<240J) Miniature Repetitive Plasma Focus Neutron Source`, PhD thesis, 2010 | `78b15cba0c57936cdfd24d2a8dc697abaff34f778a0f0a69ac741b80802536a5` | FP-8, FP-10, FP-12, FP-15 | FMPF decks, repetition-rate behavior, cathode/anode optimization, electrode erosion/aging context. KR: `KnowledgeReference/construction-and-optimization-of-low-energy-240j-miniature-repetitive-plasma-focus-neutron-78b15cba.md`. | Repetition/scaling data are not active first-principles closure. |
| Avaria et al., `Bayesian inference of spectrometric data and validation with numerical simulations of plasma sheath diagnostics of a plasma focus discharge`, Scientific Reports, 2022 | `9ff0186062bd335802e1aa5e204e040182cbee36a04b00c3c2832c2913b6cda4` | FP-5, FP-11, FP-13 | Stark-broadened H-alpha electron-density diagnostics, Bayesian posterior UQ, rundown sheath velocity/temperature target candidate. KR: `KnowledgeReference/bayesian-inference-of-spectrometric-data-and-validation-with-numerical-simulations-of-plas-9ff01860.md`. | 400 J hydrogen scope and CShock comparison do not validate DPF-Unified. |

## Useful Extracted Facts

- Arwinder 2015: text extraction reports that the thesis analyzed 44
  Mather-type plasma focus machines in deuterium, neon, and argon, with
  PF-1000, Speed-2, and Filippov-type examples. The useful output is a
  baseline/deck/waveform location map, not active closure.
- Talebitaher 2012: NX2 D-D fusion source imaging used D(d,p)T protons,
  CR-39, coded aperture masks, beryllium activation detectors, and MCNP5
  response calculations. Reported neutron-optimized NX2 operation is around
  `1-3e8` neutrons/shot at 1.6 kJ.
- Saw 1990: the UMCSZP experiment compared radial compression with and without
  current stepping using current/voltage signals, streak photography, and
  radial magnetic-field mapping. The model extension uses 1D shock jump
  equations, thermal EOS, Saha ionization, and caloric equations to produce a
  gamma-varying energy-balance model.
- Serban 1995: neutron-optimized operation is discussed around axial sheath
  velocities of about `10 cm/us` or less, with a 3 kJ composite-anode device
  reaching up to about `15 cm/us` and a reported 70 percent neutron-output
  increase under the optimum composite-anode case.
- Rafique 2000: deuteron spectra from `80-250 keV` correlate strongly with
  total neutron yield; average neutron energies are reported around `2.48 MeV`
  radial and `3 MeV` axial, with average anisotropy about `1.45`.
- Verma 2010: FMPF-1/2/3 provide low-energy repetitive DPF source decks and
  repetition-rate/electrode-aging material, including FMPF-1 around `1.15e6`
  neutrons/shot at 230 J, 80 kA, 5.5 mbar D2, and FMPF-3 around `1.4e7`
  neutrons/s at 10 Hz under the reported operating point.
- Avaria 2022: a 400 J hydrogen plasma focus used Stark-broadened H-alpha
  spectra and Bayesian posterior inference to estimate rundown sheath density;
  extracted text reports sheath temperature around `4-20 eV` and velocity
  around `62.5 km/s`.

## Gate Application

| Gate | Candidate sources | How to apply |
| --- | --- | --- |
| FP-5 startup BVP | Saw, Serban, Avaria | Extract current-step/shock model equations, sheath velocity targets, and rundown density/velocity observables. Keep them cross-scope until a PF-1000/Akel startup packet exists. |
| FP-6 power port | Saw | Use current-step driver and voltage/current evidence as method context only; do not replace Poynting or `J.E` power-port authority. |
| FP-8 physics closure | Saw, Serban, Verma | Extract EOS/ionization assumptions, pinch-impedance context, and electrode erosion/repetition material as closure candidates or blocker context. |
| FP-10 waveform/phase | Arwinder, Verma | Use as a map to measured waveform figures and baseline-only Lee/GV comparisons. |
| FP-11 spatial/field/temperature | Talebitaher, Serban, Rafique, Avaria | Extract source images, density profiles, pinch radii/lifetime, electron temperature traces, and diagnostic geometry. |
| FP-12 neutron authority | Talebitaher, Serban, Rafique, Verma | Extract deuteron spectra, neutron energy, anisotropy, source images, yield timing, and detector response. |
| FP-13 comparator/UQ | Talebitaher, Rafique, Avaria | Build detector-response, spectroscopic, anisotropy, and Bayesian UQ target packets after review. |
| FP-15 generalization | Arwinder, Talebitaher, Serban, Rafique, Verma | Add second-scope device/diagnostic candidates only after typed extraction and source promotion. |

## Required Next Actions

1. Extract typed tables and figures into machine-readable target packets with
   units, coordinates, uncertainty, page/figure references, and artifact hashes.
2. Keep Lee-model, GV, and other reduced-model material labeled baseline-only.
3. Bind reviewed targets into the fail-closed FP packets only when same-scope
   output mapping, metric/tolerance, UQ, and independent review are present.
