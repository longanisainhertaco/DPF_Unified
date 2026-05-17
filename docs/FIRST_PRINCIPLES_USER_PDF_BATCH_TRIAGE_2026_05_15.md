# User PDF Batch Triage For First-Principles DPF

Date: 2026-05-15

Scope: usefulness review for the eight user-supplied PDFs in
`/Users/anthonyzamora/Downloads`. This is source triage only. It does not
promote any target, curve, table, formula, uncertainty value, or simulation
result to accepted first-principles evidence.

User validation update: on 2026-05-15 the user confirmed that all eight PDFs in
this batch are verified validated research papers. That updates source status
for the batch. It does not accept extracted targets, curves, formulas,
uncertainties, simulation outputs, or whole-shot first-principles claims.

Intermediate extraction files are under `tmp/pdfs/may15_batch/`.

First-principles code binding: the non-promoting helper
`src/dpf/first_principles/source_targets.py` now exposes structured source
packets for this batch through `may15_user_validated_source_targets()`. The
helper is source-target plumbing only; every packet keeps whole-shot and
validation acceptance closed. The package-native deck module also exposes
`may15_second_scope_engineering_decks()`, which builds runnable non-promoting
3D engineering candidates for IR-MPF-100, the compact Chinese Mather DPF, and
the Willenborg/Hendricks startup-design device.

## Verdict

The batch is useful. It does not close the full-shot first-principles simulator,
but it strengthens the source queue in four concrete areas:

- second-scope device decks and target extraction;
- startup/breakdown and engineering-design constraints;
- 2D/3D MHD plus PIC-startup handoff architecture;
- scaling-law and generalization constraints.

The strongest immediate actions are:

1. Treat all eight PDFs as user-validated source-truth inputs.
2. Use the promotion report
   `docs/USER_VALIDATED_PDF_KR_PROMOTION_2026_05_15.md` / `.json` as the
   source-ingestion ledger: six PDFs were promoted into parity-passed local
   `KnowledgeReference` records and two were already represented locally.
3. Move next to typed target extraction and gate binding. Do not code against
   untyped values, figure impressions, or unstated uncertainties.
4. Use the Arnab fluids/plasmas textbook only as method/derivation support, not
   DPF validation evidence.
5. Use the Gribkov applications paper as mechanism/source-map context and
   application/beam-current material, not as a whole-shot validation packet.

## Paper-by-Paper Triage

### 1. `s41598-025-07939-x.pdf`

Title: Experimental results and analysis of plasma dynamics and radiation output
of the 100 kV dense plasma focus FAETON-I

SHA-256: `b02a711d90395e9c9ee3d4c4ab4d11c61db7be2c28a4b2b50578ee85ad0bb2e5`

Status: user-validated and already represented in `KnowledgeReference/`:

- `KnowledgeReference/faeton-i-investigation-of-plasma-dynamics-and-radiation-output-of-a-100-kv-plasma-focus-device.md`
- `KnowledgeReference/experimental-results-and-analysis-of-plasma-dynamics-and-radiation-output-of-the-100-kv-dense-5.md`

Usefulness: P1 for second-scope/generalization and high-voltage DPF behavior.

Useful content:

- FAETON-I deck: 100 kV, 125 kJ, about 1 MA, about 3.7 microsecond current
  rise, 25 microF bank, about 220 nH static inductance, 10-40 Torr operating
  range, 5 cm anode radius, 17 cm effective anode length, 10.6 cm cathode
  radius, 6.5 cm MACOR insulator, 5 cm anode-cathode gap.
- Diagnostics: current probes, Rogowski coils, voltage probe, bubble detectors,
  silver activation counters, PMT scintillators, Faraday cup.
- Shot data: consistent about `2.5e10` D-D neutrons/shot, peak `8e10`, peak
  dynamics-induced voltage up to 194 kV, about 350 keV deuterons, anisotropy
  factor about 1.6, neutron energy about `2.5 +/- 0.3 MeV`.
- Physics implications: high-voltage re-strike behavior, pre-stagnation
  dynamics-induced voltage, and beam-target timing are important blockers for
  first-principles neutron authority.

Use rules:

- Good source for a declared FAETON-I second-scope packet.
- Good target-extraction candidate for current, voltage, neutron yield,
  re-strike timing, and detector geometry.
- Do not use its Lee-code fitted `fcr`, `fmr`, or empirical multiplier as active
  first-principles closure. Those remain baseline/comparator metadata.

Does not close:

- first-principles startup BVP;
- accepted mechanism-separated neutron authority;
- accepted comparator/UQ packet;
- general DPF-machine certificate.

### 2. `salehizadeh2012.pdf`

Title: Preliminary Results of the 115 kJ Dense Plasma Focus Device IR-MPF-100

SHA-256: `f7894f85fd4d1826a5d98933453bd09664e260d46a2c9fedc4ce79491d2be4ad`

Status: user-validated and promoted to local `KnowledgeReference`:

- `KnowledgeReference/original-research-f7894f85.md`
- `KnowledgeReference/original-research-f7894f85.json`

Usefulness: P1 source for device-deck extraction and second-scope
engineering targets.

Useful content:

- IR-MPF-100 deck: 144 microF bank, maximum 40 kV, 115 kJ maximum energy, total
  inductance 120 nH, theoretical current 1.224 MA, anode radius 6.25 cm,
  anode length 22 cm, cathode radius 10.2 cm, twelve 12 mm cathode rods,
  insulator length 5 cm, design pressure 7.7 Torr.
- Measured examples: current, voltage, hard X-ray, current-derivative
  waveforms; 1.9 Torr deuterium and 20 kV shot context; neutron activation
  counter at 130 cm from anode top.
- Preliminary neutron result: about `1.5e9` n/shot at 29 kJ and 1.9 Torr
  deuterium, with about `1e9` n/shot at 65 kJ in the abstract and expectation
  of higher full-energy yield.
- Double-pinch observation at 0.3 Torr deuterium and 20 kV.

Use rules:

- Use the promoted KR record for typed deck and waveform/diagnostic target
  extraction.
- Do not code against untyped values or preliminary figure reads.
- Treat Lee/snowplow design formulas as design context only, not active
  first-principles closure.

Does not close:

- complete startup BVP;
- full spatial field/temperature history;
- mechanism-separated neutron history;
- detector-response/UQ certificate.

### 3. `1000284.pdf`

Title: ALEGRA-HEDP Simulations of the Dense Plasma Focus

SHA-256: `b93aec67a34ed9cd63176dc3fdf404df4aa29ff16cf8807eb68568ed1dbc0f9c`

Status: user-validated. Already represented in `KnowledgeReference/`, and the
exact user batch payload was also promoted as a parity-passed local source:

- `KnowledgeReference/unlimited-release-printed-september-2009-alegra-hedp-simulations-of-the-dense-plasma-focus.md`
- `KnowledgeReference/sand2009-6373-b93aec67.md`
- `KnowledgeReference/sand2009-6373-b93aec67.json`

Usefulness: P0/P1 architecture and blocker source.

Useful content:

- 2D simulations of Bernard Long, Bernard Short, and Tallboy using
  ALEGRA-HEDP.
- Explicit code capabilities: hydrodynamics, magnetics, thermal conduction,
  radiation, EOS, conductivity, lumped circuit, neutron yield/time-of-flight
  diagnostics, and 2D/3D operation.
- Explicit first-principles blocker: ALEGRA import of LSP/PIC initial gas
  breakdown is presented as the path toward simulations from voltage discharge
  through pinch without assumed early sheath structure.
- Explicit dimensionality blocker: 2D cathode treatment and residual plasma
  path can change current/rundown; 3D is required for faithful cathode-bar and
  pinch structure.
- Explicit neutron-authority blocker: MHD thermonuclear yield is far below
  measured total yield; nonthermal mechanisms require kinetic/particle physics.

Use rules:

- Keep as a primary source for startup-BVP, dimensionality, numerical-fidelity,
  EOS/conductivity/radiation, and kinetic-handoff gate definitions.
- Extract the Bernard/Tallboy table as target/context evidence, but do not use
  ALEGRA thermonuclear-only neutron output as total-yield authority.

Does not close:

- accepted full-shot 3D simulator;
- accepted neutron total-yield authority;
- same-scope PF-1000/Akel packet.

### 4. `Development of dense plasma focus device.pdf`

Working title: Development of a dense plasma focus device

SHA-256: `d1758d55ea9a32f6edb17107a86b033d8078cad337f0531ca10f18190fb220b5`

Status: user-validated and promoted to local `KnowledgeReference`:

- `KnowledgeReference/high-power-laser-and-particle-beams-d1758d55.md`
- `KnowledgeReference/high-power-laser-and-particle-beams-d1758d55.json`

Text/rendering is usable with extraction warnings. Target extraction should
still include visual table review and translation checks where needed.

Usefulness: P1/P2 source for a compact pulsed-neutron-source DPF deck.

Useful content identified from text and rendered page checks:

- Mather-type chamber for pulsed neutron source.
- Four 10 microF capacitors, 10-20 kV charging, about 400 kA delivered current.
- Chosen anode radius about 17 mm.
- Cathode: eight oxygen-free copper rods, 8 mm diameter, placed on an 88 mm
  diameter circle.
- Insulator: alumina ceramic, approximately 36 mm inner diameter, 46 mm outer
  diameter, 40-45 mm exposed length depending on reference point.
- Simulated current waveform at 20 kV with focus near quarter-cycle current
  maximum; simulated pressure-yield curve with maximum near 1 kPa and about
  `6.45e8` D-D neutrons/pulse.
- Experimental optimum deuterium pressure about 550-600 Pa; charging voltage
  above 19 kV gives average yield greater than `5.0e8` D-D neutrons/pulse;
  neutron pulse FWHM about `40 +/- 5 ns`; focus time about 1.8 microsecond.

Use rules:

- Good second-scope deck source after typed target extraction with visual and
  translation checks where needed.
- Good target candidate for pressure-yield curve, current waveform, neutron
  TOF/FWHM, and detector geometry.
- Do not use the S. Lee design calculation as active first-principles closure.

Does not close:

- startup BVP;
- mechanism-separated neutron authority;
- field/temperature history;
- UQ certificate.

### 5. `2307.07715v1.pdf`

Title: On the failure of neutron yield scaling in the Dense Plasma Focus

SHA-256: `0f7f5a0deae96f074d37e192ed2825cbaaa8407b510d27dd3887e8675402b244`

Status: user-validated and already represented in `KnowledgeReference/`:

- `KnowledgeReference/on-the-failure-of-neutron-yield-scaling-in-the-dense-plasma-focus-s-k-h-auluck-international.md`

Usefulness: P1 for scaling/generalization constraints, not for direct shot
validation.

Useful content:

- Identifies three failure modes: pinch-current limitation, neutron-scaling
  failure after pinch formation, and failure to form a pinch.
- Connects the drive parameter and scaling constraints to conservation of mass,
  momentum, and energy.
- Frames the Generalized Plasma Focus approach and the separation between
  device scaling parameters and dimensionless plasma dynamics.
- Suggests lift-off time and drive-parameter checks as experimentally testable
  indicators.

Use rules:

- Use to define generalization sanity gates and nondimensional scale-transition
  review.
- Do not use the scaling law as a predictive closure or acceptance substitute.

Does not close:

- any same-scope validation packet;
- whole-shot first-principles runtime;
- neutron authority.

### 6. `Arnab-the-physics-of-fluids-and-plasmas_compress.pdf`

Title: The Physics of Fluids and Plasmas: An Introduction for Astrophysicists

SHA-256: `eef02f492c5dec82c501fb5040f1fc3bdaa5603ab3a16bdf58dfe909caa16c70`

Status: user-validated and promoted as a method/reference source:

- `KnowledgeReference/the-physics-of-fluids-and-plasmas-eef02f49.md`
- `KnowledgeReference/the-physics-of-fluids-and-plasmas-eef02f49.json`
- `KnowledgeReference/chunks/the-physics-of-fluids-and-plasmas-eef02f49/`

Usefulness: P3 method/reference support.

Useful content:

- Boltzmann and Vlasov hierarchy.
- Moment equations, transport, collisions, diffusion.
- One-fluid MHD, weakly ionized plasma applicability, shocks, field freezing,
  resistivity, and plasma-column stability.

Use rules:

- Good for educational derivations, notation checks, and closure-review
  background.
- Not a DPF paper and not a validation target.
- Do not use for DPF geometry, shot data, thresholds, or certificate evidence.

### 7. `Dense+Plasma+Focus+physics+and+applications.pdf`

Title: Dense Plasma Focus: physics and applications

SHA-256: `ed1967114c762f608493bd4d049b627ed0d13165d435ed7d5c23efa92a93cc2a`

Status: user-validated and promoted to local `KnowledgeReference`:

- `KnowledgeReference/open-access-proceedings-journal-of-physics-conference-series-ed196711.md`
- `KnowledgeReference/open-access-proceedings-journal-of-physics-conference-series-ed196711.json`

Usefulness: P2 mechanism/source-map and application context.

Useful content:

- Current-abruption/plasma-diode view of DPF late phase.
- Fast electron/fast ion current estimates from X-ray and neutron data.
- PF-1000/PF-6/Bora application examples.
- Radiation-material, neutron/gamma, and detector/application contexts.
- Large-DPF matching concerns: transition from simple C-L-R circuitry to
  MHD plus telegraph-equation/MITL-like treatment for very large devices.
- Scaling discussion for neutron and hard X-ray production.

Use rules:

- Use as mechanism/source-map context and to define open questions for
  beam/current-abruption physics.
- Use the promoted KR record for typed mechanism, diagnostic, and application
  context extraction.
- Treat large-device and miniature-device extrapolations as speculative until
  tied to accepted sources and engineering evidence.
- Do not use as a whole-shot validation packet.

### 8. `ADA037245.pdf`

Title: Design and Construction of a Dense Plasma Focus Device

SHA-256: `12205ba4bb0d1edc11b069dda4e0e084b89597a8f14ff61c3a65e0b712926a75`

Status: user-validated and promoted to local `KnowledgeReference`:

- `KnowledgeReference/design-and-construction-of-a-dense-plasma-focus-device-12205ba4.md`
- `KnowledgeReference/design-and-construction-of-a-dense-plasma-focus-device-12205ba4.json`

Usefulness: P1 engineering and startup/breakdown design source.

Useful content:

- Detailed Mather-type DPF design/construction report.
- Breaks out inner electrode, outer electrode, spacer, insulator, switch,
  capacitor bank, vacuum/gas, voltage and current diagnostics, and X-ray
  measurement design.
- Circuit/deck context: 43.5 microF bank, 8.7 kJ at 20 kV, three 14.5 microF
  capacitors, measured/managed capacitor/switch inductance, triggered spark gap,
  and 0.1-10 Torr working pressure range.
- Startup relevance: insulator flashover, sheath symmetry, conditioning shots,
  insulator material failures, and the relation between current waveform,
  voltage spike, focus timing, and X-ray signal.
- Diagnostics relevance: capacitive voltage divider, Rogowski loop, X-ray
  detector, grounding and high-voltage noise issues.

Use rules:

- Use the promoted KR record to extract startup/breakdown and
  diagnostic-engineering requirements.
- Use as engineering evidence and design constraints, not modern validation
  evidence.

Does not close:

- accepted first-principles startup BVP by itself;
- any same-scope PF-1000/Akel evidence group;
- neutron authority.

## Impact On First-Principles Blockers

| Blocker | Batch impact |
| --- | --- |
| Startup BVP | Helped by ALEGRA PIC-startup import source, Willenborg/Hendricks insulator-breakdown design, and Chinese/IR-MPF device details. Still blocked because none supplies a reviewed full breakdown/preionization/sheath-liftoff state packet. |
| Package-native decks | Strongly helped. FAETON-I, IR-MPF-100, Chinese DPF device, and Willenborg/Hendricks can become deck candidates. |
| Dimensionality and handoff | Strongly helped by Sandia ALEGRA-HEDP. It explicitly supports 3D need and PIC-to-MHD startup import direction. |
| Physics closure | Helped by ALEGRA EOS/conductivity/radiation framing and Arnab method background. Still blocked for accepted DPF closure packets. |
| Neutron authority | Helped by FAETON-I, IR-MPF-100, Gribkov, and ALEGRA statements that nonthermal mechanisms dominate or must be separated. Still blocked because no complete mechanism-separated same-scope history and detector/UQ packet is accepted. |
| Comparator/UQ | Helped by candidate measured current/voltage/neutron tables and diagnostics. Still blocked until typed extraction, units, metrics, tolerances, uncertainty, and review exist. |
| Generalization | Strongly helped. Adds or reinforces FAETON-I, IR-MPF-100, compact Chinese device, historical Illinois device, Bernard/Tallboy, and scaling-law source material. None is accepted as a second-scope certificate. |

## Recommended Queue Actions

1. Target-extract already-ingested FAETON-I:
   - Table 1 deck;
   - current/voltage/radial-trajectory figures;
   - Table 3 shot factors and yields;
   - neutron TOF/energy/anisotropy evidence.
2. Target-extract already-ingested ALEGRA-HEDP:
   - Table 1 Bernard/Tallboy current/yield comparison;
   - startup PIC import requirements;
   - 2D/3D limitations;
   - MHD thermonuclear-only neutron limitation.
3. Target-extract already-ingested Auluck scaling failure:
   - drive-parameter and lift-off-time constraints;
   - scale-transition/generalization review requirements.
4. Target-extract promoted Salehizadeh 2012:
   - IR-MPF-100 deck;
   - preliminary current/voltage/hard-X-ray targets;
   - neutron activation yield targets and detector geometry.
5. Target-extract promoted `ADA037245.pdf`:
   - startup/breakdown and insulator conditioning;
   - switch, circuit, and low-inductance design requirements;
   - voltage/current/X-ray diagnostic requirements.
6. Target-extract promoted Chinese DPF device-development paper:
   - compact-device deck;
   - pressure-yield curve;
   - current waveform;
   - neutron TOF/FWHM and detector context.
7. Target-extract promoted Gribkov applications paper as mechanism/source-map
   context; do not use it as a whole-shot validation packet.
8. Keep the promoted Arnab textbook as method context only; link derivations to
   solver-review requirements where useful.

## Whole-Shot Readiness

This batch does not make the simulator runnable as an accepted full-shot
first-principles tool. It gives better source material for building the missing
packets. The whole-shot gate remains closed until the startup BVP, 3D solver
fidelity, power-port coupling, closure packets, same-scope targets,
mechanism-separated neutron authority, detector response, comparator/UQ, and
certificate chain all pass with reviewed local-source evidence.
