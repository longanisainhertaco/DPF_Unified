# First-Principles Blocker Source Search - Mechanism-Separated Neutron Authority

Date: 2026-05-15

Scope: local source of truth only. Scientific claims in this note are limited to
`KnowledgeReference/` and source-truth index artifacts already in the repo.

Blocker: `FP-12`, mechanism-separated neutron authority.

Question: can PF-1000/Akel neutron evidence support accepted first-principles
total-yield authority today?

## Verdict

No. PF-1000/Akel supports scalar neutron-yield context and detector-layout text,
but it does not provide an accepted mechanism-separated neutron packet.

The current accepted-contract answer is:

- Akel 2021 describes downstream/upstream/side-on scintillators, TOF-derived
  mean neutron/deuteron energy method, two calibrated silver activation counters,
  Am-Be activation calibration, scalar yield uncertainty, and measured scalar
  yield for shot 12581.
- Akel/Lee text includes thermonuclear and beam-target model context, but this is
  reduced-model baseline material and cannot act as first-principles neutron
  authority.
- The new hybrid PIC-fluid source supports time-resolved neutron production from
  simulated ion distributions, but its own text limits the result to
  order-of-magnitude validation and it is 2D/other-scope relative to PF-1000/Akel.
- Fully kinetic and MA-class mechanism sources show why beam formation,
  beam-target fusion, spectrum broadening, anisotropy, and mechanism timing must
  be resolved or explicitly bounded before total-yield authority can pass.
- TOF tomography and PF-1000 anisotropy sources define detector-response,
  scatter-subtraction, direct/scattered neutron, and angular/spectral schema, but
  they are not accepted same-scope Akel packets.

Therefore `FP-12` remains blocked for accepted whole-shot authority. The
package-native runner must expose the candidate PIC ion-yield history while
returning a fail-closed `neutron_authority` packet until thermonuclear,
beam-target, spectrum, anisotropy, detector response, and UQ channels are all
accepted in the same scope.

## Source Answers

| Source | What it answers | What remains blocked |
| --- | --- | --- |
| `KnowledgeReference/radiation-physics-and-chemistry-188-2021-109633.md:120-131` | PF-1000/Akel detector directions, TOF energy method, silver activation total-yield measurement, Am-Be calibration, scalar yield uncertainty, and derivative-dip timing origin. | No accepted response function, time-energy spectrum, angular-yield packet, or propagated UQ. |
| `KnowledgeReference/radiation-physics-and-chemistry-188-2021-109633.md:190-215,282-288,862-889` | Lee model includes thermonuclear and beam-target yield formulas and reports computed/measured scalar yield agreement. | Reduced-model fit context is baseline comparison only; it is not resolved mechanism-separated first-principles evidence. |
| `KnowledgeReference/fully-electromagnetic-hybrid-pic-fluid-dpf-neutron-yield-acb71fa9.md:952-970,1037-1040` | Hybrid model computes neutron production rate and cumulative yield from simulated ion distributions, with a refinement check. | It is other-scope and does not close mechanism-separated validation, detector response, or same-scope PF-1000/Akel UQ. |
| `KnowledgeReference/fully-electromagnetic-hybrid-pic-fluid-dpf-neutron-yield-acb71fa9.md:1083-1089,1214-1266` | Yield is sensitive to electron-temperature closure and is framed as order-of-magnitude, with limitations from scalar electron pressure, simplified conductivity, quasineutrality, 2D geometry, and missing separate electron energy. | Current hybrid result cannot be promoted as precise total-yield authority. |
| `KnowledgeReference/fully-kinetic-simulations-of-dense-plasma-focus-z-pinch.md:34-43,68-78,126-161` | Fully kinetic simulations are needed for MeV ions and beam-target yield in low-current DPFs; fluid/hybrid can miss or underpredict nonthermal ion tails. | Does not provide PF-1000/Akel same-scope evidence; defines kinetic requirement and risk boundary. |
| `KnowledgeReference/neutron-generation-dynamics-inside-a-ma-class-dense-plasma-focus-z-pinch-5.md:39-44,409-418,433-445,551-613` | Mechanism-separated DPF neutron production can include thermonuclear stagnation plus beam-target disruption bursts, with spectrum broadening and anisotropy as mechanism signatures. | MA-class/MJOLNIR scope is not Akel; usable as mechanism schema only. |
| `KnowledgeReference/tomographic-reconstruction-of-the-neutron-time-energy-spectrum-from-a-dense-plasma-focus-b78f1154.md:32-53,337-351,390-427,518-526` | Time-energy neutron spectrum is a strong validation observable; detector distance, efficiency, area, scatter subtraction, shadow-bar scaling, sampling rate, and impulse response matter. | Detector setup is not Akel/PF-1000 same scope; usable as detector-response and spectrum-inversion schema only. |
| `KnowledgeReference/anisotropy-of-the-emission-of-dd-fusion-neutrons-caused-by-the-plasma-focus-vessel-527cc533.md:121-137,175-204,269-288` | PF-1000 anisotropy experiments use Bonner/TLD/activation/TOF detectors and MCNP direct/scattered neutron transport; direct/scattered separation is required before spectrum/anisotropy interpretation. | Full-energy PF-1000 at 450-500 kJ and 3.5 Torr is not Akel 16 kV shot 12581. |

## Required Neutron Authority Packet

An accepted first-principles neutron comparator needs these channels:

| Channel | PF-1000/Akel current state |
| --- | --- |
| Accepted thermonuclear yield history | Blocked. |
| Accepted beam-target yield history | Blocked. |
| Mechanism-separated yield channels | Blocked. |
| Ion energy distribution history | Blocked for same scope. |
| Beam angular distribution history | Blocked. |
| Beam transport and stopping model | Blocked. |
| Target-density/path-length history | Blocked. |
| D-D cross-section source and units | Needed for implementation packet. |
| Neutron timing history | Partial text context only; accepted packet blocked. |
| Neutron energy spectrum | Blocked for same scope. |
| Neutron anisotropy/angular yield | Blocked for same scope. |
| Detector response model | Blocked for same scope. |
| Activation counter response model | Partial calibration text only; accepted packet blocked. |
| Direct/scattered neutron transport | Other-scope schema only. |
| Same-scope scalar yield | Text-supported. |
| Yield uncertainty budget | Partial scalar uncertainty only; full propagated UQ blocked. |
| Electron-temperature yield sensitivity UQ | Blocked. |
| Output mapping and comparator | Blocked. |
| Source review certificate | Blocked. |

## Implementation Impact

Immediate implementation requirements:

- Emit a `neutron_authority` packet from every package-native first-principles run
  and manifest.
- Mark status
  `blocked_mechanism_separated_neutron_authority_not_available` until accepted
  thermonuclear, beam-target, spectrum, anisotropy, detector-response, and UQ
  packets exist.
- Treat current PIC ion D-D yield history as a candidate runtime diagnostic only.
- Keep Lee/Saw beam-target estimates and scalar measured-yield agreement as
  baseline/comparison context, not first-principles authority.
- Require same-scope detector response and direct/scattered transport before
  neutron spectrum or angular evidence can affect a certificate.

Next blocker to search after this one: `FP-13`, comparator and UQ matrix.
