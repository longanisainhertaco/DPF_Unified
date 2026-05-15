# First-Principles Blocker Source Search - Spatial, Field, And Temperature Evidence

Date: 2026-05-15

Scope: local source of truth only. Scientific claims in this note are limited to
`KnowledgeReference/` and source-truth index artifacts already in the repo.

Blocker: `FP-11`, accepted spatial, field, and temperature evidence.

Question: can PF-1000/Akel density, EM-field, and temperature evidence support
accepted first-principles comparators today?

## Verdict

No. PF-1000/Akel has useful scalar plasma-parameter context, but the local
source set does not contain accepted same-scope spatial density, EM-field, or
electron/ion temperature evidence for a whole-shot first-principles claim.

The current accepted-contract answer is:

- Akel 2021 Table 2 supplies Lee-output scalars for shot 12581, including
  maximum pinch density, pinch radius, pinch length, velocity scalars, and
  measured scalar yield. These are not direct experimental spatial/field/
  temperature diagnostics.
- Broader PF-1000 interferometry sources show how to measure spatial density
  and reconstruct pinch geometry, but their shot, voltage, pressure, and yield
  scopes do not match Akel 16 kV shot 12581.
- Broader PF-1000 plasma-current-sheath sources provide magnetic-probe and PCS
  structure context, but those campaigns are 20-27 kV full-energy PF-1000, not
  Akel 16 kV.
- Broader PF-1000 spectroscopy sources provide density and temperature-method
  context, but they also state electron temperature could not be estimated in
  the cited experiment and are not same-scope Akel evidence.

Therefore `FP-11` remains blocked for accepted whole-shot authority. The runner
must expose a fail-closed spatial/field/temperature packet and cannot use
other-scope PF-1000 diagnostics to promote PF-1000/Akel first-principles
validation.

## Source Answers

| Source | What it answers | What remains blocked |
| --- | --- | --- |
| `KnowledgeReference/radiation-physics-and-chemistry-188-2021-109633.md:585-607` | Shot 12581 Lee-output plasma scalars: peak axial/shock/radial velocities, maximum pinch density, pinch radius/length, computed yield, and measured scalar yield. | These are model-output scalars, not direct experimental spatial density, EM-field, or temperature histories. |
| `KnowledgeReference/sixteenframe-interferometer-for-a-study-of-a-pinch-dynamics-in-pf1000-device-f8dc9d1b.md:27-33,129-176` | Sixteen-frame interferometry can provide time-ordered density, pinch geometry, MHD-instability, x-ray/neutron correlation, and plasma-density diagnostics in PF-1000. | The example shot is a different PF-1000 scope, with different pressure/yield, and has no accepted target-extracted arrays for Akel 16 kV shot 12581. |
| `KnowledgeReference/experimental-study-of-the-structure-of-the-plasma-current-sheath-on-the-pf-1000-facility-705bcc83.md:1459-1522` | PF-1000 PCS interferograms, azimuthal magnetic-field probe data, magneto-optical probe context, current-density distributions, and electron-temperature estimate context. | The campaign uses different voltage, pressure, shot numbers, and energy; it cannot be mixed into Akel 16 kV same-scope validation. |
| `KnowledgeReference/final-stages-of-the-plasma-column-evolution-in-the-plasma-focus-pf1000-device-plasma-scien-fa128cfd.md:21-65,180-195` | PF-1000 high-speed imaging and continuum-radiation method for reconstructing spatial electron-density distributions in final plasma-column stages. | Full-energy PF-1000 conditions differ from Akel 16 kV, and extracted density fields are not accepted same-scope comparator targets. |
| `KnowledgeReference/optical-spectroscopy-of-freepropagating-plasma-and-its-interaction-with-tungsten-targets-i-3a20181e.md:119-160` | PF-1000 Stark-broadening density estimate and spectroscopy limitations; explicitly reports that electron temperatures could not be estimated in the cited experiment. | Not same-scope Akel evidence and does not close electron/ion temperature diagnostics. |
| `docs/SCIENTIFIC_CLOSURE_SOURCE_QUEUE.md:266-289` | Records spatial-temperature and uncertainty closure needs, including direct experimental temperature diagnostics and same-shot density uncertainty. | Current KR state is partial only; direct temperature and uncertainty evidence remain missing. |

## Required Spatial/Field/Temperature Packet

An accepted first-principles spatial/field/temperature comparator needs these
channels:

| Channel | PF-1000/Akel current state |
| --- | --- |
| Accepted same-scope density history | Blocked. |
| Density diagnostic geometry | Other-scope PF-1000 sources only. |
| Density registration/interpolation | Blocked. |
| Density uncertainty | Blocked for same scope. |
| Accepted same-scope magnetic-field history | Blocked. |
| Accepted same-scope electric-field history | Blocked. |
| Field probe geometry/calibration | Other-scope PF-1000 sources only. |
| Field uncertainty | Blocked. |
| Accepted same-scope electron-temperature history | Blocked. |
| Accepted same-scope ion-temperature or ion-distribution history | Blocked. |
| Temperature diagnostic model | Blocked for same scope. |
| Temperature uncertainty | Blocked. |
| Output field mapping | Blocked for accepted claim. |
| Comparator metric and tolerance | Blocked until accepted evidence/UQ exists. |
| Source review certificate | Blocked. |

## Implementation Impact

Immediate implementation requirements:

- Emit a `spatial_field_temperature` packet from every package-native
  first-principles run and manifest.
- Mark Akel density/pinch scalars as text-supported context only, not
  experimental spatial or temperature validation.
- Keep broader PF-1000 interferometry, magnetic-probe, imaging, and
  spectroscopy sources as requirement/schema material unless their exact source
  scope is selected as the accepted demonstrator.
- Require accepted same-scope density, field, temperature, output-mapping,
  comparator, and UQ packets before any spatial/field/temperature evidence can
  affect a certificate.
- Scope-gate every validation target before accepting it as density, field, or
  temperature evidence. A target with accepted status but missing/mismatched
  scope metadata remains rejected for first-principles promotion.
- Emit per-channel status, text-supported-not-acceptance channels, validation
  target scope decisions, and a cross-scope transfer-rule block so other-scope
  PF-1000 diagnostics cannot silently satisfy Akel 16 kV requirements.

## Implementation Ratchet

Implemented in this pass:

- `src/dpf/first_principles/spatial_field_temperature.py` now rejects loose
  target metadata, maps accepted same-scope target observables onto required
  acceptance channels, and exposes a per-channel
  `spatial_field_temperature_channel_status` matrix.
- The packet now carries an explicit `acceptance_gate`,
  `text_supported_not_acceptance_channels`, `validation_target_scope_decisions`,
  and `cross_scope_policy` with required transfer-rule channels.
- `tests/test_first_principles_runner.py` asserts that Lee-output scalars remain
  non-acceptance context, same-scope density/field channels remain blocked, and
  cross-scope PF-1000 diagnostic sources are schema-only.

Next blocker to search after this one: `FP-12`, mechanism-separated neutron
authority.
