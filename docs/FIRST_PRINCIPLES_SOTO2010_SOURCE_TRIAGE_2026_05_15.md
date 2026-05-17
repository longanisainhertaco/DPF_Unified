# Soto 2010 Source Triage For First-Principles DPF

Date: 2026-05-15

Source:

- Local PDF: `downloaded_books_papers/Research Papers/2026-05-15-user-ingest/soto2010.pdf`
- KR markdown: `KnowledgeReference/studies-on-scalability-and-scaling-laws-for-the-plasma-focus-similarities-and-differences-5f680756.md`
- KR JSON: `KnowledgeReference/studies-on-scalability-and-scaling-laws-for-the-plasma-focus-similarities-and-differences-5f680756.json`
- SHA-256: `5f680756a4a1ba12192e60d2e6d773be9f5279d89e05fdf6e66e2f5260f1a7fa`
- DOI: `10.1088/0963-0252/19/5/055017`
- Ingestion status: `text_parity_extracted_review_needed`
- Validation status: `source_available_not_target_extracted`

This source is useful for first-principles development, but it does not close
the whole-shot simulator. It is an excellent machine-configuration and
cross-device scaling source, plus a source of PF-400J/PF-50J aggregate
diagnostic targets. It is not a complete same-shot time-history packet.

## What It Provides

### Machine Configuration Data

Table 1 provides CCHEN device configurations and observations:

- SPEED2: capacitance 4.16 nF equivalent, typical 150 kV, 20 nH, typical 67 kJ, typical 2.4 MA, anode radius 5.4 cm, cathode radius 11 cm, effective anode length 1.5-2.5 cm, insulator length 6.5 cm.
- SPEED4: capacitance 1.25 nF equivalent, typical 60 kV, 40 nH, typical 2.25 kJ, typical 330 kA, anode radius 1.6 cm, cathode radius 4.5 cm, effective anode length 1-2 cm, insulator length 2.7-3.9 cm.
- PF-400J: capacitance 880 nF, typical 30 kV, 38 nH, typical 400 J, typical 127 kA, anode radius 0.6 cm, cathode radius 1.3 cm, effective anode length 0.7 cm, insulator length 2.1 cm.
- PF-50J: capacitance 160 nF, typical 25-30 kV, 38 nH, typical 50-70 J, typical 50-60 kA, anode radius 0.3 cm, cathode radius 1.1 cm, effective anode length 0.48 cm, insulator length 2.4 cm.
- Nanofocus: capacitance 5 nF, typical 5-10 kV, 5 nH, typical 0.1 J, typical 5-10 kA, anode radius 0.08-0.022 cm, effective anode length 0.04 cm, insulator length 1 cm.

Source lines:
`KnowledgeReference/studies-on-scalability-and-scaling-laws-for-the-plasma-focus-similarities-and-differences-5f680756.md:526-732`

### Aggregate Shot/Diagnostic Data

The source reports:

- PF-400J maximum measured neutron yield `(1.06 +/- 0.13)e6` neutrons/shot at 9 mbar and about 400 J.
- PF-50J yields `(3.6 +/- 1.5)e4` neutrons/shot at 9 mbar and 67 J, and `(1.3 +/- 0.5)e4` neutrons/shot at 6 mbar and 50 J.
- SPEED2 preliminary CCHEN maximum near `2e10` neutrons/shot at 2-3 mbar, 70 kJ, 2.4 MA.
- PF-400J neutron angular distribution: isotropic component 57.5 percent and anisotropic component 42.5 percent, with anisotropy roughly between +/-50 degrees.
- PF-400J neutron energy `(2.5 +/- 1) MeV`; PF-50J neutron energy `2.7 +/- 1.8 MeV`.
- PF-400J density `(8.4 +/- 1.3)e24 m^-3` in H2; PF-50J density `(1.5 +/- 0.2)e25 m^-3` in D2.
- PF-400J line density `(8.6 +/- 1)e18 m^-1`; PF-50J line density `(2.2 +/- 0.3)e18 m^-1`.
- PF-50J radial velocity of order `1e5 m/s`, rising to about `2e5 m/s` near pinch.

Source lines:
`KnowledgeReference/studies-on-scalability-and-scaling-laws-for-the-plasma-focus-similarities-and-differences-5f680756.md:232-239`,
`KnowledgeReference/studies-on-scalability-and-scaling-laws-for-the-plasma-focus-similarities-and-differences-5f680756.md:305-320`,
`KnowledgeReference/studies-on-scalability-and-scaling-laws-for-the-plasma-focus-similarities-and-differences-5f680756.md:331-354`,
`KnowledgeReference/studies-on-scalability-and-scaling-laws-for-the-plasma-focus-similarities-and-differences-5f680756.md:403-438`,
`KnowledgeReference/studies-on-scalability-and-scaling-laws-for-the-plasma-focus-similarities-and-differences-5f680756.md:680-731`

### Cross-Device Scaling Matrix

Table 2 provides a useful target-extraction matrix for PF-1000, PF-360,
SPEED2, 7 kJ PF, GN1, Fuego Nuevo II, UNU/ICTP-PF, PACO, PF-400J, FMPF-1,
200 J Batt-PF, 125 J PF, PF-50J, and Nanofocus. It includes stored energy,
anode radius, peak current, pressure, energy density parameter, drive
parameter, and energy-per-mass parameter.

For PF-1000 it gives: `E=1064 kJ`, `a=12.2 cm`, `I=2300 kA`, `p=6.6 mbar`,
`28E/a^3=1.6e10 J/m^3`, drive parameter `73.4`, and energy-per-mass
parameter `8.5`.

Source lines:
`KnowledgeReference/studies-on-scalability-and-scaling-laws-for-the-plasma-focus-similarities-and-differences-5f680756.md:741-898`

### First-Principles-Relevant Scaling Constraints

The paper supports source-backed engineering constraints:

- Pinch radius and pinch length scale as `rp ~ (0.1-0.2)a` and `zp ~ (0.8-1)a`.
- Mean pinch density scales with fill gas as `<n> ~ 18 n0`, with typical average order `5e24 m^-3`.
- Pinch-edge magnetic field is expected around `30-40 T` for optimized neutron-emitting PF devices.
- Alfven speed is estimated above `1e5 m/s`.
- Properly operating devices with similar drive/energy-density parameters have temperature of the same order, with cited spectroscopy around `0.6-1 keV`.

Source lines:
`KnowledgeReference/studies-on-scalability-and-scaling-laws-for-the-plasma-focus-similarities-and-differences-5f680756.md:917-957`,
`KnowledgeReference/studies-on-scalability-and-scaling-laws-for-the-plasma-focus-similarities-and-differences-5f680756.md:976-1006`,
`KnowledgeReference/studies-on-scalability-and-scaling-laws-for-the-plasma-focus-similarities-and-differences-5f680756.md:1170-1202`

## Impact On Whole-Shot Blockers

This source helps but does not unblock acceptance:

| Blocker | Impact |
| --- | --- |
| Startup BVP | Does not close. It describes plasma formation sequence and machine geometry but gives no first-principles breakdown, preionization, flashover, current-density, field, or sheath-liftoff arrays. |
| Device deck inputs | Helps. It provides CCHEN device geometry, capacitance, voltage, inductance, current, pressure, and yield targets. |
| Same-scope PF-1000/Akel | Does not close. It gives PF-1000 aggregate scaling row, not Akel shot 12581 same-shot time histories. |
| Dimensionality/handoff | Helps as constraint material. It gives pinch scale, line density, stability-regime context, and LLR/resistive stability discussion, but no 3D field history. |
| Physics closure | Helps with bounds for density, velocity, magnetic field, temperature, Bennett/drive-parameter scaling. It must remain target/comparison material, not a closure model. |
| Neutron authority | Helps for second-scope PF-400J/PF-50J targets and PF-400J angular/energy observables. It does not provide mechanism-separated thermonuclear vs beam-target histories. |
| Comparator/UQ | Helps create typed extraction tasks. It does not itself provide complete comparator metrics, tolerance policy, model uncertainty, or artifact review. |
| Generalization | Strongly helpful. It gives a candidate second-scope device set: PF-400J, PF-50J, SPEED2, Nanofocus, plus cross-device scaling matrix. |

## Required Follow-Up

1. Extract typed table targets from Table 1 and Table 2 into a reviewable packet.
2. Add engineering-only device deck candidates for PF-400J, PF-50J, SPEED2, and Nanofocus.
3. Add first-principles sanity checks that compare candidate pinch radius, density, magnetic field, Alfven speed, and temperature envelopes against this source, without using the empirical scaling as a solver closure.
4. Promote a selected Soto CCHEN device into a second-scope evidence chain only after typed targets and independent review are complete.
5. Keep PF-1000/Akel whole-shot acceptance blocked. This paper does not supply the missing startup BVP, same-shot current waveform, spatial fields/temperatures, mechanism-separated neutron histories, detector response, or UQ certificate.

## Implementation Applied

- Added Soto 2010 to `docs/SCIENTIFIC_CLOSURE_SOURCE_QUEUE.md`.
- Added Soto 2010 to the local source basis and second-scope candidate list in
  `docs/FIRST_PRINCIPLES_FINISH_LINE_PLAN.md`.
- Added `soto2010_cchen_pf400j_pf50j_speed_nanofocus_matrix` to the
  fail-closed runtime generalization packet in
  `src/dpf/first_principles/generalization.py`.
- Updated `docs/FIRST_PRINCIPLES_GENERALIZATION_SOURCE_SEARCH_2026_05_15.md`
  and `.json` to list Soto 2010 as second-scope requirement material.

## Verdict

Useful and ingested. It should be used to build second-scope target extraction
and engineering deck inputs. It does not make the simulator whole-shot-ready.
