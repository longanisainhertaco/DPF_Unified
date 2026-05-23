# SS12 P1 Phase 3 Blocked-Channel Retrieval Snapshot

Date: 2026-05-22 UTC
Scope: PF-1000 blocked/weak source-channel retrieval using local `KnowledgeReference` and `HeliosMatrix_KB` text.

## Evaluate

Helios clean gold eval completed successfully:

```text
question_count 29
fallback False
modes_run ['bm25', 'dense', 'hybrid', 'hybrid_rerank']

bm25          n=29 R@20=0.9655 R@50=0.9655 MRR@10=0.8448 nDCG@10=0.7212
 dense         n=29 R@20=0.8276 R@50=0.9310 MRR@10=0.7069 nDCG@10=0.5199
hybrid        n=29 R@20=1.0000 R@50=1.0000 MRR@10=0.7814 nDCG@10=0.6507
hybrid_rerank n=29 R@20=1.0000 R@50=1.0000 MRR@10=0.9080 nDCG@10=0.8997
```

Interpretation: use `hybrid_rerank` for source discovery when available; use BM25 as a strong fallback because it also performs well on this gold set.

## Same-source boundary

The current same-source matrix source remains:

`KnowledgeReference/neutron-and-fast-ion-emission-from-pf-1000-facility-equipped-with-new-large-electrodes-dc61e78e.md`

It does **not** close the blocked channels below. Several transfer candidates exist, but they must stay `transfer_candidate` or `blocked` until an explicit transfer rule and review exist.

## EM field history / current waveform / circuit-power candidates

### Same-source current/circuit baseline

Source:
`KnowledgeReference/neutron-and-fast-ion-emission-from-pf-1000-facility-equipped-with-new-large-electrodes-dc61e78e.md`

Lines 81-86:

> The outer electrode is 400 mm in diameter and the inner one is 230 mm in diam- ... trodes are about 600 mm in length. These new large elec- trodes seem to be better matched to transmit electrical energy (up to 1 MJ stored in the condenser battery) to plas- ma discharges.

Assessment: same-source, same-scope candidate for geometry and qualitative stored-energy transfer only. Not a current waveform or field history.

Lines 131-136:

> contains only the most important operational parameters: the initial charging voltage (U0), the electrical energy stored in the condenser battery (W0), the D2 filling pressure (p0), the maximum current amplitude (Imax), and a coefficient describing the neutron emission anisotropy...

Assessment: same-source scalar Imax only. Does not close waveform.

### PF-1000 waveform transfer candidate

Source:
`KnowledgeReference/recent-progress-in-1-mj-plasma-focus-research-d3e51f6c.md`

Lines 98-115:

> The PF-1000 condenser bank consists of twelve condenser modules each comprising twenty four 50 kV, 4.625 µF low inductance condensers connected in parallel... main parameters ... U0 = 20-40 kV, E0 = 266-1064 kJ, quarter discharge time T = 5.4 µs ... current derivative probe (dI/dt) is mounted inside the outer collector.

Lines 169-178:

> The “good” shot ... was performed at the discharge energy equal to 1070 kJ and neutron yield amounted to 2.06×10^11 in this shot. The time resolved signals (from Rogowski coil, two PIN diodes, dI/dt probe) as registered in the good shot are presented in Fig. 6.

Lines 237-239:

> Fig. 6. The time resolved signals (from Rogowski coil, two PIN diodes, dI/dt probe).

Assessment: strong PF-1000 / near-1MJ transfer candidate for current waveform evidence, but numeric waveform validation requires figure extraction/digitization and review.

### PF-1000 typical waveform transfer candidate

Source:
`KnowledgeReference/pf-1000-device-a2d6bc15.md`

Lines 160-177:

> The basic diagnostics applied ... measurements of a discharge current and a voltage drop across its electrodes... The current, the time derivative of the current and voltage waveforms of typically plasma focus discharge registered on the PF-1000 device are shown in Fig. 4. The sharp voltage spike and current dip are characteristic for a focusing discharge...

Assessment: same facility, but likely earlier/typical PF-1000, not the Szydlowski large-electrode matrix source. Transfer candidate only.

### Magnetic/EM field history transfer candidate

Source:
`KnowledgeReference/experimental-study-of-the-structure-of-the-plasma-current-sheath-on-the-pf-1000-facility-705bcc83.md`

Lines 202-216:

> measurements of the PCS current ... PF-1000 facility operating with deuterium at an energy stored in the capacitor bank of 250-500 kJ... absolutely calibrated magnetic probes... magneto-optical probes simultaneously recording the dB/dt signal and the plasma optical glow... Rogowski coil recording the total discharge current and three magnetic probes...

Lines 262-297:

> probes designed for measurements of the azimuthal magnetic field distribution... Each probe measured the time derivative of the azimuthal magnetic field dBφ(r, φ)/dt... signal was integrated numerically... calibration accuracy was better than 5%, and accuracy in determining magnetic induction... about 15-20%.

Lines 1202-1236:

> absolutely calibrated magnetic probes ... obtain new data on the parameters and dynamics of the PCS... current ... measured... uncertainty of 20%... neutron yield is determined just by the current compressed onto the axis... demonstrated with the help of laser interferometry...

Assessment: best EM/current uncertainty transfer candidate found. Not same-scope: lower energy, different campaign/configuration. Useful for method/uncertainty transfer rule only.

### Circuit-power coupling transfer candidates

Source:
`KnowledgeReference/pf-1000-device-a2d6bc15.md`

Lines 138-154:

> condenser bank of 1200 kJ, 40 kV ... twelve condenser modules... electric energy is transferred to a collector and electrodes by means of low-inductance cables... C0 = 1.332 mF, E0 = 266÷1064 kJ, L0 = 8.9 nH, T1/4 = 5.4 µs, ISC = 15 MA, Z0 = 2.6 mΩ.

Assessment: strong PF-1000 bank/circuit transfer source. Not exact same-source Szydlowski matrix; use as transfer candidate.

Source:
`KnowledgeReference/scholz-2007-pf1000-part2-jphysd.md`

Lines 225-242:

> total capacitance 1320 µF... voltage U0 ranging between 27 and 36 kV... energies 480 kJ to 850 kJ... electrode sizes were increased... to match external and internal inductances... current and voltage traces...

Assessment: strong transfer candidate for circuit/electrode matching and traces; not full 1MJ same-source.

Source:
`KnowledgeReference/scholz-2006-pf1000-mega-joule.md`

Lines 325-331:

> disagreement between the computed and recorded current waveforms... could be induced by wrong values of the circuit parameters... or ... applied model does not work sufficiently. This question must still be investigated experimentally and theoretically.

Assessment: important fail-closed warning. Current waveform/circuit coupling cannot be promoted without exact reviewed waveform/circuit extraction.

## Density / temperature / startup candidates

Source:
`KnowledgeReference/scholz-2006-pf1000-mega-joule.md`

Lines 149-170:

> The initial breakdown occurs at the insulator surface... current-sheath layer, as formed at the insulator, cannot be accelerated within the inter-electrode gap effectively at a very low gas pressure... Numerical simulation of the breakdown phase... agree relatively well with experimental observations... accurate quantitative model... is still missing... influence of a status of the insulator surface should be taken into consideration... modifications of the insulator surface have not been performed so far.

Assessment: startup BVP transfer candidate and warning. It explicitly says accurate quantitative startup model is missing. This keeps startup blocked.

Lines 190-216:

> 2-fluid MHD model using plasma continuity, momentum and energy equations, Maxwell equations and electrical circuit equation... sensitive to ionization and transport coefficients... specified for chosen electrode configuration and gas conditions... Braginskii transport coefficients... ionization formula... anomalous resistivity... MHD modeling of collapse phase is efficient until maximum compression...

Assessment: model-equation transfer candidate for collapse phase, not same-source validation evidence.

Lines 303-305:

> Fig. 4. Plasma density distribution during the radial collapse, as computed for the PF-1000 experiment and different instants: 9, 9.5 and 10 µs after the discharge beginning.

Assessment: computed density history transfer candidate only; requires figure extraction and model provenance review. Not an experimental same-source density history.

Lines 339-345:

> filtered frame camera recorded emission ... bremsstrahlung depended on the square of electron density and thus the intensity of the pictures depended also on plasma density. The frames in Fig. 8 image evolution of the pinch phase in the time of neutron production.

Assessment: qualitative density proxy transfer candidate; not a calibrated density history.

## Neutron spectrum / detector / uncertainty candidates

Source:
`KnowledgeReference/neutron-and-fast-ion-emission-from-pf-1000-facility-equipped-with-new-large-electrodes-dc61e78e.md`

Lines 96-102:

> scintillation probes were used... located ~15 m from the electrode outlet... In future experiments the same probes will be used to measure neutron energy spectra by means of the time-of-flight method.

Assessment: explicit same-source absence proof for neutron spectrum: spectra were future work in this paper. Keep neutron_spectrum blocked.

Lines 88-96:

> total neutron yield ... measured with four silver activation counters... calibrated using an Am-Be neutron source...

Assessment: same-source detector-calibration candidate, but not full detector response/uncertainty budget.

Source:
`KnowledgeReference/experimental-study-of-the-structure-of-the-plasma-current-sheath-on-the-pf-1000-facility-705bcc83.md`

Lines 1202-1222:

> absolutely calibrated magnetic probes... uncertainty of 20%...

Assessment: transfer uncertainty candidate for PCS-current measurement, not same-source neutron detector uncertainty.

## Learn

The next physics blocker is not absence of code. It is absence of same-scope accepted evidence for full field histories and startup/spectrum/uncertainty.

Most promising transfer candidates:

1. `recent-progress-in-1-mj-plasma-focus-research-d3e51f6c.md` for near-1MJ Rogowski/dI/dt/PIN time traces.
2. `experimental-study-of-the-structure-of-the-plasma-current-sheath-on-the-pf-1000-facility-705bcc83.md` for magnetic probe dB/dt, current compression, calibration, and uncertainty.
3. `scholz-2006-pf1000-mega-joule.md` for startup caveats, MHD collapse model limits, computed density distributions, and current-waveform mismatch warning.

## Continue

Next executable step:

- Create a transfer-candidate matrix separate from the same-source acceptance matrix.
- Add a validator that transfer candidates cannot promote acceptance.
- Then extract/digitize figure-backed waveform/density candidates only after choosing exact figures and review criteria.

Acceptance flags remain false.
