# GV Verified Shot Bundle Triage

Date: 2026-05-16

Scope: `/Users/anthonyzamora/Downloads/GV`

Status: useful, but not sufficient for a whole-shot first-principles DPF
simulation or accepted validation certificate.

## Verdict

The GV bundle is not the missing complete first-principles shot packet. It is a
verified set of machine/circuit/gas decks plus current-waveform workbooks and
Gratton-Vargas reduced-model output.

Use it for:

- source-scoped second-device engineering decks;
- measured-current waveform target extraction;
- reduced-model baseline comparison;
- generalization testing across PF-24, PF-360, LPP-FF1, Gemini, and OneSys.

Do not use it for:

- startup BVP closure;
- spatial density, field, or temperature history;
- mechanism-separated neutron authority;
- detector response or uncertainty;
- first-principles validation certificate promotion.

`GV.exe` is a Windows PE32 FORTRAN executable and was not run.

## What Was Found

The directory contains:

- eight unique shot/device examples represented by `.inp`, `.TXT`, and `.xlsx`
  files;
- duplicate PF-360 input aliases (`Gvinp.inp` and `Gvinp .inp`);
- `Resistive Gratton-Vargas Model.pdf`, a 3-page usage document;
- `GV.exe`, the reduced-model executable.

The `.inp` files are Fortran-style namelists with:

- anode radius and length;
- insulator radius and length;
- cathode radius;
- capacitance, inductance, resistance, voltage;
- deuterium pressure;
- shot/device note.

The `.TXT` files are GV reduced-model output tables with 11 columns:

1. time normalized to quarter-cycle time
2. current normalized to `I0`
3. fraction of energy remaining in capacitor
4. fraction converted to magnetic energy
5. fraction dissipated in resistance
6. remaining energy as work done
7. `tau`
8. dynamic dimensionless inductance
9. time in microseconds
10. current in kiloamperes
11. total inductance in nH

The `.xlsx` workbooks are more useful than the `.TXT` files because they carry
experimental current waveform columns alongside GV output.

## Shot Inventory

| Shot target | Device | Deck values | Experimental waveform columns | Useful status |
| --- | --- | --- | --- | --- |
| `lpp_ff1_05_23_16_1` | LPP-FF1 | `a=28 mm`, `z=140 mm`, `C=75.2 uF`, `L=34 nH`, `R=7.0 mohm`, `V=39.9 kV`, `p_fit=12.0 torr` | `L=time_us`, `M=current_kA`, 6789 rows | current target candidate |
| `lpp_ff1_05_24_16_6` | LPP-FF1 | `a=28 mm`, `z=140 mm`, `C=75.2 uF`, `L=35 nH`, `R=5.6 mohm`, `V=40.1 kV`, `p_fit=12.2 torr` | `L=time_us`, `M=current_kA`, 6789 rows | current target candidate |
| `pf24_krakow_14082734` | PF-24-KRAKOW | `a=31 mm`, `z=172 mm`, `C=115.2 uF`, `L=15 nH`, `R=8 mohm`, `V=16 kV`, `p_fit=1.35 torr` | `L=time_us`, `M=current_kA`, 600 rows | current target candidate |
| `pf24_krakow_16052007` | PF-24-KRAKOW | `a=31 mm`, `z=172 mm`, `C=115.2 uF`, `L=20 nH`, `R=14 mohm`, `V=16 kV`, `p_fit=1.1 torr` | `L=time_us`, `M=current_kA`, 651 rows | current target candidate |
| `pf24_krakow_16092202` | PF-24-KRAKOW | `a=31 mm`, `z=172 mm`, `C=115.2 uF`, `L=21 nH`, `R=22 mohm`, `V=16 kV`, `p_fit=1.1 torr` | `L=time_us`, `M=current_kA`, 649 rows | current target candidate and built-in deck preset |
| `pf360_20140122_7` | PF-360 | `a=60 mm`, `z=304 mm`, `C=262.6 uF`, `L=17 nH`, `R=5.2 mohm`, `V=31 kV`, `p_fit=10 torr` | smoothed `L/M`, raw `AC/AD`, 22980 rows | stronger waveform candidate because raw columns are present |
| `gemini_rog_i005_20130716` | Gemini | `a=76.2 mm`, `z=596.9 mm`, `C=432 uF`, `L=29.7 nH`, `R=2.5 mohm`, `V=40.13 kV`, `p_fit=5.7 torr` | reduced `L/M`, raw `AC/AD`, 30387 rows | stronger waveform candidate because raw columns are present |
| `onesys_rog01004_20051208` | OneSys | `a=50.8 mm`, `z=393.7 mm`, `C=216 uF`, `L=46 nH`, `R=2 mohm`, `V=35 kV`, `p_fit=6.8 torr` | reduced `L/M`, raw `AC/AD`, 20002 rows | stronger waveform candidate because raw columns are present |

## Source-Truth Handling

These files are user-verified local artifacts, but they are not yet
`KnowledgeReference/` source-truth records. They must remain non-promoting until
raw artifacts or one-for-one verified extracts are promoted into the source of
truth with hashes, units, column contracts, and review status.

Code added from this triage:

- `dpf.first_principles.source_targets.gv_verified_shot_targets()`
- `dpf.first_principles.deck.gv_verified_engineering_deck()`
- `dpf.first_principles.deck.gv_verified_engineering_decks()`
- `dpf.first_principles.gv_waveforms.extract_gv_current_waveform_packet()`
- `dpf.first_principles.gv_waveforms.extract_all_gv_current_waveform_packets()`
- `dpf.first_principles.gv_waveforms.gv_waveform_packet_summary()`
- CLI preset `--deck-preset gv_pf24_krakow_16092202`
- CLI extractor `dpf first-principles-gv-waveform --shot-id <shot> --summary`

The new decks use only machine/circuit/gas values and target metadata. They do
not import GV reduced-model dynamics into the first-principles solver.

The waveform extractor reads the workbook columns directly, checks the workbook
hash against the verified local artifact, emits full `x=time_us` and
`y=current_kA` arrays, and marks every packet as
`accepted_for_first_principles_validation=false`. For PF-360, Gemini, and
OneSys, `series=preferred` chooses the raw `AC/AD` columns. For PF-24 and
LPP-FF1 shots, it chooses the available `L/M` experimental waveform columns.

Example packet status for `pf24_krakow_16092202`:

- task id: `gv_pf24_krakow_16092202_current_waveform_candidate`
- points: 649
- columns: `L=time_us`, `M=current_kA`
- time range: `-0.5` to `6.0 us`
- maximum current: `401.6 kA`
- promotion state: engineering comparator candidate only

Example packet status for `pf360_20140122_7`:

- task id: `gv_pf360_20140122_7_current_waveform_candidate`
- preferred columns: `AC=raw_time_us`, `AD=raw_current_kA`
- points: 22980
- time range: `-0.5564` to `9.2232 us`
- maximum current: `2015.325 kA`
- promotion state: engineering comparator candidate only

## PIConGPU Architecture Guidance

PIConGPU remains architecture guidance only. The useful patterns to reimplement
in DPF-Unified are:

- typed user-facing simulation setup that compiles into code-native input
  decks, similar in spirit to PICMI setup generation;
- clear separation between field solver, particle pusher, current deposition,
  collisions, ionization, diagnostics, restart, and output plugins;
- openPMD-like field/particle output and restart contracts;
- backend/precision decisions surfaced as explicit run metadata and fidelity
  gates;
- model pipelines that expose particle filters, collision-screening species,
  debug output, and precision choices.

Do not import PIConGPU as a DPF physics authority. Its documented PIC cycle,
openPMD plugin, PICMI setup pattern, and collision-pipeline organization are
software architecture references for our own source-grounded implementation.

Primary architecture references reviewed:

- PIConGPU README, 3D3V PIC features, charge-conserving deposition, field
  solvers, particle pushers, diagnostics, restart/output, and plugin structure:
  <https://github.com/ComputationalRadiationPhysics/picongpu>
- PIConGPU PICMI intro, standardized Python simulation object and generated
  setup workflow:
  <https://picongpu.readthedocs.io/en/latest/usage/picmi/intro.html>
- PIConGPU openPMD plugin, field/particle data, restart/output, streaming, and
  backend configuration:
  <https://picongpu.readthedocs.io/en/latest/usage/plugins/openPMD.html>
- PIConGPU binary-collision model, collision pipeline, particle filters,
  precision/debug controls, and per-cell algorithm structure:
  <https://picongpu.readthedocs.io/en/latest/models/binary_collisions.html>

## Next Actions

1. Promote the raw GV artifacts or verified extracts into `KnowledgeReference/`
   before accepted-source use.
2. Extract workbook current waveforms into typed target packets with units,
   point counts, hashes, and waveform notes.
3. Add comparator bindings that compare first-principles `I(t)` only against
   experimental workbook columns, not GV output columns.
4. Keep the GV `.TXT` output as a reduced-model baseline regression surface.
5. Use PF-360, Gemini, and OneSys first for high-value waveform extraction
   because they include raw `AC/AD` columns in addition to reduced/calibrated
   waveform columns.
