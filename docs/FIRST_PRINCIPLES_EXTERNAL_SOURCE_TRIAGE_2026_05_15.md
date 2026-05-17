# First-Principles External Source Triage - 2026-05-15

This note classifies the PlasmaFocus.net IPFS machine package, the Corona
Calculations workbooks, and the verified local PF1000 spreadsheet for use in the
first-principles DPF simulator. The classification is intentionally strict:
reduced-model outputs can support baseline comparison and regression fixtures,
but they must not become first-principles closure authority.

## Sources Reviewed

- PlasmaFocus.net machine index:
  `https://www.plasmafocus.net/IPFS/machines/`
- PF1000 workbook linked from that index:
  `https://www.plasmafocus.net/IPFS/machines/PF1000%2005.15.xls`
- Verified local PF1000 workbook:
  `/Users/anthonyzamora/Downloads/PF1000 05.15.xls`
- Corona Calculations introduction:
  `http://www.plasmafocus.net/IPFS/modelpackage/Corona%20Calculations/C1coronaIntroduction.htm`
- Deuterium corona workbook:
  `https://www.plasmafocus.net/IPFS/modelpackage/Corona%20Calculations/C6coronadeuterium.xls`
- General macro workbook:
  `https://www.plasmafocus.net/IPFS/modelpackage/Corona%20Calculations/corona1-54.xlsm`

## Source Classification

| Source | Use | First-principles authority? | Notes |
| --- | --- | --- | --- |
| PlasmaFocus.net machine index | Source-discovery metadata for machine workbooks | No | The page describes a Lee/RADPF model workflow using fitted mass/current factors and measured current traces. |
| `PF1000 05.15.xls` | PF1000 machine deck, measured current trace, reduced-model baseline outputs | No | Useful as a non-promoting PF1000 full-energy reference case. Not a closure source. |
| Corona Calculation workbooks | Optional comparison tables for `Zeff` and specific heat ratio | No | Useful to inspect model assumptions and regression tables. Not sufficient for dynamic first-principles ionization, EOS, or radiation closure. |

## PF1000 Workbook Findings

The local workbook is an old-format Excel file:

- Path: `/Users/anthonyzamora/Downloads/PF1000 05.15.xls`
- Size: `3,244,032` bytes
- SHA-256: `4ea8add69e1a812bd84bdcbcd37df19a9ecbc1c38914bafe000f774794f7a917`
- Sheets: `Sheet1`, `Sheet2`, `Sheet3`

The local file size matches the remote PF1000 workbook advertised by the
machine index.

### Machine And Operating Parameters

The workbook gives a PF1000 27 kV deuterium case with these main parameters:

| Parameter | Value |
| --- | --- |
| Capacitance | `1332 uF` |
| Voltage | `27 kV` |
| Static inductance | `33.5 nH` |
| Static resistance | `6.1 mOhm` in Sheet1; `6.3 mOhm` in Sheet2 notes |
| Anode radius | `11.55 cm` |
| Cathode radius | `16.0 cm` |
| Anode length | `60.0 cm` |
| Fill | `3.5 Torr D2` |

These values are useful for a PF1000 reference deck and for exercising the
first-principles circuit plus geometry initialization path.

### Measured Current Trace

`Sheet2` contains a measured PF1000 current trace:

- 94 digitized points
- Time range: approximately `-1.68234 us` to `14.732 us`
- Peak current: approximately `1845.76 kA` at `6.39003 us`
- Last listed current: approximately `794.325 kA` at `14.732 us`

This trace is useful for downstream engineering comparison after the
first-principles simulator exists. It should not be used as a fitted closure
inside the first-principles run.

### Reduced-Model Baseline Outputs

`Sheet1` is a Lee/RADPF V5.015 computation. It contains fitted factors and
computed outputs, including:

| Quantity | Workbook value |
| --- | --- |
| Bank energy | `485.514 kJ` |
| Peak current | `1844.713 kA` |
| Pinch-start current | `772.468 kA` |
| Peak pinch temperature | `0.892354e6 K` |
| Minimum radius | `2.16784 cm` |
| Pinch duration | `263.995 ns` |
| Maximum tube voltage | `40.1341 kV` |
| Peak pinch density | `4.1024e23 m^-3` |
| Neutron yield | `9.2961420062416e10` |
| Axial-end time | `7.414766766 us` |
| Radial-end time | `9.068839100 us` |

The fitted factors are:

| Factor | Value |
| --- | --- |
| Axial mass factor | `0.13` |
| Axial current factor | `0.7` |
| Radial mass factor | `0.35` |
| Radial current factor | `0.65` |

These fitted factors are reduced-model calibration artifacts. They must remain
baseline-only and must not be imported into first-principles dynamics.

### Existing Repository Coverage

The PF1000 spreadsheet is already partially represented in the repository as a
validation/reference artifact:

- `src/dpf/validation/experimental_waveforms.py` includes the 94-point PF1000
  Gribkov current trace and cites the PlasmaFocus.net PF1000 workbook.
- `src/dpf/validation/experimental_devices.py` includes the corresponding
  PF1000 bank, geometry, fill, peak-current, and neutron-yield metadata.

For the first-principles simulator, the useful next step is to mirror these
values into a non-promoting reference deck or source registry that the
first-principles workflow can read for setup and post-run comparison without
granting reduced-model closure authority.

## Corona Workbook Findings

The Corona Calculations page points to per-gas and macro-enabled workbooks for
ionization fractions, effective charge, and specific heat ratio. The deuterium
workbook contains:

- Temperature range: about `10,000 K` to `9,720,557 K`
- Electron-temperature equivalent: about `0.8617 eV` to `837.628 eV`
- Neutral and ion fractions
- `Zeff`
- Specific heat ratio values

Sample deuterium values from the workbook:

| Temperature | Electron-temperature equivalent | `Zeff` |
| --- | --- | --- |
| `10,000 K` | `0.8617 eV` | about `0.000547` |
| `16,288.95 K` | `1.4036 eV` | about `0.4947` |
| `19,799.32 K` | `1.7061 eV` | about `0.8941` |
| `26,532.98 K` | `2.2864 eV` | about `0.9911` |
| `37,334.56 K` | `3.2171 eV` | about `0.9991` |

The table is useful as a comparison target or as a clue for what Lee-model
workflows expected from a corona approximation. It is not enough to close the
first-principles model because it does not provide the dynamic collisional-
radiative equations, rate coefficients, D2 molecular pathways, excited-state
kinetics, impurity handling, radiation transport, or non-equilibrium electron
and ion energy coupling needed for a whole-shot DPF calculation.

## What These Sources Help With

These sources can help with:

1. Creating a PF1000 full-energy reference deck for geometry, bank, fill, and
   circuit setup.
2. Preserving a measured current waveform for post-run engineering comparison.
3. Providing reduced-model Lee/RADPF baseline outputs for regression plots and
   sanity checks.
4. Identifying legacy `Zeff` and gamma table expectations from the Lee-model
   ecosystem.
5. Building non-promoting tests that verify importers, unit conversion, deck
   parsing, and comparison report generation.

## What These Sources Do Not Solve

These sources do not close the first-principles blockers:

1. No accepted startup breakdown or surface-flashover boundary-value problem.
2. No dynamic D2 molecular, atomic, ion, excited-state, or impurity
   collisional-radiative state model.
3. No first-principles EOS table for the multi-species plasma state.
4. No magnetized tensor transport closure with Hall, Nernst, and anisotropic
   thermal conduction support.
5. No radiation transport closure.
6. No electrode, sheath, ablation, impurity, or wall-coupling material model.
7. No mechanism-separated neutron-production authority for beam-target,
   thermonuclear, anisotropy, and detector-response prediction.
8. No 3D MHD/PIC/kinetic numerical algorithm or verification evidence.

## Required Ingestion Decision

Recommended ingestion policy:

1. Ingest `PF1000 05.15.xls` as `reference/baseline`, not as
   `first_principles/closure`.
2. Add a PF1000 reference-deck fixture for machine setup and post-run
   comparison.
3. Keep Lee/RADPF fitted factors and computed yield/timing outputs out of the
   first-principles solver state.
4. Optionally ingest the deuterium corona table under a `legacy_corona_baseline`
   namespace for comparison only.
5. Keep runtime ionization/EOS/radiation work blocked until the source-of-truth
   corpus provides equations, rate data, and acceptance gates sufficient for a
   first-principles implementation.

