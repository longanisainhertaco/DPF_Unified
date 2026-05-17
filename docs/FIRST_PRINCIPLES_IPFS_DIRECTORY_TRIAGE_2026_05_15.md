# First-Principles IPFS Directory Triage - 2026-05-15

This note classifies the PlasmaFocus.net `/IPFS/` directory for first-principles
DPF simulator development. The directory is useful, but mostly as a Lee-model
archive, reference-deck source, training/course corpus, and baseline-comparison
source. It does not, by itself, provide the missing whole-shot 3D first-principles
closures.

Primary index reviewed:

- `https://www.plasmafocus.net/IPFS/`

## Directory Inventory

High-signal directories and pages:

| Path | Contents | First-principles use |
| --- | --- | --- |
| `/IPFS/modelpackage/` | RADPF/Lee-model workbooks, theory PDF, appendix PDF, corona calculations | Baseline/reduced-model provenance; no predictive authority for first-principles mode |
| `/IPFS/machines/` | Machine workbooks for PF1000, PF400, DPF78, Poseidon, NX2, UNU/ICTP PFF | Non-promoting reference decks, waveform fixtures, importer tests |
| `/IPFS/IctpIaeaCoursePfne/` | ICTP/IAEA course manual plus code/data workbooks | Lee fitting workflow provenance and reference datasets |
| `/IPFS/usefulfiles/` | `LCR.xls` | Circuit-only damped-discharge sanity fixture |
| `/IPFS/NEWPF2022/` | Workshop announcement/course scope | Confirms Lee-code workflow and data use; not physics closure |
| `/IPFS/2008/` through `/IPFS/2019papers/` | Yearly Lee/Saw/Akel/IPFS publications | Source-discovery queue for baselines, diagnostics, and reduced-model comparisons |
| `/IPFS/phdtheses/` | Thesis PDFs | Possible background/source-discovery queue, not accepted closure without review |
| `/IPFS/otherpapers/` | Older DPF theory, energy balance, current stepping, lab manuals | Possible historical context and source-discovery queue |

Low-signal directories for simulator physics:

- `associatedactivities`
- `associates`
- `claims`
- `conferenceprogrammes`
- `tripreports`
- `turkeytour_files`
- PowerPoint/Word-only event material unless a specific physics paper is missing

## High-Value Finds

### 1. Model Package

Reviewed:

- `https://www.plasmafocus.net/IPFS/modelpackage/`
- `https://www.plasmafocus.net/IPFS/modelpackage/File1RADPF.htm`
- `https://www.plasmafocus.net/IPFS/modelpackage/File2Theory.pdf`
- `https://www.plasmafocus.net/IPFS/modelpackage/File3Appendix.pdf`
- `https://www.plasmafocus.net/IPFS/modelpackage/UPF.htm`

The model package contains the RADPF/Lee model workbooks, the theory document,
the radiation/self-absorption appendix, and the universal plasma focus download
page. The theory document lays out the five-phase reduced model: axial snowplow,
radial inward shock/slug, reflected shock, slow compression/radiative pinch, and
expanded column. It includes circuit coupling, fitted mass/current factors,
radiation loss terms, thermonuclear neutron estimate, and a phenomenological
beam-target neutron estimate.

Classification:

- Useful as `baseline_reduced_model`.
- Useful for regression comparisons against existing Lee-model behavior.
- Useful for checking that first-principles mode does not silently import Lee
  calibration factors.
- Not first-principles closure authority.

Existing local coverage:

- `KnowledgeReference/lee_radpf_theory.md`
- `KnowledgeReference/lee-2014-plasma-focus-radiative-model.md`

The key Lee theory appears already represented in the local source corpus. Do not
duplicate it as a new accepted source unless the local corpus is found incomplete.

### 2. Machine Workbooks

Reviewed:

- `https://www.plasmafocus.net/IPFS/machines/`
- `PF1000 05.15.xls`
- `PF400 05.15.xls`
- `DPF78 05.15.xls`
- `poseidon 05.15.xls`
- `NX2/`
- `UNU ICTP PFF/`

The machine directory is valuable for reference decks and current-trace fixtures.
The PF1000 workbook was already examined separately and contains the PF1000 27 kV
geometry, bank, fill, measured current trace, and Lee-model outputs.

Classification:

- Useful for non-promoting machine decks.
- Useful for waveform-comparison fixtures.
- Useful for importer/unit conversion tests.
- Not first-principles closure authority.

Recommended ingestion:

- Create a `reference/baseline/plasmafocus_ipfs` manifest with file hashes,
  source URLs, machine parameters, waveform column metadata, and explicit
  `authority = baseline_reduced_model`.
- Keep Lee-model fitted factors out of first-principles solver state.

### 3. ICTP/IAEA Course Code And Data

Reviewed:

- `https://www.plasmafocus.net/IPFS/IctpIaeaCoursePfne/`
- `https://www.plasmafocus.net/IPFS/IctpIaeaCoursePfne/2%20ICTP%20e-manual%202.pdf`
- `https://www.plasmafocus.net/IPFS/IctpIaeaCoursePfne/CodeNdata.htm`
- `https://www.plasmafocus.net/IPFS/IctpIaeaCoursePfne/3%20code%20%26%20data/`

The course data directory contains:

- `1 RADPFV5.15de.xls`
- `2 PF1000data.xls`
- `3 PF400data.xls`
- `5 PF1000dataNom.xls`
- `6 DPF78dataNom.xls`
- `7 PFcomparisonpf1000pf400.xls`
- `8 PF1000pressureblank.xls`
- `9 NX2pressureblank.xls`
- `10 PF400YnComparison.xls`

The smaller PF1000 course data workbook contains the same useful deck shape:

- `Lo = 33.5 nH`
- `Co = 1332 uF`
- `ro = 6.3 mOhm`
- `b = 16.0 cm`
- `a = 11.55 cm`
- `zo = 60.0 cm`
- `Vo = 27 kV`

The nominal workbook uses `Lo = 20 nH` and unknown `ro`, which is explicitly a
course fitting exercise rather than a validated machine deck.

Classification:

- Useful for importer fixtures and Lee fitting workflow provenance.
- Useful for pressure-sweep baseline comparison charts.
- Not first-principles closure authority.

### 4. `LCR.xls`

Reviewed:

- `https://www.plasmafocus.net/IPFS/usefulfiles/LCR.xls`

The workbook is a circuit-only L-C-R computation. Example default values are:

- `Co = 28 uF`
- `Lo = 20 nH`
- `ro = 2.3 mOhm`
- `Vo = 11 kV`

Classification:

- Useful as a circuit sanity fixture for unloaded/short-circuit current and
  capacitor voltage.
- Useful for checking first-principles circuit integration before plasma loading.
- Not plasma physics closure.

### 5. Current-Voltage, Pressure, Beam, And Akel Papers

Sampled high-signal files from the yearly paper directories:

- `/IPFS/2017 Papers/15  The current-voltage characteristics of the plasma focus.pdf`
- `/IPFS/2017 Papers/2 effectVariationPressureDynamicsNeutron.pdf`
- `/IPFS/2017 Papers/4a eBeamPropsD2PF.pdf`
- `/IPFS/2012papers/11 POP FIB Deuterium .pdf`
- `/IPFS/2016 papers/7 effectsPowerTermsThermPinchRad.pdf`
- `/IPFS/2019papers/AkelPF24Yn.pdf`
- `/IPFS/2019papers/ionbeamcodevalidationnear.pdf`
- `/IPFS/2010 Papers/2010PP4Akel.pdf`

These papers are potentially useful for:

- Expected circuit/current/voltage waveform features.
- Pressure-trend comparisons.
- Beam/electron/ion comparator outputs.
- PF1000, PF24, PF400, NX2, and other device reference cases.
- Reduced-model literature provenance.

They are not sufficient first-principles closures because they generally use the
Lee model, fitted current traces, phenomenological beam-target terms, or
diagnostic correlations rather than deriving a self-consistent whole-shot 3D
plasma state from Maxwell/fluid/kinetic equations and accepted material/atomic
data.

## Blocker Mapping

| First-principles blocker | Does `/IPFS/` appear to solve it? | Notes |
| --- | --- | --- |
| Startup breakdown / surface flashover BVP | No | No accepted first-principles startup model found in the sampled directory structure. |
| D2 molecular/atomic/excited-state collisional-radiative kinetics | No | Corona workbooks provide legacy lookup behavior only. |
| EOS for multi-species plasma state | No | No accepted EOS table or derivation found. |
| Magnetized tensor transport | No | No Braginskii/Spitzer-class implementation source found in this pass. |
| Radiation transport | Partial background only | Appendix provides Lee-model self-absorption approximations, not full radiation transport closure. |
| Electrode/wall/ablation/impurity coupling | No | Some papers discuss materials/effects, but no complete closure source found. |
| Mechanism-separated neutron authority | No | Lee package uses thermonuclear plus phenomenological beam-target estimates. |
| 3D whole-shot MHD/PIC/kinetic algorithm | No | Directory is centered on 0D/1D/phase-model Lee/RADPF workflows and experimental comparisons. |

## Use Decision

This directory should be used in three lanes:

1. `baseline_reduced_model`: Lee/RADPF theory, workbooks, fitted outputs, and
   pressure/yield/beam trends.
2. `reference_machine_data`: PF1000, PF400, DPF78, Poseidon, NX2, UNU/ICTP
   machine parameters and measured current traces.
3. `source_discovery_queue`: papers and theses that may point to primary
   physics sources, diagnostic data, or comparison datasets.

It should not be used as:

- `first_principles_closure`
- `same_scope_authority`
- `neutron_predictive_authority`
- `3d_algorithm_authority`

## Immediate Engineering Actions

1. Add a non-promoting PlasmaFocus IPFS manifest for machine/workbook imports.
2. Add PF1000 course-data and machine-workbook fixtures behind an explicit
   baseline/reference namespace.
3. Add an `LCR.xls`-derived circuit-only regression fixture for unloaded bank
   behavior.
4. Add a source-discovery queue for unsampled high-signal PDFs, especially Akel,
   current-voltage, pressure/neutron, beam, and PF1000/PF24 papers.
5. Keep first-principles readiness gates blocked until local source-of-truth
   materials provide accepted equations, rate data, numerical methods, and
   acceptance criteria for the missing closures listed above.

