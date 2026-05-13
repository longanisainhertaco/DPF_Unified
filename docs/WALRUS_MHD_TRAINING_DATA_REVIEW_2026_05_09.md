# WALRUS and MHD Training Data Review - 2026-05-09

## Scope

This review answers whether local MHD/WALRUS training artifacts can be treated
as valid DPF-Unified data, and what external publications or source records are
needed before they can support stronger claims.

The short answer is:

- Use the current local files for pipeline development, schema experiments,
  negative tests, and exploratory ML only.
- Do not use the current local HDF5 training data for scientific validation,
  publication claims, high-fidelity readiness, or source-backed physics claims.
- Do not treat generic public WALRUS/The Well MHD data as DPF validation data.
  It can support ML pretraining or benchmark comparisons only.

## Source-Of-Truth Rule

External WALRUS, The Well, CATS, and V&V pages are acquisition and method leads.
They are not DPF physics evidence for this project until exact source records
are acquired, hashed, reviewed into `KnowledgeReference/`, and mapped to the
specific claim they support.

For DPF physics validation, the current local `KnowledgeReference/` gate still
controls all scientific claims.

## External Search Findings

| Source area | Lead found | What it can support | What it cannot support |
| --- | --- | --- | --- |
| WALRUS model | PolymathicAI WALRUS GitHub and arXiv `2511.15684`; repository states MIT license and 19 physical scenarios across 2D/3D continuum fields. | ML architecture, local inference/fine-tuning workflow, model-card/license review, benchmark context. | DPF physics validation or acceptance of our generated training data. |
| The Well | NeurIPS 2024 paper "The Well: a Large-Scale Collection of Diverse Physics Simulations for Machine Learning", DOI `10.52202/079017-1430`; public docs describe 15 TB, 16 datasets, HDF5 format, and PyTorch interface. | Data-format authority, ML benchmark framing, pretraining/benchmark source. | Same-scope DPF validation or predictive DPF evidence. |
| The Well MHD | `MHD_64`/`MHD_256` docs describe isothermal compressible MHD turbulence, 100 trajectories, density/velocity/magnetic-field fields, periodic Cartesian grid, dimensionless time, Mach and Alfven Mach parameter combinations, and Fortran+MPI generation. | Generic MHD surrogate pretraining or benchmark comparison against public metrics such as spectra/correlation functions. | DPF validation: it is not cylindrical DPF, not circuit-coupled, not resistive/electrode/radiation/neutron DPF, and not same-device data. |
| CATS paper | Burkhart et al. 2020, "The Catalogue for Astrophysical Turbulence Simulations (CATS)", DOI `10.3847/1538-4357/abc484`. | Published source for the astrophysical MHD turbulence dataset lineage. | DPF validation, PF-1000 claims, or electrode/circuit/neutron physics closure. |
| V&V/credibility methods | NASA CFD V&V tutorial and ASME/FDA computational-model credibility framework leads. | Context-of-use and credibility framing: qualitative, incremental, or absolute claims require different evidence levels. | They do not validate any DPF training data by themselves. |

## Local Artifacts Found

| Artifact | Git status | Size/count | Contents | Immediate classification |
| --- | --- | --- | --- | --- |
| `docs/walrus_training_sweep.json` | tracked | 20 trajectories | Lee-model sweep summaries with `V0_kV`, `P_torr`, `I_peak_MA`, `t_peak_us`, `Yn`, and step counts. | Exploratory Lee-model training summary, not MHD and not validation. |
| `docs/walrus_training_192.json` | tracked | 192 trajectories | Lee-model PF-1000 current waveform/yield trajectories over `V0_kV`, `P_torr`, and `fm`; commit `d75cfb5`. | Exploratory current-waveform surrogate data, not volumetric MHD and not validation. |
| `docs/walrus_training_cross_device.json` | tracked | 80 trajectories | Lee-model cross-device waveform/yield trajectories over presets and scale factors; commit `d75cfb5`. | Exploratory cross-device Lee-model data, not same-scope validation. |
| `docs/walrus_training_pf1000.h5` | ignored by `*.h5` | 192 trajectories | HDF5 with `parameters` and `t0_fields/current_MA`; no `boundary_conditions`; only current waveform data. | Not Well-schema valid under the current validator; not MHD. |
| `training_data/dpf_v2/trajectory_*.h5` | ignored by `training_data/` | 14 HDF5 files, about 24 MB | Well-like 16 x 1 x 16 cylindrical snapshots, 186 time samples, density/temp/pressure/velocity/B fields. | Not defensible; missing metadata and contains numerical/provenance defects. |
| `training_data/dpf_batch_50/trajectory_*.h5` | ignored by `training_data/` | 50 HDF5 files, about 166 MB; metadata says 50/50 success | Well-like 32 x 1 x 64 cylindrical snapshots, 20 time samples, generated from broad parameter ranges. | Not defensible; validator passes are insufficient and data contain defects. |
| `models/walrus-pretrained/walrus.pt` | ignored by `models/` | about 4.8 GB | Local WALRUS checkpoint. | Usable only after model source, license, version, hash, and inference behavior are recorded. |

## Local Data Audit Results

### `training_data/dpf_v2`

- 14 HDF5 files were found.
- No `metadata.json` or run manifest was found for this directory.
- Sample file `trajectory_0000.h5` root attributes include circuit values,
  `dataset_name="dpf_simulation"`, `grid_type="cartesian"`, `n_spatial_dims=2`,
  and `rho0`.
- The coordinate datasets are `r`, `theta`, and `z`, so the root
  `grid_type="cartesian"` is inconsistent with the coordinate names.
- All 14 files pass the current `DatasetValidator`, but that validator does not
  check `scalars/current` or `scalars/voltage` for non-finite values.
- All 14 files have non-finite `current` and `voltage` scalars.
- Many field arrays reach near the float32 maximum, around `3.4e38`, which is
  consistent with overflow/saturation or sanitized numerics rather than a
  physically credible DPF state.
- The sampled magnetic field is all zeros.
- `scalars/energy_conservation` is absent, so the validator reports only a
  warning and returns unknown energy drift.

### `training_data/dpf_batch_50`

- 50 HDF5 files were found.
- `metadata.json` records 50 requested, 50 successful, 0 failed, grid
  `[32, 1, 64]`, `dx=5e-4`, `sim_time=2e-6`, cylindrical geometry, and full
  parameter ranges over voltage, capacitance, inductance, resistance, fill
  density, and anode radius.
- All 50 files pass the current `DatasetValidator`, but that pass is not a
  scientific validity result.
- All 50 files have non-finite `current` and `voltage` scalars.
- 169 field-dataset/file combinations exceed `1e30` in absolute value; several
  hit values near `3.4e38`.
- The sampled magnetic field is all zeros.
- `scalars/energy_conservation` is absent in every file.

### `docs/walrus_training_pf1000.h5`

- The file has `dataset_name="dpf_pf1000_lee"`, `n_trajectories=192`, parameter
  names `V0_kV`, `P_torr`, and `fm`, and a single current field
  `t0_fields/current_MA`.
- It fails the current `DatasetValidator` because `boundary_conditions` is
  missing.
- It is a Lee-model current-waveform dataset, not MHD volumetric training data.

## Validity Assessment

| Use case | Current local WALRUS/DPF data status | Decision |
| --- | --- | --- |
| Unit tests for field mapping, HDF5 shape handling, and loader behavior | Useful if explicitly labeled synthetic or negative-test data. | Allowed. |
| WALRUS integration smoke tests | Useful for checking that code can load a model or handle HDF5-like files. | Allowed with `Exploratory` classification. |
| Training a production surrogate of DPF-Unified | Not acceptable as-is because the HDF5 sets contain non-finite scalars, saturated fields, zero B fields, missing energy checks, and missing manifests. | Regenerate before use. |
| Training a surrogate of the old Lee-model waveform sweep | Possible for exploratory current-waveform emulation only, because the tracked JSON data are Lee-model generated and include no experimental truth. | Allowed only as `Derived Diagnostic` or `Exploratory`. |
| Scientific validation of DPF-Unified | Not acceptable. Synthetic data from our own unvalidated or partially validated solver cannot validate the solver that generated it. | Blocked. |
| DPF physics publication claim | Not acceptable without source records, accepted validation packets, uncertainty, and source-backed experimental comparisons. | Blocked. |
| Generic MHD ML benchmark using The Well MHD | Potentially acceptable if the exact public dataset version, CATS paper, Well paper, license, hashes, and splits are recorded. | Allowed only as external benchmark/pretraining, not DPF validation. |

## Would It Hold Up Without The Published Articles?

No, not for scientific or validation claims.

The local data can be useful engineering material, but it would not withstand
scientific scrutiny as validation evidence without:

1. Exact provenance for each dataset: generator command, code commit, dependency
   versions, backend, hardware, config, random seed, file hashes, and generation
   logs.
2. Strict data validity: finite fields and scalars, no silent NaN/Inf
   replacement, no float32-limit saturation, nonzero and physically explainable
   magnetic fields, monotonic time, units, boundary conditions, geometry
   consistency, and conservation diagnostics.
3. Solver verification evidence: numerical method tests for the backend that
   generated the data.
4. Solver validation evidence: accepted same-scope experimental targets from
   local `KnowledgeReference/`, including uncertainty.
5. Clear context of use: whether the data support qualitative exploration,
   incremental corrections, or absolute predictive claims.
6. Published external references for public data: The Well, CATS, WALRUS, and
   any external MHD dataset source used in pretraining or benchmarking.

Published articles alone are not enough for DPF validation, but without them and
without local source records, the data should remain exploratory.

## Required Remediation Before Use

1. Add a strict WALRUS/Well dataset validator mode:
   - fail on non-finite scalar datasets,
   - fail or flag `abs(value) > threshold` saturation,
   - fail on missing `energy_conservation` for MHD training sets,
   - fail on geometry/root-attribute mismatch,
   - fail on all-zero magnetic-field datasets unless explicitly justified.
2. Add dataset manifests:
   - dataset ID, source type, code commit, command, backend, config, seed,
     dependency versions, hardware, start/end time, output hashes, split IDs,
     validator version, and validation status.
3. Regenerate DPF HDF5 training data from a vetted backend after Tier-3 numerical
   verification and run-manifest emission are stable.
4. Mark every generated dataset with a result classification:
   - `Exploratory` for current local HDF5 files,
   - `Derived Diagnostic` for Lee-model current-waveform JSON data,
   - `Preview` only after strict validator and manifest pass,
   - never `Reference` unless tied to accepted source-backed validation.
5. Keep The Well MHD data separate from DPF data:
   - external pretraining/benchmark data can be used for ML capability,
   - it cannot be mixed into DPF validation tiers.

## Bottom Line

The current local WALRUS data are useful for software development but not for
science closure. The safest path is to keep them in an exploratory bucket, add
strict validators and manifests, then regenerate clean DPF training datasets
after the solver and evidence gates are in better shape. Public WALRUS/The Well
MHD data are credible ML benchmark data when cited and versioned, but they are
not DPF experimental validation data.
