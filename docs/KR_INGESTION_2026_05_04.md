# KR Ingestion — 2026-05-04

Mass-ingest of DPF-relevant PDFs from `archive_reference_OLD/references/papers/` into
`KnowledgeReference/`.  Provenance tag form: `[KR: <slug>.md §<sec> p.<page>]`.

## Inventory Summary

| metric | value |
|---|---|
| Total PDFs in archive (all paths) | 952 |
| Unique PDFs by basename | 569 |
| Already-ingested (KR before today) | 379 |
| Not-ingested before today | 190 |
| HIGH-priority unconverted (DPF authors / devices) | 19 |
| HIGH duplicates of existing KR content (md5 match) | 4 |
| HIGH unique to ingest | 15 |
| Newly-ingested this session | 15 |
| Schema validate FAIL | 0 |

Cap of 20 papers per session honored.

## Duplicates Skipped (md5 match to already-ingested)

| Candidate PDF | Already in KR as |
|---|---|
| `offermann-2021-mjolnir-dpf.pdf` | `1860821.pdf` |
| `auluck-2023-generalized-dpf-propulsion.pdf` | `2023_Generalized_plasma_focus_problem_and_its_application_to_space_propulsion.pdf` |
| `auluck-2024-poloidal-bfield-dpf-preprint.pdf` | `Poloidal_magnetic_field_in_the_dense_plasma_focus.pdf` |
| `seyler-2014-3d-dpf.pdf` | `1402.5083v1.pdf` |

## Ingested This Session (all PASS schema validation)

| Source PDF | KR file (slug) | Schema | Key anchors found in MD |
|---|---|---|---|
| `auluck-2021-dpf-circuit-element.pdf` | `auluck-2021-dpf-circuit-element` | PASS | PF-1000 interpreted-inductance discussion |
| `auluck-2022-dpf-theory-part1.pdf` | `auluck-2022-dpf-theory-part1` | PASS | Lee Model citation |
| `auluck-2022-filamentation.pdf` | `auluck-2022-filamentation` | PASS | PF-1000 deuterium filaments |
| `auluck-2022-poloidal-flux-emission.pdf` | `auluck-2022-poloidal-flux-emission` | PASS | snowplow-Vargas hypothesis ref |
| `auluck-2023-poloidal-flux-survey.pdf` | `auluck-2023-poloidal-flux-survey` | PASS | Lee model + PF-1000 + 2D snowplow |
| `beresnyak_2018_dpf_hawk_simulations.pdf` | `beresnyak_2018_dpf_hawk_simulations` | PASS | "drive parameter Imax/a*sqrt(rho)" definition |
| `esaulov_2003_2d_mhd_dpf.pdf` | `esaulov_2003_2d_mhd_dpf` | PASS | snowplow accretion + 2D MHD |
| `goyon-2022-mjolnir-high-low.pdf` | `goyon-2022-mjolnir-high-low` | PASS | MJOLNIR snowplow fit, 3.3 MA peak, 4.1e11 n/pulse |
| `malir-2024-interferometry-dpf.pdf` | `malir-2024-interferometry-dpf` | PASS | PF-1000 reconstruction, 0D Lee model, 2-3 MA peak at 40 kV |
| `scholz-2007-pf1000-part2-jphysd.pdf` | `scholz-2007-pf1000-part2-jphysd` | PASS | PF-1000 27 kV waveforms (calibration target) |
| `seyler-2021-kr-doped-dpf-mhd.pdf` | `seyler-2021-kr-doped-dpf-mhd` | PASS | Lee model ref, 35/40/45/50 kV scaling, drive parameter constant |
| `beresnyak_2022_pulsed_power_ideal_mhd.pdf` | `beresnyak_2022_pulsed_power_ideal_mhd` | PASS | (large 419k MD; ideal-MHD pulsed-power textbook) |
| `lee_radpf_theory.pdf` | `lee_radpf_theory` | PASS | RADPF code theory: Lee Model, axial snowplow, peak current I0 |
| `snowplow_modeling_plasma_switch.pdf` | `snowplow_modeling_plasma_switch` | PASS | Long-conduction snowplow modeling |
| `2025_Double_3 MJ_dense_plasma_focus_for_thermonuclear_drive_inertial_confinement_fusi.pdf` | `2025-double-3mj-dense-plasma-focus-thermonuclear-icf` | PASS | I0=3.54 MA at 50 kV, c=b/a=1.5, S=10^4 target gain |

Note: the Double_3MJ source filename had a literal space; ingested via temp copy and the
`source_pdf` field was patched back to the original filename so provenance is preserved.

## Top-5 Most Useful New Ingestions

Ranked by direct utility for DPF-Unified physics validation:

1. **`scholz-2007-pf1000-part2-jphysd`** — PF-1000 part-2 J Phys D paper.  Anchors the
   27 kV charging-voltage waveforms used as the primary calibration target.  Already
   referenced in MEMORY ("PF-1000 at 27 kV: 11.5% I_peak error"). Pairs with the existing
   `scholz-2006-pf1000-mega-joule` and `gribkov-2007-pf1000-jphysd-part2` to form the
   complete Scholz/Gribkov PF-1000 trilogy.

2. **`lee_radpf_theory`** — Lee S, "Plasma Focus Model (Radiative) - S Lee Model" theory
   document.  Anchors the axial-phase snowplow trajectory derivation and the
   `I0 = peak current of L0-C0 discharge` definition.  This is the canonical RADPF
   reference; pairs with the Lee Model code source files for direct equation-by-equation
   verification.

3. **`seyler-2021-kr-doped-dpf-mhd`** — Cornell 3D MHD code applied to Kr-doped DPF.
   Anchors the drive-parameter-constant scaling rule (`I0/(a*sqrt(rho_0))` constant when
   varying voltage from 35 to 50 kV).  Critical for any sweep-the-charge-voltage validation
   in DPF-Unified.

4. **`beresnyak_2018_dpf_hawk_simulations`** — Beresnyak et al, HAWK device DPF
   simulations.  Anchors the explicit definition of the "drive parameter"
   (`Imax/(a * p^(1/2))`) and provides an alternate (HAWK, NRL) device simulation reference
   that complements the existing PF-1000 / NX2 / MJOLNIR set.

5. **`goyon-2022-mjolnir-high-low`** — Goyon et al, MJOLNIR DPF high/low pressure regimes.
   Anchors the snowplow-fit-to-experimental-current-trace methodology and the
   3.3 MA peak / 4.1e11 n-per-pulse benchmark.  Complements `petrov-2022-mjolnir-high-low`
   already in KR (cross-validation of Goyon experiment vs Petrov simulation).

## Remaining Unconverted, by Priority

### HIGH not yet ingested (hash-duplicates already in KR)
- offermann-2021-mjolnir-dpf -> 1860821 (already in KR)
- auluck-2023-generalized-dpf-propulsion -> already in KR under renamed slug
- auluck-2024-poloidal-bfield-dpf-preprint -> Poloidal_magnetic_field_... already in KR
- seyler-2014-3d-dpf -> 1402.5083v1 already in KR

These do not need to be re-ingested (content is identical).

### MEDIUM priority (Z-pinch / general MHD / pinch other devices) — not ingested
18 unique PDFs identified.  Defer to a follow-on session (next cap-of-20 batch).
Examples (full list available via `python3 -c` filter on `archive_reference_OLD/`):

| keyword | example file |
|---|---|
| z-pinch | `gas_puff_z_pinch_*` |
| sheath | `2026_Effects_of_parallel_magnetic_fields_on_sheaths_*` |
| MHD | various AMRVAC + extended-MHD numerical papers |
| x-pinch | `sph-xpinch-2024-*.pdf` |

### LOW priority — 153 PDFs
Tangential physics (general plasma, GAMMA10 mirror, terahertz, image-feature DPF
detection, off-domain).  Skip for KR purposes.

## Mechanics

- Tool: `python3 scripts/extract_papers.py <pdf> --out KnowledgeReference/`
- Validator: `python3 scripts/validate_kr_schema.py KnowledgeReference/<file>.json`
- All 15 ingested files passed schema validation with no errors and no schema-drift warnings.
- KnowledgeReference/ is gitignored; this ingestion will not appear in `git status`.
