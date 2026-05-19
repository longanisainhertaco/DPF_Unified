# WP-N3 — PF-1000 / Akel Geometry Source Packet

Date: 2026-05-19
Branch: `codex/corpus`
Lane: Allowed parallel lane 1 (WP-N3 PF-1000/Akel geometry source packet),
`docs/FIRST_PRINCIPLES_CODEX_AUDIT_SPRINT2_FOLLOWUP_2026_05_19.md` lines 245-254.
Status: `source_research_packet` — NOT a validation, acceptance, or
first-principles-authority artifact.

## 0. Scope, integrity, and non-promotion statement

This packet enumerates every geometry element required to build the Auluck
`Omega` / `Sigma_p` partition masks for the PF-1000 device, together with the
local `KnowledgeReference/` (KR) citation that supports each dimension. It is a
research deliverable only.

- This packet does NOT promote validation, acceptance, or first-principles
  authority. `can_support_first_principles_acceptance` remains `false`.
- This packet does NOT mark any runtime feature `implemented`. No code or test
  diff is submitted with it.
- Every dimension below cites a local KR file with a line range and the
  figure/table/sentence carrying the number. Where KR has no source for a
  dimension, the row is marked `blocked` and the value is left empty. No
  outside material and no training-data value is used.
- This packet does NOT edit any Sprint 2.2-owned file. It only creates this
  one file.

Integrity caveat on the richest source: the KR file
`experimental-study-of-the-structure-of-the-plasma-current-sheath-on-the-pf-1000-facility-705bcc83.md`
(Krauz et al. 2012, *Plasma Phys. Control. Fusion* 54:025010,
DOI 10.1088/0741-3335/54/2/025010) carries
`KR ingestion status: text_parity_extracted_review_needed` and
`Validation status: source_available_not_target_extracted`. Its header states
"Figures, tables, plotted curves, and numeric validation targets are not
accepted until separately reviewed and target-extracted." The geometry numbers
quoted from it below appear in running prose (not in a figure or plotted
curve), so they are usable as `candidate` evidence, but they MUST NOT be
promoted to accepted validation targets until that file is target-extracted
and reviewed. The same `text_parity` caveat applies to every KR `.md` paper
extract cited here; numbers in prose are `candidate`, numbers that exist only
inside figures/plots are `blocked`.

## 1. Source-backed findings — per geometry element

All KR paths below are relative to the repo root
(`/Users/anthonyzamora/dpf-unified/`).

### KR source register (PF-1000 family)

| Tag | KR file | Identity / DOI | Ingestion status |
| --- | --- | --- | --- |
| KR-KRAUZ12 | `KnowledgeReference/experimental-study-of-the-structure-of-the-plasma-current-sheath-on-the-pf-1000-facility-705bcc83.md` | Krauz et al. 2012, Plasma Phys. Control. Fusion 54:025010, DOI 10.1088/0741-3335/54/2/025010 | `text_parity_extracted_review_needed` |
| KR-AKEL21 | `KnowledgeReference/radiation-physics-and-chemistry-188-2021-109633.md` | Akel et al. 2021, Radiat. Phys. Chem. 188:109633 | `text_parity` paper extract |
| KR-SCHOLZ07 | `KnowledgeReference/scholz-2007-pf1000-part2-jphysd.md` | Gribkov/Scholz et al. (PF-1000 part II), J. Phys. D | `text_parity` paper extract |
| KR-SCHOLZ06 | `KnowledgeReference/scholz-2006-pf1000-mega-joule.md` | Scholz et al. 2006, Nukleonika 51(1) | `text_parity` paper extract |
| KR-GRIBKOV07 | `KnowledgeReference/gribkov-2007-pf1000-jphysd-part2.md` | Gribkov et al. 2007 (PF-1000 part II), J. Phys. D | `text_parity` paper extract |
| KR-FINALSTAGES | `KnowledgeReference/final-stages-of-the-plasma-column-evolution-in-the-plasma-focus-pf1000-device-plasma-scien-fa128cfd.md` | "Final stages of the plasma column evolution in the plasma focus PF1000 device" | `text_parity` paper extract |
| KR-LEECOURSE | `KnowledgeReference/a-course-on-plasma-focus-numerical-experiments-s-lee-and-s-h-saw-part-1-basic-course.md` | Lee & Saw, "A Course on Plasma Focus Numerical Experiments", Part 1 | `text_parity` extract |
| KR-AULUCK21 | `KnowledgeReference/auluck-2021-dpf-circuit-element.md` | Auluck 2021, DPF circuit-element / power-port paper | `text_parity` extract |

NOTE on device-instance ambiguity: PF-1000 has been operated in several
hardware configurations and at several charging voltages. The KR sources
disagree on electrode dimensions because they describe DIFFERENT PF-1000
hardware revisions and/or different Lee-model fit conventions. Section 1.x
records every value found; the conflicts are NOT resolved by this packet and
are surfaced in Section 4. The runtime must select one source-tagged
configuration explicitly and may not silently average across revisions.

### 1.1 Outer electrode — 12 cathode rods (squirrel cage)

| Field | Value | Unit | KR citation |
| --- | --- | --- | --- |
| Number of rods | 12 | count | KR-KRAUZ12 lines 344-345 ("The outer electrode (OE) (cathode) consists of 12 stainless steel rods"); KR-AKEL21 lines 112-114 ("twelve 8-cm diameter stainless-steel tubes") |
| Rod diameter | 80 | mm | KR-KRAUZ12 line 345 ("12 stainless steel rods with 80 mm in diameter"); KR-AKEL21 line 113 ("twelve 8-cm diameter stainless-steel tubes") |
| Rod material | stainless steel | — | KR-KRAUZ12 line 345; KR-AKEL21 line 113; KR-SCHOLZ07 lines 216-217 ("cathode stainless steel bars") |
| Cathode-cage effective radius (b) | 200 (= 0.200 m) | mm | KR-KRAUZ12 lines 346-347 ("The OE ... radii are 200 mm") |
| Cathode-cage effective radius (b), Lee-model fit | 160 (= 16 cm) | mm | KR-AKEL21 line 264 ("Tube: b = 16 cm"); KR-LEECOURSE line 2203 ("b=16 cm") |
| Rod length | see Section 4 (conflict) | mm | KR-SCHOLZ07 lines 213-219 states two cathode revisions ("cathode stainless steel bars were much longer than the anode in the first case whereas both electrodes were equal in length in the second configuration") — no single number |
| Number of rods (alternate hardware revision) | 24 | count | KR-FINALSTAGES lines 38-39 ("outer grounded electrode of the 368-mm diameter is made of 24 stainless steel rods") — DIFFERENT revision; see Section 4 |

CONFLICT: rod count is 12 in KR-KRAUZ12 / KR-AKEL21 but 24 in
KR-FINALSTAGES; outer-electrode diameter is 400 mm (2 x 200 mm radius,
KR-KRAUZ12) vs 368 mm (KR-FINALSTAGES). These are different PF-1000 hardware
revisions. The current code (`pf1000` preset) uses `n_cathode_rods = 12` and
marks it `# UNVERIFIED` (`src/dpf/presets.py:111`); KR-KRAUZ12 line 345 and
KR-AKEL21 line 113 now supply a `candidate` source for the 12-rod count, so
the `UNVERIFIED` tag can be downgraded to `candidate` once the runtime fixes
on the Krauz/Akel hardware revision.

### 1.2 Inner electrode — anode (radius, length)

| Field | Value | Unit | KR citation |
| --- | --- | --- | --- |
| Anode (center electrode, CE) radius | 115.5 (= 0.1155 m) | mm | KR-KRAUZ12 lines 346-347 ("CE ... radii are ... 115.5 mm"); KR-AKEL21 line 264 ("a = 11.55 cm") |
| Anode diameter | 231 | mm | KR-AKEL21 line 112 ("copper anode comprises a tube of diameter 231 mm") |
| Anode diameter (alternate) | 230 | mm | KR-SCHOLZ07 line 198 ("The cylindrical copper anode (diam = 230 mm, l = 600 mm)") |
| Anode diameter (alternate revision) | 240 | mm | KR-FINALSTAGES lines 40-41 ("inner copper electrode (240-mm diameter and 450 mm in active length)") |
| Anode active length (z0) | 460 (= 0.460 m) | mm | KR-KRAUZ12 line 347 ("with CE length of 460 mm") |
| Anode length (z0), Akel shot-12581 | 480 (= 48 cm) | mm | KR-AKEL21 line 111 ("480 mm long coaxial electrodes"); KR-AKEL21 line 264 ("z0 = 48 cm") |
| Anode length (z0), Scholz-2007 hardware | 600 (= 600 mm) | mm | KR-SCHOLZ07 line 198 ("l = 600 mm") |
| Anode length (z0), Lee-course 27 kV scope | 600 (= 60 cm) | mm | KR-LEECOURSE line 2203 ("z0=60 cm") |
| Anode length (z0), alternate revision | 450 | mm | KR-FINALSTAGES line 40-41 ("450 mm in active length") |
| Anode material | copper | — | KR-AKEL21 line 112 ("copper anode"); KR-SCHOLZ07 line 198 ("copper anode"); KR-KRAUZ12 line 346 ("copper center electrode"); KR-FINALSTAGES line 40 ("inner copper electrode") |

CONFLICT: anode length is 460 mm (KR-KRAUZ12), 480 mm (KR-AKEL21), 600 mm
(KR-SCHOLZ07, KR-LEECOURSE), and 450 mm (KR-FINALSTAGES). Anode diameter is
231 / 230 / 240 mm across sources. These reflect different PF-1000 hardware
revisions and different Lee-model fit periods. The current code carries
multiple presets that each pick a self-consistent source set
(`pf1000` -> z0 = 0.60 m Lee/Malek scope; `pf1000_akel` -> z0 = 0.48 m Akel
scope, `src/dpf/presets.py:109-110,174-186`). The runtime mask build MUST
inherit z0 and the anode radius from a single chosen source-tagged
configuration.

### 1.3 Anode hollow bore

| Field | Value | Unit | KR citation |
| --- | --- | --- | --- |
| Anode is hollow (axial bore present) | yes (qualitative) | — | KR-KRAUZ12 lines 372-373 ("probes were introduced from the collector side through the vacuum lock along the axis of the hollow anode") |
| Anode bore radius | NOT FOUND | mm | No KR source. `blocked`. |
| Anode bore length | NOT FOUND | mm | No KR source. `blocked`. |
| Anode end-flange / cap geometry | qualitative only | — | KR-SCHOLZ07 lines 198-201 ("closed by a lid ... a circular hat-shaped 'cap' at its end" — diameter "same or a slightly larger" than the tube, no number) |

UNRESOLVED: the PF-1000 anode is explicitly hollow (KR-KRAUZ12 line 373), but
no KR source in the corpus gives a numeric bore radius or bore length for
PF-1000. The probe-access radii in KR-KRAUZ12 lines 375-377 ("near the anode
surface at radii of 1.2 and 4 cm") describe diagnostic probe placement, NOT
the bore wall, and must NOT be used as a bore dimension. The hollow-anode
dimensions in `KnowledgeReference/a-thesis-...md` lines 1957-1958, 2554 and in
KR-LEECOURSE lines 454, 1413, 1456 describe OTHER (non-PF-1000) devices and
the Lee-model generic "max length (hollow anode) z = 1.5a..1.6a" rule, not a
measured PF-1000 bore. The bore is therefore `blocked` — the runtime must
treat the anode as solid for the mask build, or expose the bore as a
`blocked` field, and may NOT invent a bore radius.

### 1.4 Alumina insulator

| Field | Value | Unit | KR citation |
| --- | --- | --- | --- |
| Insulator material | alumina | — | KR-KRAUZ12 line 348 ("cylindrical alumina insulator"); KR-SCHOLZ07 line 223 ("An alumina insulator"); KR-FINALSTAGES line 42 ("cylindrical alumina insulator") |
| Insulator placement | sleeves the lower part of the anode (CE) | — | KR-KRAUZ12 lines 348-349 ("cylindrical alumina insulator sits on the CE"); KR-SCHOLZ07 line 223 ("envelops the anode at its lower part"); KR-GRIBKOV07 lines 57-58 ("cylindrical insulator positioned at the lower part of the internal electrode (anode)") |
| Insulator exposed (operating) length | 85 (= 0.085 m) | mm | KR-KRAUZ12 lines 349-350 ("the main part of the insulator extends 85 mm along the CE into the vacuum chamber") |
| Insulator exposed length (alternate revision) | 113 (= 0.113 m) | mm | KR-SCHOLZ07 lines 223-225 ("Its main part extends 113 mm along the anode into the vacuum chamber"); KR-FINALSTAGES lines 42-43 ("113 mm in operating length") |
| Insulator outer radius | NOT FOUND | mm | No KR source. `blocked`. |
| Insulator wall thickness | NOT FOUND | mm | No KR source. `blocked`. |

CONFLICT: insulator exposed length is 85 mm (KR-KRAUZ12) vs 113 mm
(KR-SCHOLZ07, KR-FINALSTAGES). KR-KRAUZ12 lines 356-358 explicitly note "a new
alumina insulator" was installed, which is the likely reason for the
discrepancy. The insulator outer radius and wall thickness are absent from all
KR sources — the inner radius is bounded below by the anode radius
(115.5 mm), but the outer radius is `blocked`.

### 1.5 Backplate / source interface

| Field | Value | Unit | KR citation |
| --- | --- | --- | --- |
| Backplate exists; insulator sets the initial sheath between CE and the OE back plate | yes (qualitative) | — | KR-KRAUZ12 lines 351-352 ("The insulator prescribes the shape of the initial current sheet between the CE and the back plate of the OE") |
| Auluck source-interface definition (the surface excluded from `Omega`) | "the cathode plate that is in contact with the insulator and the squirrel cage" | — | KR-AULUCK21 lines 203-209 ("Excluded from this domain is the interface between the 'circuit element' and the external power source ... this would be the cathode plate that is in contact with the insulator and the squirrel cage") |
| Backplate radial extent | NOT FOUND | mm | No KR source. `blocked`. |
| Backplate axial thickness | NOT FOUND | mm | No KR source. `blocked`. |
| Backplate axial position | z = 0 (breech plane, by construction of z0) | — | INFERRED from the anode-length definition (z0 is measured from the breech). Not a directly stated KR dimension; flagged. |

The backplate IS the Auluck source interface (the surface through which power
enters the device, KR-AULUCK21 lines 205-209). Its existence and role are
source-backed; its numeric radial/axial dimensions are `blocked`. For the mask
build it is sufficient to identify the breech axial slice (z = z_port);
matching the existing code, `terminal_source_interface_faces` is the
`k = k_port` axial slab (`src/dpf/fields/source_geometry.py:168-173`).

### 1.6 Chamber wall (vacuum vessel)

| Field | Value | Unit | KR citation |
| --- | --- | --- | --- |
| Chamber inner diameter | 1400 (= 1.4 m) | mm | KR-KRAUZ12 lines 342-343 ("The vacuum chamber, which surrounds the electrodes, has a large volume (1400 mm in diameter ...") |
| Chamber length | 2500 (= 2.5 m) | mm | KR-KRAUZ12 line 343 ("... and 2500 mm in length)") |
| Current-collector diameter | 3000 (= 3 m) | mm | KR-SCHOLZ07 lines 191-192 ("a collector of diameter 3 m") — collector, NOT the vacuum-chamber wall |
| Chamber wall material | NOT FOUND | — | No KR source. `blocked`. |
| Chamber wall thickness | NOT FOUND | — | No KR source. `blocked`. |

The chamber inner diameter (1400 mm) and length (2500 mm) are source-backed by
KR-KRAUZ12. Note the current `pf1000` preset uses `cathode_radius = 0.16 m`
(`src/dpf/presets.py:110`) which is the Lee-model effective cathode radius
b = 16 cm (KR-AKEL21 line 264 / KR-LEECOURSE line 2203), NOT the 1400 mm
chamber wall and NOT the 200 mm cathode-cage radius. The mask build must keep
the chamber wall (700 mm radius), the cathode-cage radius (160 or 200 mm),
and the anode radius (115.5 mm) as three distinct surfaces.

### 1.7 Open / PML boundary

| Field | Value | Unit | KR citation |
| --- | --- | --- | --- |
| Physical open end of the device | the open end of the coaxial electrodes (qualitative) | — | KR-SCHOLZ06 lines 25-30 ("formation of a current sheath at the insulator surface ... accelerated towards the open end of the electrodes") |
| Auluck outer plasma boundary `Sigma` | the extreme boundary of the plasma separating the current-carrying region from the current-free region | — | KR-AULUCK21 lines 211-217 ("the extreme boundary of the plasma bridging the two electrodes that separates the current carrying region of the plasma from the current free region") |
| PML layer count | NOT a device dimension | — | A numerical absorbing-boundary parameter. No KR device source applies. Code default: `axial_pml_layers = 20` (`src/dpf/fields/source_geometry.py:46`) for the LLNL-like 2-D source, NOT PF-1000. |
| PML thickness / placement for PF-1000 | NOT FOUND | — | No KR source; numerical choice. `candidate` only, must be tagged as a solver parameter. |

The PML / open boundary is a numerical absorbing layer, not a measured device
dimension. KR has no PF-1000 PML source and none should be expected. The
runtime must classify the PML faces as `open_pml_faces` (the existing fourth
Auluck label, `src/dpf/fields/source_geometry.py:18-23`) and record the layer
count and thickness as solver parameters with no physics-source claim.

### 1.8 Plasma domain (Auluck `Omega`)

| Field | Value | Unit | KR citation |
| --- | --- | --- | --- |
| `Omega` definition | 3-D spatial domain such that current density J is zero outside it | — | KR-AULUCK21 lines 203-204 ("This 3-D spatial integration is over a domain ... such that J is zero outside it") |
| `Omega` excludes the source interface | yes | — | KR-AULUCK21 lines 205-209 ("Excluded from this domain is the interface between the 'circuit element' and the external power source") |
| `Omega` topology (pre-breakup) | toroid (not simply connected) | — | KR-AULUCK21 lines 218-223 ("topologically it is a toroid") |
| Bounding surface `Sigma` | plasma-electrode contact surfaces + the extreme current-carrying plasma boundary | — | KR-AULUCK21 lines 211-217 |
| `Sigma_p` (moving boundary) | the moving sub-boundary of `Omega`; only `Sigma_p` contributes to the motional surface integral; stationary boundaries contribute zero | — | KR-AULUCK21 lines 426-429 ("The second integral is evaluated only on the moving boundary `Sigma_p` of the domain ... stationary boundaries do not contribute to it") |
| Plasma-boundary transition-region width | < 1 mm (density falls ~2 orders of magnitude) | mm | KR-AULUCK21 lines 211-214 ("the dense plasma has a boundary region where the density falls over two orders of magnitude over a radial distance less than 1 mm") |
| Plasma domain absolute dimensions | NOT a fixed dimension (time-dependent) | — | `Omega` geometry is a function of time (KR-AULUCK21 lines 429-431, "the geometry of the integration domains is a function of time"); it is derived at runtime from the n_e and J fields, not a static device dimension. |

`Omega` is NOT a static geometry: it is the current-carrying plasma volume,
derived per timestep from the runtime n_e and J fields. The existing code
already builds it this way (`build_auluck_omega_domain`,
`src/dpf/fields/source_geometry.py:126-251`). The static device geometry above
(rods, anode, insulator, chamber) sets the *container* surfaces that bound
`Omega`; `Omega` and `Sigma_p` themselves are runtime-derived.

## 2. Blocker table — `supported` / `candidate` / `blocked`

Classification rule used:
- `supported` — a numeric value exists in KR running prose of a paper whose
  extract is text-parity available (usable as a candidate validation input,
  not yet a promoted target).
- `candidate` — a value exists but sources conflict (multiple PF-1000 hardware
  revisions), so no single value is authoritative, OR the value is a numerical
  solver parameter not a measured dimension.
- `blocked` — no KR source provides the dimension.

| # | Geometry element | Dimension | Class | Value (if any) | Primary KR source |
| --- | --- | --- | --- | --- | --- |
| 1 | Cathode rods | count = 12 | candidate | 12 (conflicts with 24 in KR-FINALSTAGES) | KR-KRAUZ12 344-345 |
| 2 | Cathode rods | rod diameter = 80 mm | supported | 80 mm | KR-KRAUZ12 345; KR-AKEL21 113 |
| 3 | Cathode rods | rod material = stainless steel | supported | stainless steel | KR-KRAUZ12 345 |
| 4 | Cathode rods | rod length | blocked | (two unequal revisions, no number) | KR-SCHOLZ07 213-219 |
| 5 | Cathode cage | effective radius b | candidate | 200 mm (KR-KRAUZ12) vs 160 mm Lee-fit (KR-AKEL21) | KR-KRAUZ12 346-347 |
| 6 | Anode | radius a = 115.5 mm | supported | 115.5 mm | KR-KRAUZ12 346-347; KR-AKEL21 264 |
| 7 | Anode | length z0 | candidate | 460 / 480 / 600 / 450 mm across revisions | KR-KRAUZ12 347; KR-AKEL21 264 |
| 8 | Anode | material = copper | supported | copper | KR-AKEL21 112; KR-SCHOLZ07 198 |
| 9 | Anode | hollow bore radius | blocked | — | none |
| 10 | Anode | hollow bore length | blocked | — | none |
| 11 | Anode | end-cap / lid geometry | blocked | qualitative only | KR-SCHOLZ07 198-201 |
| 12 | Alumina insulator | material = alumina | supported | alumina | KR-KRAUZ12 348; KR-SCHOLZ07 223 |
| 13 | Alumina insulator | exposed length | candidate | 85 mm (KR-KRAUZ12) vs 113 mm (KR-SCHOLZ07) | KR-KRAUZ12 349-350 |
| 14 | Alumina insulator | outer radius | blocked | — | none |
| 15 | Alumina insulator | wall thickness | blocked | — | none |
| 16 | Backplate / source interface | existence + Auluck role | supported | qualitative (is the Auluck source interface) | KR-KRAUZ12 351-352; KR-AULUCK21 205-209 |
| 17 | Backplate / source interface | radial extent | blocked | — | none |
| 18 | Backplate / source interface | axial thickness | blocked | — | none |
| 19 | Chamber wall | inner diameter = 1400 mm | supported | 1400 mm | KR-KRAUZ12 342-343 |
| 20 | Chamber wall | length = 2500 mm | supported | 2500 mm | KR-KRAUZ12 343 |
| 21 | Chamber wall | material | blocked | — | none |
| 22 | Chamber wall | wall thickness | blocked | — | none |
| 23 | Open / PML boundary | physical open end exists | supported | qualitative | KR-SCHOLZ06 25-30 |
| 24 | Open / PML boundary | PML layer count / thickness | candidate | solver parameter, not a device dimension | (numerical) |
| 25 | Plasma domain `Omega` | definition + source-interface exclusion | supported | runtime-derived (not static) | KR-AULUCK21 203-209 |
| 26 | Plasma domain `Sigma_p` | moving-boundary definition | supported | runtime-derived (not static) | KR-AULUCK21 426-429 |

Counts: `supported` = 12, `candidate` = 5, `blocked` = 9. Total rows = 26.

## 3. Runtime fields required to build the masks

The runtime must expose the following so a deterministic reviewed mask builder
can construct `Omega` and the per-class material masks. Fields are grouped by
origin.

### 3.1 Static geometry config fields (device, source-tagged)

The runtime must accept ONE `geometry_source_tag` (e.g.
`pf1000_krauz2012`, `pf1000_akel_shot12581`, `pf1000_lee_malek_27kv`) and, for
that tag only, the following typed fields:

- `geometry_source_tag: str` — selects a single self-consistent KR source set.
- `geometry_source_refs: tuple[str, ...]` — KR `path:line-range` strings for
  every field, mirroring `AULUCK_OMEGA_SOURCE_REFS`
  (`src/dpf/fields/source_geometry.py:15-17`).
- `anode_radius_m: float` — supported (row 6).
- `anode_length_m: float` — candidate, MUST equal the z0 of the chosen tag
  (row 7).
- `anode_material: str` — supported (row 8).
- `anode_hollow_bore_radius_m: float | None` — `None` until row 9 unblocked.
- `cathode_rod_count: int` — candidate, default 12 for Krauz/Akel tag (row 1).
- `cathode_rod_diameter_m: float` — supported (row 2).
- `cathode_cage_radius_m: float` — candidate, MUST record whether it is the
  geometric 200 mm or the Lee-fit 160 mm (rows 5).
- `insulator_material: str` — supported (row 12).
- `insulator_exposed_length_m: float` — candidate (row 13).
- `insulator_outer_radius_m: float | None` — `None` until row 14 unblocked.
- `chamber_inner_radius_m: float` — supported, 0.700 m (row 19).
- `chamber_length_m: float` — supported, 2.5 m (row 20).
- `source_interface_axial_index: int` — the breech (z = 0) axial slice index
  (row 16); reuses `source_interface_z_index`
  (`src/dpf/fields/source_geometry.py:131,154-157`).
- `pml_layers: int` and `pml_thickness_m: float` — solver parameters (row 24),
  tagged non-physics.

### 3.2 Runtime field arrays required for `Omega` / `Sigma_p`

`Omega` is time-dependent and derived, so the runtime must additionally emit,
on the simulation grid:

- `electron_density_m3: ndarray` — cell-centered; `Omega` membership test
  (`src/dpf/fields/source_geometry.py:161-189`).
- `current_density_norm_A_m2: ndarray` — `|J|`; `Omega` current-carrying test;
  Auluck requires J = 0 outside `Omega` (KR-AULUCK21 203-204).
- `electron_density_floor_m3: float` — the telemetry floor used, so the
  membership threshold is auditable.
- `grid_shape: tuple[int,int,int]` and per-axis cell sizes / coordinates.
- For `Sigma_p` (consumed by the separate WP-N3 `Sigma_p` runtime-interface
  spec, lane 2): face-centered `B`, `E`, `J`, `v`, `eta`, outward `dS`, face
  area, and centering metadata on the moving boundary. This packet only
  records that those fields are REQUIRED; their schema is lane 2's deliverable.

### 3.3 Mask outputs the runtime must expose

Per the existing partition contract
(`build_auluck_omega_domain`, `src/dpf/fields/source_geometry.py:196-251`):

- one boolean mask per class: `omega_volume_cells`,
  `terminal_source_interface_faces`, `wall_material_faces`, `open_pml_faces`;
- per-class SHA-256 hash (`_mask_sha256`,
  `src/dpf/fields/source_geometry.py:79-85`);
- per-class index-space bounding box (`_mask_bounds`);
- `source_refs` on every class packet;
- partition constraint flags: `mutually_disjoint`, `exhaustive`,
  `terminal_source_interface_disjoint_from_omega`.

For Sprint 3 the `wall_material_faces` class must be SPLIT into source-tagged
sub-classes (anode / cathode-rods / insulator / chamber-wall / backplate), each
with its own hash and KR `source_refs`, instead of the current single
exhaustive-complement class.

## 4. Missing parameters — dimensions with NO KR source

The following dimensions have NO supporting KR source in the corpus. They are
`blocked`. The runtime MUST expose each as `None` / `blocked` and MUST NOT
invent a value.

1. Anode hollow bore radius (row 9). Anode is confirmed hollow
   (KR-KRAUZ12 373) but no numeric bore radius exists.
2. Anode hollow bore length (row 10).
3. Anode end-cap / lid diameter and thickness (row 11) — KR-SCHOLZ07 198-201
   is qualitative ("same or a slightly larger diameter").
4. Cathode rod length (row 4) — KR-SCHOLZ07 213-219 describes two unequal
   revisions with no number.
5. Alumina insulator outer radius (row 14).
6. Alumina insulator wall thickness (row 15).
7. Backplate radial extent (row 17).
8. Backplate axial thickness (row 18).
9. Chamber wall material and wall thickness (rows 21-22) — only the inner
   bore (1400 mm dia) and length (2500 mm) are sourced.

Unresolved CONFLICTS (sourced, but sources disagree — must be pinned by
`geometry_source_tag`, not averaged):

- Cathode rod count: 12 (KR-KRAUZ12, KR-AKEL21) vs 24 (KR-FINALSTAGES).
- Outer-electrode diameter: 400 mm (KR-KRAUZ12) vs 368 mm (KR-FINALSTAGES).
- Cathode effective radius b: 200 mm geometric (KR-KRAUZ12) vs 160 mm Lee-fit
  (KR-AKEL21, KR-LEECOURSE).
- Anode length z0: 460 / 480 / 600 / 450 mm.
- Anode diameter: 231 / 230 / 240 mm.
- Insulator exposed length: 85 mm (KR-KRAUZ12) vs 113 mm (KR-SCHOLZ07,
  KR-FINALSTAGES).

## 5. Proposed tests and fail-closed negative controls

These are PROPOSED tests for the Sprint 3 implementation diff. They are NOT
implemented here (no code/test diff is submitted with this research packet).
They extend `tests/test_source_geometry_packet.py`.

Positive / structural tests:

- `test_geometry_masks_are_mutually_disjoint` — every grid cell carries
  exactly one class label (`per_cell_label_count <= 1`).
- `test_geometry_masks_are_exhaustive` — every grid cell carries at least one
  class label (`per_cell_label_count == 1`).
- `test_omega_excludes_source_interface` — `omega_volume_cells &
  terminal_source_interface_faces` is empty (Auluck KR-AULUCK21 205-209).
- `test_each_material_subclass_has_distinct_hash` — anode, cathode-rods,
  insulator, chamber-wall, backplate masks each produce a different
  `mask_sha256`.
- `test_every_mask_class_carries_source_refs` — each class packet has a
  non-empty `source_refs` list of KR `path:line` strings.
- `test_moving_boundary_faces_nonempty_in_synthetic_case` — for a controlled
  synthetic moving-`Sigma_p` case, the moving-boundary face set is non-empty.
- `test_stationary_boundaries_contribute_zero_to_motional_terms` — for a
  controlled synthetic case the stationary chamber-wall / backplate faces
  contribute exactly zero to the motional surface integral
  (KR-AULUCK21 426-429).
- `test_geometry_manifest_lists_all_mask_hashes_and_source_refs` — the emitted
  manifest contains every per-class hash and every KR source reference.

Fail-closed negative controls (must FAIL CLOSED, i.e. raise an attributable
error or return a `blocked` status — never silently default):

- `test_missing_geometry_source_tag_fails_closed` — building masks without a
  `geometry_source_tag` raises.
- `test_blocked_bore_radius_stays_none` — `anode_hollow_bore_radius_m` is
  `None` and any consumer that needs it fails closed; no numeric default is
  injected.
- `test_blocked_insulator_outer_radius_stays_none` — same for the insulator
  outer radius.
- `test_conflicting_dimension_not_averaged` — supplying two source tags, or a
  z0 that matches no tag, raises; the builder never averages 460/480/600 mm.
- `test_chamber_radius_not_confused_with_cathode_radius` — a config that sets
  `chamber_inner_radius_m` equal to `cathode_cage_radius_m` raises (1400 mm
  wall vs 160-200 mm cage are distinct surfaces).
- `test_overlapping_masks_fail_closed` — a constructed overlap between any two
  material sub-classes raises, not silently merged.
- `test_non_exhaustive_partition_fails_closed` — an uncovered cell raises.
- `test_omega_with_source_interface_overlap_fails_closed` — if `Omega`
  intersects the source interface the builder raises.
- `test_missing_mask_hash_fails_manifest` — a manifest missing any per-class
  hash is rejected.
- `test_pml_layers_not_treated_as_physics` — PML layer count carries a
  non-physics tag and no KR `source_refs`; a test asserts it is never labelled
  a sourced device dimension.

## 6. Exact implementation recommendations

Build deterministic, reviewed masks with per-class hashes as follows. These
are recommendations for the Sprint 3 implementation diff; nothing here is
`implemented` until code and tests are submitted together.

1. Add a frozen `PF1000GeometryPacket` dataclass alongside
   `HybridPICSourceGeometry` in `src/dpf/fields/source_geometry.py`, carrying
   the Section 3.1 fields. Every field that is `blocked` (Section 4) is typed
   `float | None` and defaults to `None`. Every `candidate`/`supported` field
   carries a companion `*_source_ref: str` KR `path:line-range`.

2. Add a `PF1000_GEOMETRY_SOURCE_REFS` tuple (mirroring
   `AULUCK_OMEGA_SOURCE_REFS`, `src/dpf/fields/source_geometry.py:15-17`)
   listing every KR citation used. A test asserts each file in the tuple
   exists under `KnowledgeReference/`.

3. Provide named source-tagged constructors, one per self-consistent KR set,
   e.g. `PF1000GeometryPacket.krauz_2012()`,
   `PF1000GeometryPacket.akel_shot_12581()`. Each constructor hard-codes ONLY
   the dimensions sourced for that revision; conflicting fields differ between
   constructors. No constructor averages across revisions.

4. Extend `build_auluck_omega_domain` (or add
   `build_pf1000_material_partition`) to split the current single
   `wall_material_faces` exhaustive-complement class into source-tagged
   material sub-classes: `anode_material_faces`, `cathode_rod_faces`,
   `insulator_material_faces`, `chamber_wall_faces`,
   `backplate_source_interface_faces`. Keep the four Auluck top-level labels;
   the material sub-classes refine `wall_material_faces`.

5. Reuse `_mask_sha256` (`src/dpf/fields/source_geometry.py:79-85`) to emit one
   deterministic SHA-256 per material sub-class. The hash covers
   `array.shape` then the packed `uint8` view — keep that exact algorithm so
   hashes are reproducible and reviewable. Emit all hashes into the geometry
   manifest.

6. Determinism: build every mask purely from the static config + grid
   (no RNG, no floor-dependent branch except the documented n_e/J thresholds).
   Sort any cell-index iteration. The same config + grid must always yield
   identical hashes — assert this in `test_mask_hash_is_deterministic`.

7. Keep the existing partition guarantees: `mutually_disjoint`, `exhaustive`,
   `terminal_source_interface_disjoint_from_omega`,
   `omega_contains_only_current_carrying_cells`
   (`src/dpf/fields/source_geometry.py:203-247`). Add disjointness/exhaustive
   checks ACROSS the new material sub-classes.

8. Add an under-resolution gate: if the grid cell size does not resolve the
   smallest sourced feature (insulator exposed length 85 mm, rod diameter
   80 mm, sub-millimeter plasma transition region KR-AULUCK21 211-214), the
   builder returns a `blocked`/under-resolved status rather than a mask. This
   matches the Sprint 3 "under-resolution gate" requirement.

9. Emit a geometry manifest section containing: `geometry_source_tag`, every
   field value, every `*_source_ref`, every per-class mask hash, the
   under-resolution verdict, and `can_support_first_principles_acceptance:
   false`. The blocked fields appear explicitly as `null` with a
   `blocked_reason`.

10. The `Omega` / `Sigma_p` build itself stays runtime-derived from n_e and J
    (Section 1.8); the static packet supplies only the bounding container
    surfaces. Do NOT hardcode `Omega` extents.

## 7. "Do not promote" notes (mandatory)

- DO NOT promote this packet, or any geometry mask built from it, to a
  validation, acceptance, or first-principles-authority artifact.
  `can_support_first_principles_acceptance` stays `false`; the existing
  `geometry_review_status: "geometry_candidate_not_reviewed"` and
  `can_support_power_port_acceptance: false`
  (`src/dpf/fields/source_geometry.py:248-250`) must stay until a reviewed
  PF-1000 geometry packet exists.
- DO NOT treat the KR-KRAUZ12 numbers as accepted validation targets: that KR
  file is `text_parity_extracted_review_needed` /
  `source_available_not_target_extracted`. Its prose dimensions are
  `candidate` evidence only.
- DO NOT resolve the Section 4 conflicts by averaging. Each PF-1000 hardware
  revision is a distinct device; the runtime must pin one
  `geometry_source_tag`.
- DO NOT invent the anode bore radius/length, the insulator outer
  radius/wall, the backplate dimensions, the cathode rod length, or the
  chamber wall material/thickness. They are `blocked` (Section 4). The runtime
  must expose them as `None` with a `blocked_reason` and fail closed in any
  consumer that needs them.
- DO NOT mark any Sprint 3 runtime feature `implemented` on the basis of this
  packet: per the parallel-deliverable acceptance rules, an `implemented`
  claim requires code and tests in the same submitted diff.
- DO NOT use the PF-1000 probe-access radii (KR-KRAUZ12 375-377) or the
  generic Lee-model "hollow anode z = 1.5a..1.6a" rule (KR-LEECOURSE
  1413-1456) as a PF-1000 bore dimension — they describe diagnostics and a
  different modeling abstraction respectively.
- The downstream `Sigma_p` field schema (face-centered B/E/J/v/eta, outward
  dS, centering metadata) is owned by the WP-N3 `Sigma_p` runtime-interface
  spec (lane 2) and is NOT specified or implemented here.

## 8. KR sources used by this packet

- `KnowledgeReference/experimental-study-of-the-structure-of-the-plasma-current-sheath-on-the-pf-1000-facility-705bcc83.md`
  (Krauz et al. 2012) — lines 342-352, 356-358, 372-377.
- `KnowledgeReference/radiation-physics-and-chemistry-188-2021-109633.md`
  (Akel et al. 2021) — lines 111-114, 264-268.
- `KnowledgeReference/scholz-2007-pf1000-part2-jphysd.md` — lines 191-201,
  213-225.
- `KnowledgeReference/scholz-2006-pf1000-mega-joule.md` — lines 22-33.
- `KnowledgeReference/gribkov-2007-pf1000-jphysd-part2.md` — lines 56-63.
- `KnowledgeReference/final-stages-of-the-plasma-column-evolution-in-the-plasma-focus-pf1000-device-plasma-scien-fa128cfd.md`
  — lines 38-43.
- `KnowledgeReference/a-course-on-plasma-focus-numerical-experiments-s-lee-and-s-h-saw-part-1-basic-course.md`
  — lines 2199-2210, 1413-1456.
- `KnowledgeReference/auluck-2021-dpf-circuit-element.md` — lines 203-223,
  426-431.

Codebase files inspected (read-only, not edited):
`src/dpf/presets.py`, `src/dpf/fields/source_geometry.py`,
`src/dpf/validation/experimental.py`,
`src/dpf/validation/experimental_devices.py`,
`tests/test_source_geometry_packet.py`.
