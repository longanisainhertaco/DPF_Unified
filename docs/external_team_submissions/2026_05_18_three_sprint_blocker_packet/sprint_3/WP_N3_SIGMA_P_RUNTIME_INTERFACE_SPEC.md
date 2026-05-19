# WP-N3 — Sigma_p Runtime-Interface Spec (Auluck eq. (6) terms II/IV/V/VI)

Date: 2026-05-19
Branch: `codex/corpus`
Status: `source_backed_interface_spec_not_validation_not_acceptance`
Owner lane: Codex audit follow-up "Allowed parallel lanes" #2
(`docs/FIRST_PRINCIPLES_CODEX_AUDIT_SPRINT2_FOLLOWUP_2026_05_19.md`).

## 0. Purpose and scope

This packet specifies the **exact runtime data interface** the simulator must
expose so that `src/dpf/first_principles/power_port.py` can compute Auluck
eq. (6) terms **II, IV, V, VI** — the `Sigma_p` moving-boundary surface
integrals — **each independently, without closure substitution** (no term
derived as `I*V` minus the others).

This is a research/interface spec only. It:

- creates exactly one file (this file);
- edits no existing file and ships no code;
- does NOT promote validation, acceptance, or first-principles authority;
- marks no runtime feature `implemented` — per the audit's
  parallel-deliverable rule #4, an `implemented` claim requires code and tests
  in the same diff, and this packet has neither.

The downstream computation (the actual term II/IV/V/VI evaluation) is Sprint 4
work per the audit ("Sprint 4 — full Auluck six-term power-port computation").
This packet only defines the data contract Sprint 3 must satisfy so Sprint 4
can proceed.

## 1. Source-backed findings

### 1.1 The four Sigma_p integrands (verified Auluck extract)

All four terms are taken verbatim from the verified extract
`docs/external_team_submissions/2026_05_18_three_sprint_blocker_packet/sprint_2/AULUCK_2021_POWER_BALANCE_EQUATIONS_VERIFIED.md`,
eq. (6), p.8 (term labels I-VI are Auluck's own), with the `E x B` Sigma_p
decomposition supplied by eq. (5), p.8 (via the generalized Ohm's law eq. (4),
p.8 `E = -(v x B) + eta J`).

| Term | Auluck eq. (6) integrand (ASCII) | Physical name | Source |
| --- | --- | --- | --- |
| II  | `+ integral_Sigma_p dS . v ( (1/2) mu0^-1 B^2 )` | motional magnetic | `AULUCK..._VERIFIED.md` eq. (6) p.8, line 81 |
| IV  | `- integral_Sigma_p dS . v ( (1/2) eps0 E^2 )` | motional electric | `AULUCK..._VERIFIED.md` eq. (6) p.8, line 83 |
| V   | `+ mu0^-1 closed_integral_Sigma_p dS . ( eta J x B )` | resistive | `AULUCK..._VERIFIED.md` eq. (6) p.8, line 84; eq. (5) line 75 |
| VI  | `- mu0^-1 closed_integral_Sigma_p dS . B ( B . v )` | anomalous / poloidal | `AULUCK..._VERIFIED.md` eq. (6) p.8, line 85; eq. (5) line 74 |

Per-term operand inventory (what each integrand contracts, on each `Sigma_p`
face):

- **II** needs `dS` (oriented), `v`, `B`. Integrand contracts `dS` with the
  vector `v * ((1/2) mu0^-1 B.B)`. Vanishes when `dS.v = 0`.
- **IV** needs `dS`, `v`, `E`. Integrand contracts `dS` with the vector
  `v * ((1/2) eps0 E.E)`. Carries the eq. (6) leading minus.
- **V** needs `dS`, `eta`, `J`, `B`. Integrand contracts `dS` with the vector
  `eta * (J x B)`, scaled by `mu0^-1`. Independent of `v`.
- **VI** needs `dS`, `B`, `v`. Integrand contracts `dS` with the vector
  `B * (B.v)`, scaled by `-mu0^-1`. Depends on `B` normal to the surface AND
  the `v` component along `B`.

Sign discipline (verified extract "Sign convention" section): `I(t)V(t)` is the
power **input** crossing the **excluded** electrode/source interface; eq. (1)
carries a load-bearing leading minus. The eq. (6) term signs above are verbatim
and MUST NOT be flipped to make a residual close.

### 1.2 Moving vs stationary boundary distinction (Auluck p.8)

Verified extract, p.8 (lines 31-34, quoted verbatim from the primary PDF):

> "The second integral is evaluated only on the moving boundary Sigma_p of the
> domain Omega since stationary boundaries do not contribute to it. The third
> integral is evaluated over the entire surface Sigma."

Consequences for the runtime interface:

1. `Sigma_p` is the **moving** subset of the closed bounding surface `Sigma` of
   `Omega`. Terms II, IV, V, VI integrate over `Sigma_p` ONLY.
2. **Stationary** boundary faces (electrodes held fixed, chamber wall, fixed
   PML/open boundary, the excluded electrode/source interface) contribute
   **exactly zero** to all four `dS.v`-bearing motional terms. This is not a
   numerical approximation — it is the definition. A face whose material
   velocity `v = 0`, or which the geometry classifies as stationary, must
   contribute identically zero.
3. The electrode/power-source interface is **excluded from `Omega`** entirely
   (verified extract p.6-7 "Domain"); its Poynting flux **is** the LHS
   `I(t)V(t)`. It is therefore neither part of `Sigma_p` nor a separate
   "electrode work" term. Any runtime field named `electrode_*_work` is NOT an
   Auluck eq. (6) quantity and is out of scope here (consistent with
   `power_port.py` lines 43-50 and the verified extract "DOES NOT provide").
4. Because `Sigma_p` is `Omega`'s boundary, the deformation of `Omega` between
   steps is the physical origin of `Sigma_p`: a face is "moving" when the
   `Omega` partition it bounds changes, or when the bounding material carries
   `v != 0`. The runtime must record the classification explicitly; it must not
   be inferred at compute time by `power_port.py`.

### 1.3 What the verified extract does NOT provide

The verified extract explicitly does not provide a discretisation, a face
quadrature order, a time-centering, or a numerical residual tolerance for the
`Sigma_p` integrals (extract "What this source DOES and DOES NOT provide", and
WP-N1B time-centering/tolerance status packets). Those choices are runtime
engineering decisions and must be carried as metadata (centering, quadrature),
not promoted as source-backed. This packet therefore specifies the data
**schema** and leaves the accepted centering/tolerance to the separate WP-N1B
proposals.

## 2. supported / candidate / blocked table

Status of each runtime quantity required to evaluate terms II/IV/V/VI, as the
code stands on `codex/corpus` HEAD. Evidence is by direct read of
`src/dpf/first_principles/power_port.py`, `src/dpf/fields/maxwell_3d.py`,
`src/dpf/fields/hybrid_stepper.py`, `src/dpf/fields/hybrid_simulator.py`,
`src/dpf/fields/hybrid_loop.py`, `src/dpf/fields/source_geometry.py`,
`src/dpf/fields/conductivity.py`, `src/dpf/fluid/constrained_transport.py`.

Legend:
- **supported** — the runtime already computes this quantity and it is in a
  form the power-port ledger can consume (or trivially can).
- **candidate** — the runtime computes the underlying physics but in the wrong
  location/centering, on cell labels not faces, or it is not threaded to
  `_accumulate_power_port_ledger`; usable only after a non-invasive adapter.
- **blocked** — the runtime does not compute or expose this at all; new
  reviewed geometry / new telemetry is required.

| # | Required runtime quantity | Status | Evidence (file:line) |
| --- | --- | --- | --- |
| 1 | `Sigma_p` face set (moving boundary of `Omega`) | **blocked** | No `sigma_p`/face-set in `src/` except the blocker string `power_port.py:395-397`; `source_geometry.py:18-23` labels are per-cell, not faces |
| 2 | Moving vs stationary face classification | **blocked** | No classification anywhere; `source_geometry.py` partition has no per-face moving flag; `power_port.py:556-570` blocks all four terms on this |
| 3 | Face-centered `B` on `Sigma_p` faces | **candidate** | `B` is face-centered on the Yee grid (`maxwell_3d.py:118-121` `StaggeredBField`); `face_to_cell_centered` exists (`constrained_transport.py:150`) but there is no `Sigma_p`-face sampler and no cell->face restriction |
| 4 | Face-centered `E` on `Sigma_p` faces | **candidate** | `E` is Yee **edge**-centered (`maxwell_3d.py:103-115`); `edge_E_to_cell_centered` exists (`maxwell_3d.py:344`) but there is no edge->face sampler for `Sigma_p` faces |
| 5 | Face-centered `J` on `Sigma_p` faces | **candidate** | `total_current_A_m2` is computed cell-centered (`hybrid_stepper.py:116-131`); masked to resolved plasma; no `Sigma_p`-face sampling |
| 6 | Face-centered `v` (plasma/material velocity) on `Sigma_p` faces | **blocked** | `plasma_velocity_m_s` is a caller-supplied cell-centered array (`hybrid_simulator.py:137,327`; `hybrid_loop.py:153,365,401`); it is **never** passed to `_accumulate_power_port_ledger` (`hybrid_simulator.py:350-363`) — the accumulator has no `velocity` argument (`hybrid_simulator.py:801-815`) |
| 7 | Face-centered resistivity `eta` on `Sigma_p` faces | **blocked** | `eta` is computed as `1/sigma` cell-centered inside `hybrid_loop.py:349-354`, used only for the electron-energy closure; it is **never** threaded to the power-port accumulator; `_accumulate_power_port_ledger` has no `eta` argument |
| 8 | Outward-oriented `dS` per `Sigma_p` face (unit normal + sign) | **blocked** | No oriented surface element anywhere; `wall_poynting_flux_W` (`hybrid_stepper.py:343-371`) hardcodes the axial `z` normal and `dx*dy` only — it is a cell-set estimate, not an oriented `dS` |
| 9 | Face area per `Sigma_p` face | **candidate** | Grid spacing `dx,dy,dz` is known (`maxwell_3d.py:64-73`); per-axis face areas are derivable, but no face-area array keyed to `Sigma_p` faces exists |
| 10 | Centering metadata for each emitted field | **candidate** | The runtime records some centering provenance (`hybrid_simulator.py:941-957` `snapshot_provenance`, `_field_work_telemetry` `time_centering`) but nothing face-resolved or `Sigma_p`-scoped |
| 11 | Sign-convention record for the surface terms | **candidate** | `power_port_ledger["sign_convention"]` exists as a string (`hybrid_simulator.py:1045`) and `power_port.py:507` requires it; it is the volume/Omega convention, not a per-surface-term sign record |
| 12 | Deterministic mask/face-set hashes for `Sigma_p` | **blocked** | `source_geometry.py:79-85` hashes per-cell label masks only; there is no `Sigma_p` face-set, hence no `Sigma_p` hash |

Counts: **supported 0**, **candidate 5** (#3, #4, #5, #9, #10, #11),
**blocked 7** (#1, #2, #6, #7, #8, #12).

Interpretation: nothing is fully ready. The five `candidate` rows are physics
the runtime already computes but in the wrong place (cell/edge centering, cell
labels not faces, not threaded to the accumulator). The seven `blocked` rows —
in particular the `Sigma_p` face set, the moving/stationary classification, and
`v`/`eta` on faces — require the WP-N3 reviewed geometry plus new telemetry
plumbing. Until rows #1, #2, #6, #7, #8 are resolved, `power_port.py` is correct
to keep terms II/IV/V/VI fail-closed (`_SIGMA_P_BLOCKER`).

## 3. Runtime fields required — exact schema

The runtime must emit, alongside the existing `power_port_ledger` telemetry
(`hybrid_simulator.py:_finalize_power_port_ledger`), a new sub-packet
`sigma_p_surface_packet`. The schema below is what `power_port.py` must consume.
All arrays are 1-D, indexed by a single face index `f` over the `Sigma_p` face
set, so a face's geometry and every field on it share one index — this avoids
the staggered-grid hazard called out in `CLAUDE.md` ("NEVER assume `B[i]` and
`rho[i]` are at the same spatial location"). The runtime performs the Yee
edge/face -> `Sigma_p`-face sampling; `power_port.py` never re-interpolates.

`sigma_p_surface_packet` (top level):

| Field | Type | Units | Location / centering | Notes |
| --- | --- | --- | --- | --- |
| `status` | str | — | — | Must start `candidate_sigma_p_surface_packet_not_validation` |
| `source_refs` | list[str] | — | — | Must cite the WP-N3 reviewed-geometry packet and `AULUCK..._VERIFIED.md` eq. (5)/(6) |
| `n_sigma_p_faces` | int | — | — | Length of every per-face array below; `0` is legal (fail-closed downstream) |
| `face_count_total_sigma` | int | — | — | Faces on the full closed `Sigma`; `n_sigma_p_faces <= face_count_total_sigma` |
| `geometry_review_status` | str | — | — | Must read `geometry_candidate_not_reviewed` until WP-N3 review lands |
| `sigma_p_face_set_sha256` | str | — | — | Deterministic hash of the `Sigma_p` face index set |
| `moving_classification_sha256` | str | — | — | Hash of the per-face moving/stationary flag array |
| `omega_partition_sha256` | str | — | — | Hash linking `Sigma_p` to the `Omega` partition it bounds |
| `material_mask_sha256_by_class` | dict[str,str] | — | — | Per material/domain class hash (rods, anode, insulator, wall, plasma) per the WP-N3 geometry packet |
| `centering` | dict | — | — | See "centering metadata" sub-schema below |
| `sign_convention` | dict | — | — | See "sign convention" sub-schema below |
| `faces` | dict of 1-D arrays | — | — | The per-face quantities below; each array has length `n_sigma_p_faces` |

`sigma_p_surface_packet["faces"]` (each entry is a length-`n_sigma_p_faces`
array; vector quantities are `(n_sigma_p_faces, 3)`):

| Field | Type | Units | Location / centering | Consumed by term(s) |
| --- | --- | --- | --- | --- |
| `face_index` | int array | — | face id | bookkeeping |
| `is_moving` | bool array | — | per face | gates II/IV/V/VI; `False` -> face contributes 0 |
| `material_class` | str array | — | per face | audit / under-resolution gate |
| `outward_normal` | float `(N,3)` | dimensionless unit | face-centered | dS direction, all terms |
| `face_area_m2` | float array | m^2 | face-centered | dS magnitude, all terms |
| `dS_outward_m2` | float `(N,3)` | m^2 | face-centered | `= outward_normal * face_area_m2`; all terms |
| `B_T` | float `(N,3)` | T | face-centered on `Sigma_p` | II (`B.B`), V (`J x B`), VI (`B`, `B.v`) |
| `E_V_m` | float `(N,3)` | V/m | face-centered on `Sigma_p` | IV (`E.E`) |
| `J_A_m2` | float `(N,3)` | A/m^2 | face-centered on `Sigma_p` | V (`J x B`) |
| `v_m_s` | float `(N,3)` | m/s | face-centered on `Sigma_p` | II, IV, VI (`dS.v`, `B.v`) |
| `eta_ohm_m` | float array | Ohm.m | face-centered on `Sigma_p` | V (`eta J x B`) |
| `field_centering_source` | str array | — | per face | how the face value was sampled (edge-avg, face-native, cell-avg) |

Per-term derived integrands the runtime MAY also emit for cross-checking (the
runtime must still emit the raw operands above; these are convenience, not a
substitute):

| Field | Units | Definition |
| --- | --- | --- |
| `term_ii_integrand_W_m2` | W/m^2 | `dot(v, outward_normal) * 0.5 * MU_0**-1 * sum(B*B)` |
| `term_iv_integrand_W_m2` | W/m^2 | `- dot(v, outward_normal) * 0.5 * EPSILON_0 * sum(E*E)` |
| `term_v_integrand_W_m2` | W/m^2 | `MU_0**-1 * dot(outward_normal, eta * cross(J, B))` |
| `term_vi_integrand_W_m2` | W/m^2 | `- MU_0**-1 * dot(outward_normal, B) * dot(B, v)` |

`centering` sub-schema (records the engineering numerical choice; not
source-promoted):

| Field | Type | Notes |
| --- | --- | --- |
| `b_sampling` | str | e.g. `yee_face_native` or `cell_avg_to_sigma_p_face` |
| `e_sampling` | str | e.g. `yee_edge_avg_to_sigma_p_face` |
| `j_sampling` | str | e.g. `cell_centered_avg_to_sigma_p_face` |
| `v_sampling` | str | e.g. `cell_centered_avg_to_sigma_p_face` |
| `eta_sampling` | str | e.g. `cell_centered_avg_to_sigma_p_face` |
| `time_centering` | str | must read `candidate_step_consistent_not_accepted` until WP-N1B centering is accepted |
| `quadrature` | str | e.g. `midpoint_one_point_per_face`; engineering, not source-backed |

`sign_convention` sub-schema:

| Field | Type | Notes |
| --- | --- | --- |
| `dS_orientation` | str | must be `outward_from_omega` |
| `eq6_term_signs` | dict | `{term_ii:"+", term_iv:"-", term_v:"+", term_vi:"-"}` verbatim from eq. (6) |
| `auluck_eq6_ref` | str | `AULUCK_2021_POWER_BALANCE_EQUATIONS_VERIFIED.md eq. (6) p.8` |
| `auluck_eq5_ref` | str | `AULUCK_2021_POWER_BALANCE_EQUATIONS_VERIFIED.md eq. (5) p.8` |

Constants: `MU_0 = 1.25663706212e-6` H/m, `EPSILON_0 = 8.8541878128e-12` F/m,
both already defined in `src/dpf/fields/maxwell_3d.py:26-27`. `power_port.py`
must import these, not redefine them.

## 4. Missing parameters — what the runtime does NOT expose today

Hard gaps on `codex/corpus` HEAD (each is a `blocked` row in section 2):

1. **No `Sigma_p` face set.** `source_geometry.build_auluck_omega_domain`
   (`source_geometry.py:126-251`) produces four **per-cell** boolean labels
   (`omega_volume_cells`, `terminal_source_interface_faces`,
   `wall_material_faces`, `open_pml_faces`). Despite the `_faces` suffix these
   are cell masks, not a face set. There is no enumeration of the faces of
   `Omega`, no closed surface `Sigma`, and no moving subset `Sigma_p`.
2. **No moving/stationary face classification.** Nothing distinguishes a moving
   plasma-surface face from a stationary electrode/wall/PML face. Auluck p.8
   requires this so stationary faces drop out of II/IV/V/VI.
3. **No oriented `dS`.** No outward unit normal and no signed surface element
   exists. `wall_poynting_flux_W` (`hybrid_stepper.py:343-371`) hardcodes the
   axial `z` normal and uses `dx*dy` as the area for every labeled cell — it is
   an explicit "engineering surface estimate" over a cell set, not a true
   `dS`-weighted face integral, and cannot supply terms II/IV/V/VI.
4. **No face-centered field sampler.** `B` is Yee face-centered, `E` is Yee
   edge-centered, `J`/`v`/`eta` are cell-centered. The runtime has
   `face_to_cell_centered` and `edge_E_to_cell_centered` (cell-ward only); it
   has **no** sampler that restricts any field onto a `Sigma_p` face set with
   recorded centering. `constrained_transport.py` has no `cell_to_face`.
5. **`v` is not threaded to the power-port ledger.** `plasma_velocity_m_s`
   reaches `hybrid_loop` (`:365,:401`) but `_accumulate_power_port_ledger`
   (`hybrid_simulator.py:801-815`) takes no velocity argument and the call site
   (`:350-363`) passes none. Terms II/IV/VI are therefore unreachable today.
6. **`eta` is not threaded to the power-port ledger.** `eta = 1/sigma` is built
   at `hybrid_loop.py:349-354` for the electron-energy closure only. The
   power-port accumulator never receives it. Term V is unreachable today.
   (`conductivity.partial_ionized_conductivity` does emit
   `max_resistivity_ohm_m` as a scalar telemetry value — a single max, not a
   per-face field — so even that is not a usable `Sigma_p` `eta`.)
7. **No `Sigma_p` hashes.** `source_geometry._mask_sha256` hashes per-cell
   masks; with no face set there is no `sigma_p_face_set_sha256` and no
   `moving_classification_sha256`.
8. **No accepted face time-centering or quadrature.** The verified extract
   gives no discretisation; WP-N1B time-centering is still a proposal. The
   schema carries these as metadata strings, explicitly not source-backed.

Dependency: rows 1, 2, 3, 7 are blocked on the **WP-N3 reviewed PF-1000/Akel
geometry packet** (`WP_N3_GEOMETRY_SOURCE_PACKET.md`, the sibling lane) — the
`Sigma_p` face set and its moving classification can only be derived from
reviewed material masks. Rows 5, 6 are pure plumbing (thread existing
cell-centered `v` and `eta` into `_accumulate_power_port_ledger`) and do not
need new physics. Row 8 is a metadata decision.

## 5. Proposed tests and fail-closed negative controls

These are test designs only; per audit rule #4 no test is `implemented` here.
Sprint 3 (geometry/`Sigma_p` plumbing) and Sprint 4 (term computation) must
land the tests with the code.

### 5.1 Positive / structural tests

- **T1 `Sigma_p` is a strict subset of `Sigma`.** In a controlled synthetic
  geometry, `n_sigma_p_faces <= face_count_total_sigma` and every `Sigma_p`
  face has `is_moving = True`.
- **T2 moving faces non-empty in a controlled moving case.** A synthetic case
  with a deliberately deforming `Omega` yields `n_sigma_p_faces > 0`.
- **T3 oriented `dS` consistency.** `dS_outward_m2 == outward_normal *
  face_area_m2` elementwise; every `outward_normal` is unit length to tolerance.
- **T4 face-set / classification hashes are deterministic.** Re-running the same
  synthetic geometry reproduces `sigma_p_face_set_sha256` and
  `moving_classification_sha256` bit-for-bit, and both appear in the manifest.
- **T5 schema completeness.** Every `faces` array has length `n_sigma_p_faces`;
  every field in section 3 is present and correctly typed/uniteed.

### 5.2 Fail-closed negative controls

Each control asserts that a specific missing input forces the corresponding
term(s) to `value_J = None`, `status = "blocked"` — never a fabricated or
closure-derived number.

- **N1 missing `Sigma_p` blocks II, IV, V, VI.** If `sigma_p_surface_packet`
  is absent or `n_sigma_p_faces == 0`, all four terms fail closed with
  `_SIGMA_P_BLOCKER`. (Matches current `power_port.py:556-570` behaviour;
  the test pins it so it cannot regress.)
- **N2 missing `v` blocks II, IV, VI.** If `faces.v_m_s` is absent, terms II,
  IV, VI fail closed; term V (which does not use `v`) is unaffected by the
  absence of `v` alone. Blocker reason must name the missing `v`.
- **N3 missing `eta` blocks V only.** If `faces.eta_ohm_m` is absent, term V
  fails closed; II/IV/VI are unaffected by the absence of `eta` alone.
- **N4 stationary boundary contributes exactly zero to motional terms.** A
  synthetic geometry whose `Sigma_p` faces all carry `is_moving = False` (or
  `v = 0`) must yield term II = term IV = term VI = exactly `0.0` (not a small
  residual). This is the direct Auluck p.8 invariant.
- **N5 stationary-face leakage guard.** If any face with `is_moving = False`
  carries `v != 0`, the runtime must drop it from the motional sums (the
  classification wins) — assert the contribution is zero regardless of `v`.
- **N6 closure substitution rejected.** Any attempt to set a term's `value_J`
  to `iv_work_J` minus the other five terms must fail the test:
  `derived_by_closure` must be `False` and `computed_independently` must be
  `True` for a term to count.
- **N7 missing sign-convention record blocks the residual.** If
  `sign_convention.eq6_term_signs` is absent or differs from the verbatim
  eq. (6) signs, the six-term residual stays `None`
  (consistent with `power_port.py:507,605-616`).
- **N8 missing centering metadata blocks acceptance.** Absent
  `centering` -> the packet cannot raise `can_support_power_port_acceptance`
  above `False`.

## 6. Exact implementation recommendations

How `power_port.py` should consume `sigma_p_surface_packet` once Sprint 3
delivers it. This describes the intended Sprint 4 implementation; it is NOT
implemented in this packet.

1. **New consumer, no closure.** Replace the four unconditional
   `_term_blocked_packet(_SIGMA_P_BLOCKER, ...)` calls at
   `power_port.py:567-570` with a helper `_sigma_p_surface_term(...)` that:
   - reads `ledger["power_port_ledger"]["sigma_p_surface_packet"]` (or the
     equivalent telemetry path);
   - if the packet is missing, `n_sigma_p_faces == 0`, or the required operand
     for that term is absent, returns `_term_blocked_packet` with a blocker
     string that names the missing operand (preserving fail-closed behaviour);
   - otherwise sums the per-face integrand over `Sigma_p` and returns
     `_term_computed_packet(value_J, ..., derived_by_closure=False)`.
2. **Per-term face sum (the only arithmetic `power_port.py` does).** Using the
   `faces` arrays and `dS_outward_m2 = outward_normal * face_area_m2`:
   - term II `= sum_f dot(dS_outward[f], v[f]) * 0.5 * MU_0**-1 * dot(B[f],B[f])`
   - term IV `= - sum_f dot(dS_outward[f], v[f]) * 0.5 * EPSILON_0 * dot(E[f],E[f])`
   - term V  `= MU_0**-1 * sum_f dot(dS_outward[f], eta[f] * cross(J[f],B[f]))`
   - term VI `= - MU_0**-1 * sum_f dot(dS_outward[f], B[f]) * dot(B[f],v[f])`
   `power_port.py` performs **no** interpolation and **no** re-centering — every
   field already lives on the `Sigma_p` face index. It applies only the
   contraction above and the verbatim eq. (6) sign.
3. **Stationary faces gated upstream, re-checked downstream.** The runtime
   should already exclude `is_moving == False` faces from the `Sigma_p` set;
   `power_port.py` must additionally assert no `is_moving == False` face is
   present (defence in depth for the Auluck p.8 invariant; negative control
   N5).
4. **Sign and centering propagated, not assumed.** `power_port.py` must read
   `sign_convention.eq6_term_signs` and assert it matches the verbatim eq. (6)
   signs before computing the residual; mismatch -> residual stays `None`
   (`power_port.py` already does this for the eq. (1) convention at `:507`).
   It must copy `centering` into each term packet as provenance.
5. **Hashes into the manifest.** `sigma_p_face_set_sha256`,
   `moving_classification_sha256`, `omega_partition_sha256`, and
   `material_mask_sha256_by_class` must be carried verbatim into the emitted
   power-port packet so an external reviewer can confirm the geometry that
   produced terms II/IV/V/VI.
6. **Residual unchanged in spirit.** The six-term residual
   `I*V - (I+II+III+IV+V+VI)` (`power_port.py:611-614`) becomes a genuine
   diagnostic only once **all six** terms are `computed_independently`. With any
   of II/IV/V/VI still blocked, `residual_J` stays `None` — the existing
   `all_terms_independent` gate (`power_port.py:600-616`) already enforces this
   and must not be weakened.
7. **No new `power_port.py` constants.** Import `MU_0`, `EPSILON_0` from
   `dpf.fields.maxwell_3d`; do not redefine.

## 7. Do-not-promote notes

- This packet is `source_backed_interface_spec_not_validation_not_acceptance`.
  It promotes nothing. `can_support_first_principles_acceptance` and
  `can_support_power_port_acceptance` remain `False` everywhere.
- No runtime feature here is `implemented`. Per the audit's
  parallel-deliverable rule #4, an `implemented` claim requires code and tests
  in the same diff; this packet ships neither.
- Until Sprint 3 delivers a **reviewed** `Sigma_p` face set with the section-3
  schema, `power_port.py` MUST keep terms II/IV/V/VI fail-closed under
  `_SIGMA_P_BLOCKER`. A blocked term is the honest outcome; a fabricated or
  closure-derived value is a physics-integrity violation
  (`CLAUDE.md` "Physics Integrity Rules").
- No Auluck eq. (6) term may be computed as `I*V` minus the others. The
  six-term residual is a genuine, non-trivial diagnostic only when all six
  terms are independently computed (Auluck eqs. (13)-(14) anomalous-impedance
  caveat; verified extract lines 125-128).
- The `Sigma_p` geometry is `geometry_candidate_not_reviewed` until the sibling
  WP-N3 geometry lane (`WP_N3_GEOMETRY_SOURCE_PACKET.md`) lands reviewed
  PF-1000/Akel dimensions; no `Sigma_p`-derived number is acceptance-grade
  before then.
- No same-scope PF-1000/Akel waveform, field, or yield validation is claimed or
  implied by this spec.
- This packet does not edit any Sprint 2.2-owned file
  (`segmented_whole_shot_combine.py`, `power_port.py`, `power_port.py`
  docstrings, `DPF_REQUIREMENTS_BASELINE.md`, `SRS_TRACEABILITY_MATRIX.*`, or
  the packet traceability/changelog/test-map files); the section-6
  recommendations are proposals for a future Sprint 4 diff, not changes made
  here.

## 8. Source references

- `docs/external_team_submissions/2026_05_18_three_sprint_blocker_packet/sprint_2/AULUCK_2021_POWER_BALANCE_EQUATIONS_VERIFIED.md`
  — eqs. (4), (5), (6) p.8; moving/stationary boundary prose p.8; sign
  convention.
- `docs/FIRST_PRINCIPLES_CODEX_AUDIT_SPRINT2_FOLLOWUP_2026_05_19.md`
  — "Allowed parallel lanes" #2; Sprint 3 / Sprint 4 objectives.
- `src/dpf/first_principles/power_port.py` — `build_wp_n1_auluck_power_port_ledger`,
  `_SIGMA_P_BLOCKER`, `_term_blocked_packet`/`_term_computed_packet`,
  six-term residual gate.
- `src/dpf/fields/maxwell_3d.py` — Yee layout: `B` face-centered, `E`
  edge-centered; `MU_0`, `EPSILON_0`.
- `src/dpf/fields/hybrid_stepper.py` — `wall_poynting_flux_W`,
  `omega_stored_em_energy_split_J`, cell-centered `J`.
- `src/dpf/fields/hybrid_simulator.py` — `_accumulate_power_port_ledger`,
  `_finalize_power_port_ledger`; no `v`/`eta` argument to the accumulator.
- `src/dpf/fields/hybrid_loop.py` — `plasma_velocity_m_s` use; `eta = 1/sigma`
  at `:349-354`.
- `src/dpf/fields/source_geometry.py` — `build_auluck_omega_domain` per-cell
  partition; `_mask_sha256`.
- `src/dpf/fields/conductivity.py` — `partial_ionized_conductivity` scalar
  `max_resistivity_ohm_m` telemetry.
- `src/dpf/fluid/constrained_transport.py` — `StaggeredBField`,
  `face_to_cell_centered` (cell-ward only; no `cell_to_face`).
