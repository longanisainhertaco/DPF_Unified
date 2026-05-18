# WP-N1 Power-Port SOURCE PACKET

Date: 2026-05-18
Repo: `/Users/anthonyzamora/dpf-unified` — branch `codex/corpus` — HEAD `466a0a5`
Author role: WP-N1 PLAN agent (plan-first; no code implemented).
Status: **SOURCE PACKET — NOT ACCEPTED PHYSICS.** The lead verifies this against
`KnowledgeReference/` before any implementation. The implementation agent (WP-N1
CODE) follows this packet exactly; deviations require a new source-verified packet.

Responds to: `docs/FIRST_PRINCIPLES_CODEX_AGENT_AUDIT_AND_NEXT_INSTRUCTIONS_2026_05_18.md`
finding **A-5: Power Port Is The First Physics Blocker**.

---

## 0. Provenance — sources opened and verified this session

Every citation below was produced by opening the named file at the named lines in
this session and confirming the text supports the claim. No physics from training
data. Tag form: `[KR: file:lines]`.

| Source | File | Verified line ranges (this session) |
|---|---|---|
| Auluck 2021 — DPF circuit-element relation | `KnowledgeReference/auluck-2021-dpf-circuit-element.md` | 130–289, 289–448, 458–697, 698–826, 1010–1047 |
| NRL Plasma Formulary 2019 — Poynting theorem | `KnowledgeReference/2019nrlplasma-formulary-037290d4.md` | 1855–1909 |
| Hybrid PIC/fluid DPF — external circuit | `KnowledgeReference/fully-electromagnetic-hybrid-pic-fluid-dpf-neutron-yield-acb71fa9.md` | 735–809 |

**Caveat on Auluck equation numbers.** The Auluck markdown is an OCR extract.
Equations 3, 5, 6 (lines ~306–664) are rendered as scattered, out-of-order math
tokens and are NOT machine-legible as equations. Only **equation 1** (lines
173–195, the circuit-element relation) and the **labeled-term identification
narrative** (lines 762–786) are unambiguously legible. Where this packet relies
on Auluck eq 3/5/6 structure, it cites the surrounding *prose* (which is legible),
not the garbled equation glyphs. Any term that depends only on the garbled glyphs
is marked **blocked — OCR-illegible source**.

---

## 1. The named Auluck `Omega` integration domain

### 1.1 What the source says (verified)

Auluck eq 1 — the DPF circuit-element relation [KR: auluck-2021-dpf-circuit-element.md:173-197]:

```
V_12(t) = ( 1 / I(t) ) * integral_Omega ( J . E ) d^3 r
```

> "where the right hand side is the total electric power flowing through the two
> terminals expressed in terms of fields inside the device divided by the total
> current flowing through the device."
> [KR: auluck-2021-dpf-circuit-element.md:196-197]

The integration domain `Omega` and its exclusion
[KR: auluck-2021-dpf-circuit-element.md:203-209]:

> "This 3-D spatial integration is over a domain Omega such that J is zero outside
> it. Excluded from this domain is the interface between the 'circuit element' and
> the external power source, through which, the power enters the device. In the
> case of the plasma focus, this would be the cathode plate that is in contact with
> the insulator and the squirrel cage, or its smaller portion in the initial phase."

The bounding surface `Sigma` of `Omega`
[KR: auluck-2021-dpf-circuit-element.md:215-224]:

> "This bounding surface, designated Sigma, comprises the contact surfaces between
> the plasma and each of the two electrodes, through which the current passes in,
> as well as the extreme boundary of the plasma bridging the two electrodes that
> separates the current carrying region of the plasma from the current free region.
> The domain Omega is not a simply-connected domain for a plasma focus before
> breakup of the plasma column. Rather, topologically it is a toroid."

Fig. 1 description — the source interface is a distinct, named surface
[KR: auluck-2021-dpf-circuit-element.md:225-257]:

> "Domain Omega is the volume enclosed by the current-carrying surfaces of the
> electrodes and the plasma ... and the interface with the power source depicted by
> the black dashed lines. ... The Poynting vector ... Its surface integration over
> the black-dashed interface with the power source can be easily shown to be equal
> to I(t) V(t)."

### 1.2 Runtime definition of `Omega` (build as a mask)

Coordinate frame: the candidate runtime is the Cartesian Yee grid
`Maxwell3DGrid` `shape=(nx, ny, nz)`, face-centered `B`, edge-centered `E`
(`src/dpf/fields/maxwell_3d.py:37-160`). It wraps the axisymmetric source
domain `HybridPICSourceGeometry` (`src/dpf/fields/source_geometry.py:11-37`):
anode radius `0.01 m`, cathode length `0.10 m`, physical radius `0.015 m`,
physical length `0.10 m`, cell size `2.0e-4 m`. The axial coordinate is `z`
(index `k`); the injection port is a `z = k_port` slice with a radius mask
(`src/dpf/fields/circuit_boundary.py:178-246`).

The implementation agent builds **four disjoint, exhaustive cell/face label sets**
covering the whole grid. Express each as a boolean array; emit each array's
SHA-256, cell/face count, and axis-aligned bounding box.

| Label | Definition | Runtime construction |
|---|---|---|
| `omega_volume_cells` | Cell-centered cells inside the current-carrying domain `Omega`: the toroidal volume enclosed by electrode contact surfaces and the plasma current-free boundary, with the source interface removed. **The integration volume for the J.E ledger term.** | Cells with resolved current-carrying plasma (electron density above the numerical floor AND current density above a current-floor), **minus** any cell touching the `terminal_source_interface_faces` slab. Per Auluck, J is zero outside Omega [KR: auluck:203-204], so the current mask defines the domain interior; the source-interface removal is mandatory [KR: auluck:205-209]. |
| `terminal_source_interface_faces` | The faces forming the cathode-plate / insulator / squirrel-cage interface with the external power source — **EXCLUDED from `Omega`** per Auluck. The injection-port `z = k_port` slice (or its initial-phase smaller annulus). | The `apply_injection_port_boundary` port slice: faces at axial index `k_port` inside the port radius mask (`circuit_boundary.py:178-246`). This is the "black dashed" interface of Auluck Fig. 1 [KR: auluck:225-257]. |
| `wall_material_faces` | Faces on solid/material boundaries that are NOT the declared source interface: coaxial anode and cathode current-carrying contact surfaces, hollow-anode bore wall, alumina insulator surface, chamber wall, electrode end faces. | Faces of `omega_volume_cells` that abut a material region (anode/cathode/insulator/chamber masks per A-7) and are not in `terminal_source_interface_faces`. After rundown, the coaxial-electrode Poynting integral is zero because the Poynting vector is axial there [KR: auluck:432-443] — that property is a per-phase check, not a license to drop the faces from the label set. |
| `open_pml_faces` | Outer non-material open / PML boundary faces (axial PML, `axial_pml_layers = 20`; `source_geometry.py:27`). Carry no electrode work; carry a (small, expected ~0) Poynting flux that the ledger must still account. | Grid-boundary faces in the PML layers, not material, not the source interface. |

Constraints the implementation agent must enforce, and the linter must check:
- The four label sets are **mutually disjoint** and their union covers every
  cell/face of the grid (exhaustive partition). No cell/face is unlabeled.
- `terminal_source_interface_faces` is **non-empty** and **disjoint from**
  `omega_volume_cells` boundary faces. A run with the source interface inside
  `Omega` violates Auluck eq 1's domain and must fail (negative test N2).
- `Omega` is the *current-carrying* volume: cells where `J = 0` are outside
  `Omega` by definition [KR: auluck:203-204]. The existing density-threshold
  mask in `_field_work_telemetry` (`src/dpf/fields/hybrid_stepper.py:296-313`,
  domain label `resolved_plasma_current_carrying_cells`) is a **candidate
  approximation only** — it is a density threshold, not the Auluck
  electrode-bounded toroid, and it does not remove the source interface. WP-N1
  replaces it with `omega_volume_cells` as defined here.
- Emit, for `Omega` and for each face label: `mask_sha256`, `cell_count` /
  `face_count`, `bounds` (index-space AABB), `source_refs`
  (`[KR: auluck-2021-dpf-circuit-element.md:203-257]`). Required by audit A-5
  ("Emit domain mask hash, cell count, bounds, and source refs").

---

## 2. The five-term energy ledger

Physical basis: Poynting's theorem [KR: 2019nrlplasma-formulary-037290d4.md:1880-1888]:

```
dW/dt + integral_S ( N . dS ) = - integral_V ( J . E ) dV          (SI)
W = (1/2) integral_V ( H.B + E.D ) dV ;   N = E x H
```

with `S` the closed surface bounding `V`. Applied to `V = Omega`, the closed
bounding surface `Sigma = terminal_source_interface_faces UNION wall_material_faces
UNION open_pml_faces`. Auluck's eq 1 is the same statement with the source-interface
Poynting flux identified as `I*V` [KR: auluck:239-257, 443-448]:

> "Along the cathode plate at the bottom, which is excluded from the domain for
> being the interface between the circuit element and the power source, the
> surface integral is exactly equal to the power input I(t) V(t)."

The ledger reports **five cumulative-energy terms** (time-integrated power, joules).
All five must close to a residual (Section 4).

| # | Term key | Source equation (ASCII) | Units | Sign (positive means) | KR citation | Verified supports claim? |
|---|---|---|---|---|---|---|
| 1 | `terminal_port_work_J` | `integral_t [ integral_{Sigma_port} (E x H).dS ] dt` ; equivalently `integral_t I(t) U_DPF(t) dt` over `terminal_source_interface_faces` | J | Positive = net electromagnetic energy flowing **into Omega across the source interface** (generator delivering power to the DPF). | [KR: auluck:239-257] (Poynting flux over the black-dashed source interface equals I*V); [KR: 2019nrl...:1880-1888] (Poynting theorem) | **TRUE** — Auluck states the source-interface Poynting integral "is exactly equal to the power input I(t) V(t)" [KR: auluck:443-448]. |
| 2 | `volume_j_dot_e_work_J` | `integral_t [ integral_Omega (J . E) d^3 r ] dt` over `omega_volume_cells` | J | Positive = net work **done by the field on the charges** inside `Omega` (energy leaving the EM field into kinetic + thermal + collisional channels). Equals `integral_t I(t) U_DPF(t) dt` by Auluck eq 1 in the no-storage, no-wall-flux limit. | [KR: auluck:173-197] (eq 1: V = (1/I) integral_Omega J.E d^3r); [KR: 2019nrl...:1886] (RHS of Poynting theorem is `- integral_V J.E dV`) | **TRUE** — eq 1 is the volume J.E relation verbatim; the formulary fixes the sign of the J.E term in the energy balance. |
| 3 | `wall_poynting_flux_excluding_declared_port_J` | `integral_t [ integral_{Sigma_wall} (E x H).dS ] dt` over `wall_material_faces UNION open_pml_faces` (i.e. `Sigma` minus the declared source-interface port) | J | Positive = net EM energy flowing **out of Omega** through walls / open boundary (Poynting outflow convention, `+integral_S N.dS` on the LHS of Poynting's theorem). After rundown this is ~0 on the coaxial electrodes because `N` is axial there. | [KR: 2019nrl...:1882-1888] (`integral_S N.dS` term, S closed); [KR: auluck:215-224] (Sigma comprises electrode contact surfaces + plasma current-free boundary); [KR: auluck:426-443] (coaxial-electrode Poynting integral is zero post-rundown; stationary boundaries do not contribute) | **TRUE** — the formulary's surface term is over the *entire* closed `S`; isolating the non-port part is exactly `Sigma minus port`. Auluck confirms which faces are walls and that coaxial faces contribute ~0 post-rundown. |
| 4 | `electrode_interface_work_J` | Moving-boundary / motional power on `Sigma_p`, the **moving** part of the boundary. From the legible Auluck prose: the part of the surface integral on the moving boundary `Sigma_p` carries a `v x (B.B)` motional contribution; the labeled term `II` (proportional to `dL/dt`, "motional impedance") is one such contribution. | J | Positive = net work associated with the **moving material/plasma interface** transferred at electrode/sheath surfaces (motional-EMF / `dL/dt`-type work that is part of the field-power balance but not pure stored-energy change). | [KR: auluck:426-429] ("The second integral is evaluated only on the moving boundary Sigma_p ... since stationary boundaries do not contribute"); [KR: auluck:472-474] (the part of the integral on Sigma_p written via `v B B` / `J B` motional terms — **OCR-garbled glyphs, prose only**); [KR: auluck:780-783] ("Terms II and IV ... dependent on the velocity with which the dimensions of the inductance and capacitance are changing") | **PARTIAL** — the *existence and role* of a moving-boundary electrode-interface term is verified from legible prose [KR: auluck:426-429, 780-783]. The *exact closed-form integrand* (Auluck eq 5/6) is **OCR-illegible** in this KR extract. See Section 6 gap G1: the implementation must either (a) cite a legible rendering of Auluck eq 5–6, or (b) compute this term as the *residual-free closure* `terminal_port_work - volume_j_dot_e_work - wall_poynting - stored_em_delta` and label it `electrode_interface_work_J__closure_estimate_not_independent` until an independent source-backed integrand exists. |
| 5 | `stored_em_energy_delta_J` | `Delta W = W(t_final) - W(t_initial)` , `W = (1/2) integral_Omega (H.B + E.D) d^3 r` evaluated on `omega_volume_cells` | J | Positive = net **increase** in EM energy stored inside `Omega` over the interval (`dW/dt` term of Poynting's theorem, time-integrated). | [KR: 2019nrl...:1869-1879] (`W = (1/2) integral_V (H.B + E.D) dV`); [KR: 2019nrl...:1880-1882] (`dW/dt` is the storage term of Poynting's theorem); [KR: auluck:762-763] ("terms I and III ... are time derivatives of the total magnetic and electric energy") | **TRUE** — the formulary gives `W` explicitly and identifies `dW/dt` as the storage term; Auluck independently identifies the magnetic + electric energy time-derivatives as terms I and III. |

Runtime hooks that already exist (do not re-derive):
- `magnetic_energy_J`, `electric_energy_J` per step:
  `Maxwell3DDiagnostics` (`src/dpf/fields/maxwell_3d.py:152-163`) — sum these
  over `omega_volume_cells` for term 5.
- Volume `J.E` power: `_field_work_telemetry` (`hybrid_stepper.py:289-345`) —
  re-scope its domain from `resolved_plasma_current_carrying_cells` to
  `omega_volume_cells` for term 2.
- `I(t)`, `U_DPF(t)` per step: circuit step records consumed by
  `build_engineering_power_port_packet` (`src/dpf/first_principles/power_port.py`).
- Cumulative integrals: `cumulative_j_dot_e_work_J`,
  `cumulative_active_port_work_J` already tracked in
  `src/dpf/fields/hybrid_simulator.py:177-445`.
- The Stage-0 ledger `_candidate_stage0_energy_ledger`
  (`power_port.py:597-649`) already names all five term keys with the right
  ASCII basis; WP-N1 fills the two currently-`None` terms
  (`wall_poynting_flux_excluding_declared_port_J`,
  `electrode_interface_work_J`) and re-scopes the others to `Omega`.

The Poynting flux on faces (terms 1 and 3) requires `E x H` evaluated at face
centers. The Yee state has edge-centered `E` and face-centered `B`; the
implementation must interpolate to a common face location (a `face_to_cell_centered`
helper is already imported in `maxwell_3d.py:23`). Flag any interpolation as a
candidate numerical step, not accepted.

---

## 3. Sign convention and time-centering

### 3.1 Sign convention

Single declared convention for the whole ledger, derived from Poynting's theorem
as written in the formulary [KR: 2019nrlplasma-formulary-037290d4.md:1880-1888]:

```
dW/dt  +  integral_S (N.dS)  =  - integral_V (J.E) dV
```

- **Surface fluxes (terms 1, 3): outflow-positive.** `+integral_S N.dS` on the LHS
  with `N = E x H` and `dS` the **outward** normal. A positive surface term is EM
  energy *leaving* `Omega`. Therefore `terminal_port_work_J` is reported with the
  sign such that **positive = energy entering Omega from the generator**, i.e. the
  ledger stores `-(outward source-interface flux)`. This matches Auluck's "power
  input I(t) V(t)" being delivered *into* the device [KR: auluck:443-448] and the
  existing circuit-record convention
  `positive_I_udpf_is_power_drawn_from_generator_by_DPF`
  (`hybrid_simulator.py:275`). `wall_poynting_flux_excluding_declared_port_J` keeps
  the raw outflow-positive sign (positive = energy lost through walls/open boundary).
- **Volume J.E (term 2): field-on-charges positive.** `integral_Omega J.E dV > 0`
  means the field does net positive work on the charges (EM energy converts to
  particle/thermal energy). This is the existing runtime convention
  `positive_J_dot_E_is_field_work_on_charges` (`hybrid_stepper.py:340-341`). Auluck
  eq 1 uses this `J.E` directly as the numerator of `I*V`
  [KR: auluck:173-197]. **Negative local `J.E` is retained, never clipped** — it is
  physical (dynamo / field-building regions) [KR: auluck:158-160 dynamo discussion];
  the existing Stage-0 sign policy already states this
  (`power_port.py:551-553`) and N1 must keep it.
- **Stored-energy delta (term 5): increase positive.** `Delta W > 0` means stored
  EM energy in `Omega` grew. This is the `dW/dt` term with its natural sign.
- **Electrode-interface work (term 4): into-Omega positive**, consistent with
  terms 1/5, so the closure `residual = sum of five terms` is a single signed sum.

With these signs the closed-form balance the ledger must satisfy is:
`terminal_port_work + volume_j_dot_e_work_sign_resolved` — see Section 4 for the
exact residual definition; the implementation agent must not invent a different
sign grouping.

### 3.2 Time-centering

- The circuit / terminal quantities are currently `begin-step`:
  `begin_step_current_times_begin_step_udpf_candidate`
  (`hybrid_simulator.py:278`).
- The field-work `J.E` is currently `begin_step_E_with_midpoint_candidate_current`
  (`hybrid_stepper.py:343`).
- These two **disagree** (begin-step vs mixed begin/midpoint). Auluck eq 1 and the
  Poynting theorem are *instantaneous* identities [KR: auluck:173-197;
  2019nrl...:1880-1888] — they hold at a single time `t`. To time-integrate
  consistently, the WP-N1 ledger must evaluate **all five power terms at the same
  step-centering** before multiplying by `dt`.
- **Declared target centering: step-consistent.** All five terms are evaluated
  from the *completed-step* state (begin and end snapshots of the same step) and
  combined with a single quadrature rule (trapezoidal over `[t_n, t_{n+1}]`), so
  the cumulative ledger is a consistent Riemann/trapezoid sum. The implementation
  emits which snapshot each term used; a run where the five terms use mismatched
  centering must fail negative test N5.
- Time-centering is **not accepted** until the source-backed review closes; the
  packet field stays `time_centering: candidate_step_consistent_not_accepted`.
  This is a *consistency* requirement (all terms aligned), not yet an *accuracy*
  claim — no KR source in scope prescribes a specific high-order centering, so
  claiming an accepted scheme would be **unsourced**; see gap G2.

---

## 4. Residual policy

```
residual_J  =  ( terminal_port_work_J )
             - ( volume_j_dot_e_work_J )
             - ( wall_poynting_flux_excluding_declared_port_J )
             - ( stored_em_energy_delta_J )
             - ( electrode_interface_work_J )
```

Rationale, term by term, from Poynting's theorem applied to `Omega`
[KR: 2019nrlplasma-formulary-037290d4.md:1880-1888] and Auluck's source-interface
identification [KR: auluck:239-257, 443-448]: energy entering across the source
interface (`terminal_port_work`) must equal energy converted to charges in the
volume (`volume_j_dot_e_work`), plus energy leaving through walls/open boundary
(`wall_poynting_flux`), plus the increase in stored EM energy
(`stored_em_energy_delta`), plus motional/electrode-interface work
(`electrode_interface_work`). A perfectly closed discrete scheme gives
`residual_J = 0`.

**Policy (mandatory, audit A-5: "Keep acceptance blocked until residual policy is
reviewed and source-backed"):**

1. `residual_J` is computed and emitted every run, plus a dimensionless
   `residual_fraction = residual_J / denominator` where `denominator =
   max(|terminal_port_work_J|, |volume_j_dot_e_work_J|, 1 J)` — the same
   `_residual_denominator` pattern already in `power_port.py:995-999`.
2. **NO numerical acceptance threshold is defined in this packet.** No KR source
   in scope (`KnowledgeReference/`) prescribes a tolerance for a DPF field/circuit
   power-port closure. Inventing a percentage would violate the
   papers-are-truth rule. The packet field stays:
   `accepted_residual_tolerance: "not_attached"` (already the value in
   `power_port.py:958, 252-253`).
3. The power port is **NON-ACCEPTING** regardless of how small `residual_J` is.
   `can_support_power_port_acceptance` and
   `can_support_first_principles_acceptance` stay `False`. The run keeps emitting
   `engineering_candidate_not_validation`.
4. Acceptance unblocks **only** when *both*: (a) a source-backed residual
   tolerance is attached (a cited KR equation, an experimental error bar, or an
   explicit human review record), and (b) the six negative tests of Section 5
   pass. Until then `residual_J` is an *engineering-debug diagnostic only*
   (`interpretation: candidate_budget_for_engineering_debug_only_not_power_port_acceptance`,
   `power_port.py:959-961`).
5. The tracked-total-energy delta is **not** the power-port residual; they are
   distinct quantities. The existing flag `tracked_energy_delta_is_residual:
   False` (`power_port.py:254`) must remain `False`.

---

## 5. The six negative tests (audit A-5)

Each test corrupts one input and asserts the ledger / gate detects it. All six are
**required** (audit A-5 enumerates exactly these six). They extend the existing
`negative_test_policy` block (`power_port.py:240-251`).

| # | Negative test | Corruption injected | Assertion (what it proves) |
|---|---|---|---|
| N1 | **Sign reversal** | Flip the sign of `terminal_port_work_J` (or of the source-interface Poynting normal). | `residual_J` jumps by `~2 * terminal_port_work_J`; `residual_fraction` becomes O(1). Asserts the residual is sign-sensitive and a reversed convention cannot pass review. Proves the Section 3.1 convention is load-bearing, not cosmetic. |
| N2 | **Wrong domain** | Build `Omega` *including* `terminal_source_interface_faces` (source interface not excluded), OR shift `omega_volume_cells` by one cell so it spans current-free cells. | The domain-review packet fails: `terminal_source_interface_faces` is no longer disjoint from `Omega`, or `Omega` contains `J=0` cells. `volume_j_dot_e_work_J` and `residual_J` change measurably. Asserts Auluck's "J is zero outside Omega" + source-interface exclusion [KR: auluck:203-209] is enforced, not assumed. |
| N3 | **Omitted electrode work** | Drop term 4 (`electrode_interface_work_J`) from the residual sum (set it to 0). | `residual_J` no longer closes whenever the moving boundary `Sigma_p` carries motional power (`dL/dt != 0`). Asserts term 4 is a required ledger term — Auluck: stationary boundaries do not contribute but the moving boundary does [KR: auluck:426-429], and the `dL/dt` motional term is real [KR: auluck:780-783]. |
| N4 | **Low-current P/I singularity** | Drive `I(t) -> 0` (or below a current floor) while `volume_j_dot_e_work` stays finite, then form `U_DPF = P / I`. | The low-current guard fires (`input_sequence_fallback_low_current`,
`hybrid_simulator.py:553`); `U_DPF` is NOT computed as `P/I` at low current; the singularity is flagged in `low_current_p_over_i_singularity`. Asserts Auluck eq 1's `1/I` factor [KR: auluck:173-197] cannot be evaluated through `I=0` and the runtime detects it instead of emitting `inf/NaN`. |
| N5 | **First-step fallback** | Run step 0 with no lagged field-work available (`lagged_field_work is None`). | `udpf_source` is `input_sequence_fallback_first_step`
(`hybrid_simulator.py:551`); the ledger marks step 0 as fallback and does NOT claim a closed first-step residual. Asserts the begin-of-run state is handled explicitly (no use of an uninitialized field-power term), and time-centering N5 also checks all five terms share one centering. |
| N6 | **Default-mode leakage** | Run in the default circuit mode (`input_sequence`, `_CIRCUIT_UDPF_MODES`, `hybrid_simulator.py:21-24`) and attempt to read the power port as an *accepted* field/circuit closure. | The packet still reports `accepted_load_power_source: "none"`,
`active_load_relation: input_terminal_voltage_sequence_not_active_load_authority`, and `can_support_power_port_acceptance: False`. Asserts the default input-voltage-sequence mode cannot silently be promoted to an accepted Auluck/Poynting power port — the candidate J.E ledger never leaks into acceptance state. |

Each negative test asserts on the *emitted packet fields*, not on internal state,
so the artifact linter can re-check them. Tests are pure functions over a
constructed corrupt input — no run interdependency.

---

## 6. Blocked / no-local-source gaps

These are flagged per the rule "If a needed relation is NOT in KR, mark it
'blocked — no local source'; do not invent."

- **G1 — Auluck moving-boundary integrand (term 4) is OCR-illegible.** Auluck
  equations 5 and 6 (lines ~458–664) — which give the *closed-form integrand* of
  the moving-boundary / electrode-interface term — are rendered as scrambled,
  out-of-order math glyphs in this KR extract and cannot be transcribed. The
  *role* of the term is verified from legible prose [KR: auluck:426-429, 780-783];
  the *formula* is not. **Implementation must not invent the integrand.** Two
  source-honest options: (a) the lead supplies a legible rendering of Auluck eq
  5–6 (re-OCR or the original PDF), or (b) compute term 4 as the closure estimate
  `electrode_interface_work_J__closure_estimate_not_independent =
  terminal_port_work - volume_j_dot_e_work - wall_poynting - stored_em_delta` and
  label it explicitly non-independent, which means the residual is then trivially
  zero and **cannot** support acceptance until an independent integrand exists.
- **G2 — No source-backed residual tolerance.** No file in
  `KnowledgeReference/` (within the three sources in scope, or surveyed) gives a
  numerical closure tolerance for a DPF field/circuit power-port residual.
  `accepted_residual_tolerance` stays `not_attached` (Section 4). Acceptance
  blocked until a cited tolerance, an experimental error bar, or a human review
  record exists.
- **G3 — No source-backed high-order time-centering.** No KR source in scope
  prescribes a specific time-centering / quadrature order for the power-port
  integral. The packet declares only *step-consistency* (all five terms on the
  same centering); it does **not** claim an accepted accuracy order.
- **G4 — PF-1000 material geometry not yet reviewed.** The wall/insulator/bore
  electrode masks needed to build `wall_material_faces` precisely depend on the
  reviewed PF-1000 material geometry, which is audit finding **A-7** and still
  candidate. WP-N1 must consume the A-7 masks once reviewed; until then
  `wall_material_faces` is an engineering approximation and the wall Poynting
  term carries a `geometry_candidate_not_reviewed` flag.
- **G5 — Hybrid-PIC U_DPF uses a flux relation, not Auluck's volume J.E.** The
  hybrid-PIC source defines `U_DPF = d(integral B.ds)/dt`
  [KR: fully-electromagnetic-hybrid-pic-fluid-dpf-neutron-yield-acb71fa9.md:763-767,
  Eq.36], a magnetic-flux time-derivative — **not** the Auluck volume `J.E`
  relation. The hybrid-PIC source is cited only for the *external-circuit ODE
  pattern* (Eq.35, lines 752-759) and the *current-derived azimuthal magnetic
  boundary* (Eq.34, `B_theta = mu I / 2 pi r`, lines 748-751). It is **not** a
  source for the Auluck `Omega` J.E ledger. Do not conflate the two `U_DPF`
  definitions in the implementation.

---

## 7. Acceptance gate (unchanged, restated)

The simulator continues to report `engineering_candidate_not_validation`. The
WP-N1 power port emits a complete *candidate* five-term ledger with domain mask
hashes, sign convention, step-consistent time-centering, residual, and six
passing negative tests — and still sets `can_support_power_port_acceptance:
False` until gaps G1–G2 are closed by source-verified material reviewed by the
lead. Implementing this packet does **not** constitute accepted power-port
authority (audit "Do Not Claim": `accepted power-port authority`).
