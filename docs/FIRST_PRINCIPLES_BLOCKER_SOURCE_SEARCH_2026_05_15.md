# First-Principles Blocker Source Search - Startup BVP

Date: 2026-05-15

Scope: local source of truth only. Scientific claims in this note are limited to
`KnowledgeReference/` plus the already ingested user PDF record
`KnowledgeReference/fully-electromagnetic-hybrid-pic-fluid-dpf-neutron-yield-acb71fa9.md`.

Blocker: `FP-5`, source-backed startup boundary-value problem.

Question: can the current source of truth support a true first-principles DPF
startup from neutral gas, high-voltage breakdown, insulator flashover,
preionization, current-density formation, and sheath liftoff?

## Verdict

The source of truth answers part of the blocker but does not close it for a
whole-shot first-principles claim.

The corpus supports a fail-closed startup contract:

- A whole-shot claim must start from a source-backed breakdown or flashover
  boundary-value problem, or import a reviewed PIC-derived sheath state with
  fields, particles, thermodynamic state, current density, and provenance.
- A seeded ionized layer is not accepted startup physics. It can remain an
  engineering initializer or a handoff state only.
- The newly ingested hybrid PIC-fluid DPF paper supports a source-backed
  end-of-rundown sheath candidate, not a full breakdown-to-liftoff startup
  model. It starts from an already formed sheath near the end of rundown.
- PF-1000 source material supplies geometry and qualitative startup constraints,
  but not a complete resolved t=0 breakdown BVP packet.

Therefore `FP-5` remains blocked for an accepted full-shot first-principles
simulator until the implementation has either a source-backed breakdown solver
or reviewed imported startup fields/particles for the same claimed device and
shot scope.

## Source Answers

| Source | What it answers | What remains blocked |
| --- | --- | --- |
| `KnowledgeReference/fully-electromagnetic-hybrid-pic-fluid-dpf-neutron-yield-acb71fa9.md:613-735` | Provides a concrete hybrid PIC-fluid initialization for an already formed sheath: staggered Yee mesh, absorbing/PML style boundaries, background deuterium density, sheath particle population, temperatures, drift velocity, sheath slab location, and quasi-neutral cell initialization. | This is an end-of-rundown state. It does not solve neutral-gas breakdown, surface flashover, avalanche, preionization, or the initial insulator current sheet. |
| `KnowledgeReference/fully-electromagnetic-hybrid-pic-fluid-dpf-neutron-yield-acb71fa9.md:1219-1220` | States that the method enforces quasi-neutrality and does not resolve Debye sheath and near-wall sheath microphysics. | Near-insulator surface flashover, wall sheath microphysics, double layers, and secondary emission remain outside that model. |
| `KnowledgeReference/unlimited-release-printed-september-2009-alegra-hedp-simulations-of-the-dense-plasma-focus.md:245-392` | Establishes that DPF operation starts with high-voltage breakdown along the insulator, then lift-off and rundown; also states that arbitrary MHD seed layers were used as engineering starts and that PIC-derived sheath initiation was the desired source for MHD initialization. | The actual LSP field/particle arrays are not present in the repo, and the source does not give a reusable PF-1000/Akel startup data packet. |
| `KnowledgeReference/unlimited-release-printed-september-2009-alegra-hedp-simulations-of-the-dense-plasma-focus.md:555-585` | Supports importing PIC-derived ion/electron densities, temperatures, and magnetic field data into MHD as the right handoff pattern. | Handoff data provenance, mesh mapping, charge/current consistency, and field arrays still need local artifacts. |
| `KnowledgeReference/the-dense-plasma-focus-a-versatile-dense-pinch-for-diverse-applications.md:520-590` | Confirms the first phase is insulator breakdown, that pressure strongly affects uniformity, and that sheath speed and ionization during propagation matter. | Review-level text does not provide an executable breakdown BVP or material/electron-emission closure. |
| `KnowledgeReference/the-dense-plasma-focus-a-versatile-dense-pinch-for-diverse-applications.md:1488-1545` | Supports preionization and electrode/insulator shaping as real startup controls that affect neutron yield and reproducibility. | It gives operational evidence, not enough equations and same-scope initial fields for accepted simulation startup. |
| `KnowledgeReference/effect-of-current-sheath-initiation-on-the-radial-collapse-and-energetic-particle-accelera-b2e95b88.md:452-670` | Connects insulator length, fill pressure, breakdown regime, sheath mass, and velocity to performance; distinguishes low-pressure diffuse discharge, mid-pressure surface avalanche, and high-pressure filamentary behavior. | It is experimental and reduced-model comparison material. It does not by itself specify a first-principles surface-breakdown solver. |
| `KnowledgeReference/experimental-study-of-the-structure-of-the-plasma-current-sheath-on-the-pf-1000-facility-705bcc83.md:340-390` | Supplies PF-1000 electrode, insulator, capacitance, voltage, and sensitivity context. | It does not provide full startup field/current-density/ionization arrays or a breakdown equation packet. |
| `KnowledgeReference/gribkov-2007-pf1000-jphysd-part2.md:56-74` | Identifies PF-1000 startup as a short non-equilibrium kinetic surface-discharge phase with avalanche/streamer character. | The kinetic surface-discharge model is described qualitatively; executable closure remains missing. |
| `KnowledgeReference/beresnyak_2018_dpf_hawk_simulations.md:100-220` | Defines a local-plasma-injection startup mode with controlled density asymmetry and circuit coupling for Hawk-like simulations. | This is not standard neutral-fill DPF breakdown. It can be a separate device mode, not authority for PF-1000/Akel startup. |
| `KnowledgeReference/alfven-ionization-in-an-mhd-gas-interactions-code.md:420-447` | Provides a critical-ionization-velocity style ionization closure candidate for gas/plasma interaction where relative velocity exceeds a threshold. | It is supporting closure material only. It does not close insulator flashover, secondary emission, and electrode boundary initiation. |

## Accepted Startup Contract

The first-principles tool needs an explicit startup packet with one of these
modes:

| Mode | Status | Required payload |
| --- | --- | --- |
| `imported_pic_sheath_state` | Accepted only after review | Mesh mapping, particle species and weights, ion/electron density, ion/electron temperature, velocity, `E`, `B`, current density, charge consistency, boundary labels, source references, hashes, units, and conservation checks. |
| `source_backed_end_rundown_sheath` | Engineering candidate | The hybrid PIC-fluid end-of-rundown sheath initialization values with explicit limitation that it does not cover breakdown or liftoff. |
| `surface_breakdown_bvp` | Blocked | Equations and material/secondary-emission/avalanche/streamer/preionization closures, pressure regime classifier, electrode/insulator boundary data, and verification tests. |
| `plasma_injection_startup` | Separate device scope | Source-backed density and velocity distribution for devices intentionally using plasma injection, with no authority for standard neutral-fill startup. |
| `seeded_layer` | Rejected for accepted first-principles claims | May be used only for engineering smoke tests and must fail acceptance/readiness gates. |

Every packet must include:

- device geometry and insulator material/length/radii;
- gas species, pressure, temperature, and pressure-regime metadata;
- bank voltage and early circuit derivative consistency;
- initial density, ionization fraction or charge state, current density,
  velocity, electron temperature, ion temperature, `E`, `B`, and resistivity;
- preionization evidence or explicit absent-preionization status;
- sheath-liftoff timing/status and handoff interval if startup is imported;
- local source paths, hashes, line/page/figure references, units, and reviewer
  status;
- charge, current, `div B`, and energy consistency tests.

## Implementation Impact

Immediate code implication:

- Replace the generic startup label `source_backed_candidate_uniform` with
  explicit startup modes.
- Make the package-native first-principles deck preserve startup evidence
  fields instead of collapsing rich startup policy metadata to only `mode`.
- Ensure accepted readiness rejects `seeded_layer`,
  `source_backed_candidate_uniform`, and `source_backed_end_rundown_sheath`
  for whole-shot claims unless the run scope is explicitly narrowed.
- Add negative tests that prove a seeded or arbitrary uniform layer cannot pass
  a first-principles acceptance gate.

Next blocker to search after this one: `FP-6`, resolved field/circuit power-port
coupling.

## Current Implementation Ratchet

Implemented after this source search:

- `src/dpf/first_principles/startup_bvp.py` emits a fail-closed startup BVP
  packet with accepted, engineering-only, and rejected startup-mode classes.
- The startup packet now emits `startup_channel_status`,
  `startup_mode_status`, `mode_payload_status`, `candidate_input_policy`,
  `acceptance_gate`, and `negative_test_policy`.
- Candidate device/gas/circuit/startup inputs are explicitly usable for
  engineering initialization only; they cannot support whole-shot startup
  acceptance without a reviewed imported PIC state or source-backed
  surface-breakdown BVP payload.
- `src/dpf/first_principles/runner.py` now places the startup packet in run
  telemetry, manifest candidate evidence, validation packet summaries, and the
  upstream status maps for numerical, comparator/UQ, certificate, and
  generalization gates.
- `tests/test_first_principles_runner.py` proves end-of-rundown startup remains
  blocked, startup payload fields remain unreviewed, and seeded startup is
  rejected for first-principles acceptance.

Verified command:

- `python3 -m pytest tests/test_first_principles_runner.py`
  -> `8 passed`.
- `python3 -m pytest tests/test_first_principles_input_deck.py tests/test_first_principles_runner.py tests/test_first_principles_manifest.py tests/test_cli_first_principles_3d.py tests/test_cli_backend_options.py tests/test_server_readiness.py tests/test_kinetic_yield_history.py tests/test_hybrid_3d_loop.py tests/test_hybrid_pic_3d_validation_packet.py`
  -> `60 passed`.

Remaining blocker:

- No accepted neutral-gas breakdown, insulator flashover, preionization,
  current-density formation, and sheath-liftoff BVP exists in the local source
  truth or package-native implementation.
