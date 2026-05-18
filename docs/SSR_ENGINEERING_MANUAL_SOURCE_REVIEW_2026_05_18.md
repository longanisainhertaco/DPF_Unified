# SSR Engineering Manual Source Review

Date: 2026-05-18

Reviewed proposal:

- `/Users/anthonyzamora/Downloads/DPF_First_Principles_Simulator_SSR_and_Engineering_Manual_2026_05_18.docx`

Review scope:

- Decide whether to execute the proposed instruction set.
- Verify proposed solutions against local `KnowledgeReference/` only.
- Preserve the simulator's current fail-closed status.

## Executive Verdict

Do not execute the manual as an accepted implementation plan.

Execute selected parts as an engineering Stage 0 work plan, but keep every new
operator behind candidate/specification status until local source support,
residual closure, negative tests, convergence, restart reproducibility, and
review are attached.

The strongest reason is the manual's central power-port instruction:

```text
Replace volume J.E as circuit driver.
Use quasi-TEM boundary-port line voltage as the driver.
Retain volume J.E as conservation ledger.
```

That replacement is not yet verified by the local source of truth. The local
Auluck source supports a field-power circuit-element construction using
`- integral(J.E)dV / I` over a declared domain, with source-interface exclusion.
The local hybrid-PIC source uses an external circuit equation and obtains
`U_DPF` through magnetic-field integration/time differentiation. Neither source
currently establishes the manual's quasi-TEM Sigma line-voltage operator as the
accepted DPF circuit driver.

The right execution path is therefore:

```text
Execute: source-backed ledgers, deck lock, geometry review, negative tests,
         semi-implicit candidate numerics, closure packets, and certificate
         scaffolding.

Defer: replacing the active circuit driver with Sigma line voltage.

Reject as accepted: any claim that the manual itself makes accepted_power_port
                    solved.
```

## Operator Difference

The Sigma/quasi-TEM proposal is a boundary line-voltage idea:

```text
U_DPF ~= integral_path E.dl at a declared Sigma port plane
```

That is different from the source-backed options now carried in the runtime:

| Operator | What it measures | Source status in `KnowledgeReference/` | Runtime decision |
|---|---|---|---|
| Auluck volume `J.E` voltage | Total field work in a declared DPF domain divided by terminal current, with source-interface exclusion. | Supported by local Auluck source. | Keep as candidate telemetry/lagged driver only; not accepted until domain/sign/time-centering/low-current packet is solved. |
| Poynting surface flux | EM power crossing a declared source/interface surface. | Supported by Auluck plus NRL Poynting theorem. | Use for candidate ledger design; no accepted wall/source/electrode partition yet. |
| Hybrid-PIC `U_DPF` pattern | External circuit voltage term obtained from resolved field integration/time differentiation. | Supported by local hybrid-PIC source as an architecture pattern. | Useful guidance, not same-scope PF-1000 acceptance evidence. |
| Sigma/quasi-TEM line voltage | Electric-field line integral across a proposed boundary plane. | Not found as a verified DPF driver in local sources. | Defer as exploratory diagnostic only; do not use as accepted or primary driver. |

The practical difference is power closure. Auluck and Poynting are power-domain
operators: they start from energy transfer between the external circuit and the
plasma/domain. The Sigma line-voltage operator starts from a path voltage and
then still needs a current/power closure, path-independence proof, source-plane
definition, and electrode/wall work accounting. That may become useful, but it
cannot replace the source-backed power-port path yet.

## Source Evidence

### Auluck Circuit-Element Source

Local source:

- `KnowledgeReference/auluck-2021-dpf-circuit-element.md`

Evidence:

- Lines 151-154 define the plasma focus as an idealized two-terminal circuit
  element with voltage/current and define voltage through work against the
  electric field between terminals.
- Lines 163-178 then give the physical circuit-element basis as a volume
  field-power relation:

```text
V12(t) = - integral_Omega(J.E)d^3r / I(t)
```

- Lines 199-209 state that the voltage expression must allow every physical
  phenomenon in the chamber to draw power from the external circuit, and that
  the integration domain excludes the interface between the circuit element and
  the external power source.
- Lines 235-262 connect the interface power to Poynting flux and `I(t)V(t)`.
- Lines 421-447 state that moving-boundary and surface terms change by phase,
  and that the cathode plate interface excluded from the domain carries the
  power input.

Interpretation:

- Source supports a domain-declared field-power port and Poynting ledger.
- Source does not support silently treating all negative `J.E` as unphysical.
- Source does not directly promote a quasi-TEM line integral at a Sigma plane as
  the primary DPF driver.

### Hybrid PIC Circuit Source

Local source:

- `KnowledgeReference/fully-electromagnetic-hybrid-pic-fluid-dpf-neutron-yield-acb71fa9.md`

Evidence:

- Lines 741-789 describe solving current from an external circuit, applying
  magnetic boundary conditions at the injection port, and using:

```text
d(L0 I)/dt = V0 - r0 I - U_DPF - Q/C0
dQ/dt = I
```

- Lines 761-765 say `U_DPF` is calculated through magnetic-field integration
  over the DPF system followed by time differentiation.
- Lines 992-1005 describe current and plasma-voltage histories, including
  large voltage oscillations near pinch formation.

Interpretation:

- Source supports explicit external circuit coupling and source-derived
  `U_DPF`.
- Source supports `U_DPF` as a real runtime quantity with pinch oscillations.
- Source does not verify the manual's line-voltage Sigma operator.

### Poynting Theorem Source

Local source:

- `KnowledgeReference/2019nrlplasma-formulary-037290d4.md`

Evidence:

- Lines 1880-1888 state Poynting's theorem:

```text
dW/dt + surface Poynting flux = - integral_V J.E dV
```

Interpretation:

- Source supports using `J.E`, stored field energy, and surface flux in a
  conservation ledger.
- Source supports treating negative local `J.E` as part of signed energy
  exchange, not as an automatic limiter condition.

### PF-1000 Machine Deck Sources

Local sources:

- `KnowledgeReference/radiation-physics-and-chemistry-188-2021-109633.md`
- `KnowledgeReference/experimental-study-of-the-structure-of-the-plasma-current-sheath-on-the-pf-1000-facility-705bcc83.md`

Evidence:

- Radiation Physics and Chemistry source lines 108-142 list PF-1000 electrode
  dimensions, `C0 = 1332 uF`, `16 kV`, `170.5 kJ`, `1100-1300 kA`, `1.05` and
  `1.2 Torr` deuterium, current/voltage/current-derivative diagnostics,
  neutron detectors, timing convention, and about `7 us` from breakdown to
  current-derivative dip.
- Same source lines 262-270 list shot-12581-like example parameters:
  `L0 = 25 nH`, `C0 = 1332 uF`, `r0 = 6.1 mOhm`, `b = 16 cm`,
  `a = 11.55 cm`, `z0 = 48 cm`, `V0 = 16 kV`, `p0 = 1.2 Torr`.
- Experimental current-sheath source lines 340-356 list vacuum chamber
  dimensions, 12 cathode rods of `80 mm`, center-electrode radius `115.5 mm`,
  outer-electrode radius `200 mm`, center-electrode length `460 mm`, alumina
  insulator length `85 mm`, and capacitor bank `1332 uF`.

Interpretation:

- The manual's deck-lock procedure is source-supported and should execute.
- PF-1000/PF-1000U values must not be mixed unless the target shot/family source
  is explicitly bound.

### Startup Sources

Local sources:

- `KnowledgeReference/gribkov-2007-pf1000-jphysd-part2.md`
- `KnowledgeReference/effect-of-current-sheath-initiation-on-the-radial-collapse-and-energetic-particle-accelera-b2e95b88.md`

Evidence:

- Gribkov lines 55-80 describe DPF stages: gas breakdown along the insulator,
  non-equilibrium kinetic surface discharge over nanoseconds to hundred
  nanoseconds, inverse-pinch expansion to cathode bars, then microsecond axial
  acceleration.
- Current-sheath initiation source lines 616-642 state that Paschen-style
  pressure regimes are only guidelines and that the connection between
  Paschen physics and DPF breakdown is fragile.

Interpretation:

- The manual is correct to keep startup fail-closed.
- Calibrated lift-off seeding may be useful for post-lift-off engineering
  progress, but it cannot claim neutral-gas whole-shot first principles.

### Closure And Neutron Sources

Local sources:

- `KnowledgeReference/doi-10-1016-j-vacuum-2004-05-019-f931cb0b.md`
- `KnowledgeReference/2019nrlplasma-formulary-037290d4.md`
- `KnowledgeReference/anisotropy-of-the-emission-of-dd-fusion-neutrons-caused-by-the-plasma-focus-vessel-527cc533.md`

Evidence:

- Vacuum source lines 31-32 mention Braginskii transport, ionization kinetics,
  anomalous resistivity in low-density regions, and current flow.
- Vacuum source lines 240-259 discuss anomalous resistivity, ions, ionization,
  and radiation losses.
- NRL source lines 2701-2704, 3250-3283, 4580-4686, and 4736-4774 provide
  formulas/families for resistivity, heat flux, ionization/recombination, and
  radiation.
- PF-1000 neutron anisotropy source lines 94-130 state that TOF spectra can
  estimate neutron energy spectra and that vessel/wall scattering affects
  observed spectra and anisotropy.
- Same source lines 168-180 describe activation detectors and TOF probes at
  multiple distances.
- Same source lines 276-304 state that scattered and direct neutrons should be
  separated before transforming TOF spectra into velocity or energy
  distributions.

Interpretation:

- The manual's closure-packet and neutron-mechanism separation work orders are
  directionally source-supported.
- Individual formulas/operators still need source-specific packet review before
  implementation acceptance.

### Numerical Fidelity Sources

Local sources:

- `KnowledgeReference/a-structure-preserving-semi-implicit-imex-finite-volume-scheme-for-ideal-magnetohydrodynamics-at.md`
- `KnowledgeReference/2023-mhd-numerics-mpi-amrvac-30-updates-to-an-open-source-simulation-framework.md`

Evidence:

- Semi-implicit IMEX source lines 46-62 support semi-implicit finite-volume MHD
  methods, divergence-free preservation, implicit magnetic terms, and large
  timestep stability in relevant regimes.
- Same source lines 1768-1782 report a GMRES residual on the order of `1e-14`
  for that paper's linear system.
- MPI-AMRVAC source lines around 1774-1887 discuss IMEX Euler, IMEX
  trapezoidal/Crank-Nicolson, and IMEX midpoint families.

Interpretation:

- Semi-implicit numerics are source-supported as a numerical strategy.
- The manual's exact `theta >= 0.5` and Picard residual `<= 1e-10` are not
  established by these DPF sources. They may be project engineering thresholds,
  not source-truth physics.

## Execute / Defer / Reject Matrix

| Manual proposal | Verdict | Reason |
|---|---|---|
| Keep complete-simulator claims blocked. | Execute. | Matches current artifact status and source-discipline requirements. |
| Use certificate-driven development with explicit statuses. | Execute. | Engineering-safe and consistent with source-scoped fail-closed workflow. |
| Lock PF-1000 deck values and reject same-scope drift. | Execute now. | PF-1000 geometry/circuit/gas/timing are directly supported by local sources. |
| Keep startup fail-closed; allow calibrated lift-off seed only as post-lift-off engineering scope. | Execute. | Local startup sources show breakdown/flashover are complex and not solved by current deck. |
| Remove negative `J.E` clipping from an accepted path. | Execute only after accepted power-port tests exist. | Negative `J.E` can be physical, but the accepted path still needs domain/sign/time-centering proof. |
| Four-term Poynting/energy ledger. | Execute as candidate telemetry now. | Poynting theorem and Auluck support signed energy ledgers; exact decomposition needs implementation review. |
| Replace volume `J.E` driver with quasi-TEM Sigma line voltage. | Defer; do not execute as driver yet. | Not verified in local DPF source. Auluck supports volume field-power voltage; hybrid source uses magnetic-flux-derived `U_DPF`. |
| Add Sigma path-independence tests. | Execute as exploratory telemetry only. | Useful engineering test, but the Sigma line-voltage operator itself is not source-accepted yet. |
| No primary `1/I` driver pole. | Defer as accepted physics; execute as numerical safety study. | Auluck's source formula includes division by `I`; near-zero handling is a numerical design concern that must not erase the source relation. |
| Semi-implicit circuit coupling with theta/Picard residual target. | Execute as candidate numerical experiment. | Semi-implicit/IMEX methods are source-supported; exact theta and tolerance are project-defined until reviewed. |
| Stage 0 thresholds: `R_pp <= 2%`, cumulative `<= 1% E_bank`, blocked-step fraction `<= 5%`. | Defer as acceptance thresholds. | These are not verified from local sources; keep as proposed engineering gates pending review. |
| Closure packets for energy, radiation, ionization, anomalous transport, ablation, collisions, stopping. | Execute packetization and narrow tests. | Source families exist, but each operator must be individually reviewed before acceptance. |
| Mechanism-separated neutron authority and detector folding. | Execute planning and packet scaffolding. | PF-1000 neutron sources support TOF/anisotropy/detector-response complexity. |
| 12 us segmented reproducibility and limiter-off proof. | Execute. | Necessary numerical-fidelity work; not a physics claim by itself. |

## Recommended Execution Plan

### Execute Immediately

1. Create Stage 0 packet scaffolds:
   - `power_port_source_review`
   - `power_port_domain_review`
   - `power_port_sign_review`
   - `power_port_time_centering_review`
   - `power_port_energy_ledger_review`
   - `negative_test_plan`

2. Preserve current `lagged_auluck_volume_j_dot_e` as candidate telemetry.

3. Implement the four-term ledger as a non-promoting packet:

```text
port work
volume J.E work
stored EM energy delta
wall Poynting flux excluding declared port/source interface
electrode/interface work placeholder, explicitly blocked until modeled
```

4. Add negative tests that do not require accepting a new driver:
   - sign reversal must fail residuals
   - domain corruption must fail residuals
   - time-centering downgrade must fail residuals
   - low-current `P/I` singularity must be detected and reported

5. Add segmented run support for the candidate source-sign path before another
   full `12 us` attempt.

6. Lock the PF-1000/Akel machine deck against the cited source values and emit
   a deck-diff packet.

### Defer Until Source Support Exists

1. Replacing the circuit driver with Sigma line voltage.
2. Treating path-independence at Sigma as an acceptance criterion.
3. Declaring exact residual percentages as acceptance thresholds.
4. Treating PF-1000U claims as same-scope with PF-1000/Akel unless the source
   target is explicitly bound.

### Reject

1. Any immediate promotion of `accepted_power_port`.
2. Any implementation that removes the fail-closed status because the manual
   says Stage 0 is solve-ready.
3. Any driver replacement that bypasses Auluck's domain, source-interface, and
   Poynting-theorem requirements.

## Final Decision

We should execute the manual as a candidate engineering work-order generator,
not as a source-verified solution set.

The manual is useful because it organizes Stage 0 around the right problem:
power-port authority. It is not sufficient because its central proposed
replacement driver is not verified by the local `KnowledgeReference/` corpus.
The safest path is to implement source-backed ledgers, packet scaffolds,
negative tests, deck locking, and segmented reproducibility first. Only after a
source-supported port operator closes the ledger and passes negative tests
should we replace the active circuit driver or change acceptance status.
