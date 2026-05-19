# Auluck 2021 — DPF as a Circuit Element — Verified Equation Extract

## Provenance

Equations transcribed verbatim on 2026-05-19 by a direct read of the primary PDF
`archive_reference_OLD/references/papers/core-dpf/auluck-2021-dpf-circuit-element.pdf`,
pages 6-11.

This companion file exists because the auto-extracted
`KnowledgeReference/auluck-2021-dpf-circuit-element.md` renders equations (2)-(14)
as OCR-garbled, out-of-order math tokens (that file, lines ~262-664). This file
is the legible local authority for the WP-N1 / WP-N1B power-port ledger.

Paper: S. K. H. Auluck, "On the representation of dense plasma focus as a circuit
element."

## Domain (p.6-7, verbatim prose)

> "This 3-D spatial integration is over a domain Omega such that J is zero
> outside it. Excluded from this domain is the interface between the 'circuit
> element' and the external power source, through which, the power enters the
> device. In the case of the plasma focus, this would be the cathode plate that
> is in contact with the insulator and the squirrel cage, or its smaller portion
> in the initial phase." (p.6)

> "This bounding surface, designated Sigma, comprises the contact surfaces
> between the plasma and each of the two electrodes ... as well as the extreme
> boundary of the plasma ... The domain Omega is not a simply-connected domain
> ... topologically it is a toroid." (p.6-7)

> "The second integral is evaluated only on the moving boundary Sigma_p of the
> domain Omega since stationary boundaries do not contribute to it. The third
> integral is evaluated over the entire surface Sigma." (p.8)

> "Along the cathode plate at the bottom, which is excluded from the domain for
> being the interface between the circuit element and the power source, the
> surface integral [of the Poynting vector] is exactly equal to the power input
> I(t) V(t)." (p.8)

`Sigma_p` is the **moving** part of the boundary `Sigma`. The
electrode/power-source interface is NOT part of `Omega`.

## Equations (verbatim, ASCII transcription; PDF page cited)

Eq (1) [p.6] — circuit-element relation. NOTE THE LEADING MINUS SIGN:
```
V_12(t) = -(1 / I(t)) * integral_Omega d^3r ( J . E )
```

Eq (2) [p.7] — Poynting's theorem, local form:
```
J . E = -(d/dt)( (1/2) eps0 E^2 + (1/2) mu0^-1 B^2 ) - mu0^-1 div( E x B )
```

Eq (3) [p.7] — eq (1) rewritten via (2), Gauss + Reynolds transport:
```
I(t) V(t) = integral_Omega d^3r (d/dt)( (1/2) eps0 E^2 + (1/2) mu0^-1 B^2 )
          + integral_Omega d^3r mu0^-1 div( E x B )

          = (d/dt) integral_Omega d^3r ( (1/2) eps0 E^2 + (1/2) mu0^-1 B^2 )
          - integral_Sigma_p dS . v ( (1/2) eps0 E^2 + (1/2) mu0^-1 B^2 )
          + mu0^-1 closed_integral_Sigma dS . ( E x B )
```

Eq (4) [p.8] — Generalized Ohm's Law:
```
E = -( v x B ) + eta J
```

Eq (5) [p.8] — the Sigma_p part of the Poynting surface integral, via eq (4):
```
mu0^-1 closed_integral_Sigma_p dS . ( E x B )
   = mu0^-1 closed_integral_Sigma_p dS . v ( B . B )
   - mu0^-1 closed_integral_Sigma_p dS . B ( B . v )
   + mu0^-1 closed_integral_Sigma_p dS . ( eta J x B )
```

Eq (6) [p.8] — the six-term power balance. Term labels I-VI are Auluck's own:
```
I(t) V(t) = (d/dt) integral_Omega d^3r ( (1/2) mu0^-1 B^2 )          [ I  ]
          + integral_Sigma_p dS . v ( (1/2) mu0^-1 B^2 )             [ II ]
          + (d/dt) integral_Omega d^3r ( (1/2) eps0 E^2 )            [ III]
          - integral_Sigma_p dS . v ( (1/2) eps0 E^2 )               [ IV ]
          + mu0^-1 closed_integral_Sigma_p dS . ( eta J x B )        [ V  ]
          - mu0^-1 closed_integral_Sigma_p dS . B ( B . v )          [ VI ]
```

Eq (7) [p.9]:  `P_L = I (d/dt)( L I ) = (d/dt)( (1/2) L I^2 ) + (1/2) I^2 (dL/dt)`

Eq (8) [p.9]:  `P_C = V (dQ/dt) = V (d/dt)( C V ) = (d/dt)( (1/2) C V^2 ) + (1/2) V^2 (dC/dt)`

Eq (9) [p.9]:  `P_R = I . V_R = R I^2`

Eq (10) [p.9]:  `L_p == (1 / I^2) integral_Omega d^3r ( (1/2) mu0^-1 B^2 )`

Eq (11) [p.10]: `C_p == (1 / V^2) integral_Omega d^3r ( (1/2) eps0 E^2 )`

Eq (12) [p.10]: `R_p = I^-2 mu0^-1 closed_integral_Sigma_p dS . ( eta J x B )`

Eq (13) [p.10]: `(1/2) I^2 (dL_p/dt)  !=  integral_Sigma_p dS . v ( (1/2) mu0^-1 B^2 )`

Eq (14) [p.10]: `(1/2) V^2 (dC_p/dt)  !=  - integral_Sigma_p dS . v ( (1/2) eps0 E^2 )`

## Term identification I-VI (p.9-10, verbatim prose)

- Terms **I** and **III**: "time derivatives of the total magnetic and electric
  energy that clearly must correspond to d/dt((1/2)LI^2) and d/dt((1/2)CV^2)
  respectively."
- Terms **II** and **IV**: "The terms proportional to dL/dt, which is sometimes
  called as the motional impedance ... and dC/dt ... are clearly dependent on the
  velocity with which the dimensions of the inductance and capacitance are
  changing ... Terms II and IV have this property."
- Term **V**: "proportional to the resistivity and to the square of the current
  and is independent of the velocity. This property is shared by the expression
  for power through a resistance as given in (9)."
- Term **VI**: "depends on the component of velocity along the magnetic field
  and on the magnetic field component normal to the plasma surface. In a
  conventional description ... where the magnetic field is purely azimuthal and
  the azimuthal component of velocity is zero, this term would be zero. But when
  the PF-1000 phenomenology involving axial magnetic field in the radial phase
  is considered ... this term becomes non-zero. ... Term VI apparently has no
  analog in circuit theory and would thus have to be also accounted for in terms
  of an anomalous impedance."

Eqs (13)-(14): the plasma-inductance / plasma-capacitance time derivatives do
NOT in general equal terms II / IV; "The difference ... would have to be
accounted for by invoking an 'anomalous impedance' -- an impedance that has no
analog in circuit theory" (p.10).

## Symbol map (SI)

| Symbol | Definition | SI units |
| --- | --- | --- |
| `I(t)` | total terminal current | A |
| `V(t)`, `V_12(t)` | terminal voltage of the circuit element | V |
| `Omega` | current-carrying volume, `J = 0` outside it; topological toroid; source interface excluded | m^3 domain |
| `Sigma` | closed bounding surface of `Omega` | m^2 surface |
| `Sigma_p` | the moving part of `Sigma` (stationary boundary parts do not contribute to the motional integrals) | m^2 surface |
| `E`, `B` | electric field, magnetic flux density | V/m, T |
| `J` | current density | A/m^2 |
| `v` | local plasma/material velocity | m/s |
| `eta` | resistivity | Ohm.m |
| `mu0`, `eps0` | vacuum permeability, permittivity | H/m, F/m |
| `dS` | oriented (outward-normal) surface element | m^2 |
| `L_p`, `C_p`, `R_p` | plasma inductance / capacitance / resistance (eqs 10-12) | H, F, Ohm |

## Sign convention (Auluck)

- `I(t) V(t)` on the LHS of eqs (3) and (6) is the power **input** to the device,
  crossing the excluded source interface (p.8).
- Eq (1) carries a leading **minus**: `V_12 = -(1/I) integral_Omega J.E`.
  Consistency check: `I.V_12 = -integral_Omega J.E`; integrating eq (2) over
  `Omega` gives `integral_Omega J.E = -(d/dt) integral_Omega u - mu0^-1
  closed_integral_Sigma dS.(E x B)`, hence `I.V_12 = (d/dt) integral_Omega u +
  mu0^-1 closed_integral_Sigma dS.(E x B)`, which is eq (3). The minus sign is
  load-bearing.

## What this source DOES and DOES NOT provide

PROVIDES (verbatim, legible):
- The six-term power balance, eq (6), with Auluck's own term labels I-VI.
- The `Sigma_p` moving-boundary surface-integral integrands, eq (5):
  `v(B.B)`, `B(B.v)`, `eta J x B`.
- The circuit-element correspondence eqs (7)-(9) and the plasma L/C/R
  definitions eqs (10)-(12), with the anomalous-impedance caveat eqs (13)-(14).

DOES NOT provide:
- An "electrode/interface contact-work" term. Auluck's balance has no such term;
  the electrode/power-source interface is **excluded** from `Omega` and its
  Poynting flux **is** the LHS `I(t)V(t)` input. A ledger term named
  "electrode_interface_work" has no Auluck basis as an independent quantity.
- Any numerical residual / energy-balance tolerance.
- Any time-centering or quadrature-order prescription for discretising the
  power integral.
