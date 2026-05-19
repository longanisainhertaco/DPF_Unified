# WP-N1B — Auluck Eq. 5/6 Source Status

Sprint: 2
Blocker IDs: WP-N1B, DPF-PHYS-020, gap G1
Date: 2026-05-19

## Verdict

Auluck eq. (5) and eq. (6) are **legible and now ingested into
`KnowledgeReference/`**. The blocker label `G1_auluck_eq_5_6_ocr_illegible`
carried by `src/dpf/first_principles/power_port.py` and the WP-N1 tests is
**inaccurate** and must be corrected: the equations were never physically
illegible — only the auto-extracted markdown was OCR-garbled.

A separate, more important finding: the audit's framing of WP-N1B term 4 as
"Auluck eq. 5/6 moving-boundary **electrode work**" is a **category error**.
Auluck's power balance has no electrode-contact-work term.

## Audit WP-N1B questions 1-5, answered

**Q1 — Is Auluck eq. 5/6 legible in local `KnowledgeReference/`?**
The auto-extracted `KnowledgeReference/auluck-2021-dpf-circuit-element.md` renders
eqs. (2)-(14) as OCR-garbled, out-of-order math tokens (lines ~262-664) — NOT
legible. But the primary PDF is on disk:
`archive_reference_OLD/references/papers/core-dpf/auluck-2021-dpf-circuit-element.pdf`.
It was read directly this session (pages 6-11) and the equations transcribed
verbatim into a new verified extract:
`KnowledgeReference/auluck-2021-dpf-circuit-element-EQUATIONS-VERIFIED.md`.
**Eq. 5/6 are now legible local KR authority.**

**Q2 — Exact equations, symbols, units, sign, discrete form.**
Eqs. (1)-(14), symbol map, and sign convention are in the verified extract.
Eq. (5) — the `Sigma_p` moving-boundary Poynting integrand, via Generalized
Ohm's Law eq. (4):
```
mu0^-1 closed_integral_Sigma_p dS.(E x B)
  = mu0^-1 closed_integral_Sigma_p dS.v(B.B)
  - mu0^-1 closed_integral_Sigma_p dS.B(B.v)
  + mu0^-1 closed_integral_Sigma_p dS.(eta J x B)
```
Eq. (6) — the six-term power balance `I(t)V(t) = I + II + III + IV + V + VI`,
terms defined verbatim in the extract. Discrete form: see
`WP_N1B_POWER_PORT_ACCEPTANCE_PROPOSAL.md` section 5.

**Q3 / Q4 — local alternative / external candidate.**
Not applicable: eq. 5/6 are legible and now local KR authority. No external
ingestion is required; the source was already on disk and is now extracted.

**Q5 — How is "electrode/interface work" computed independently?**
It is not — because **it is not an Auluck quantity.** Auluck eq. (6) decomposes
the input power `I(t)V(t)` into six terms, every one a volume integral over
`Omega` or a surface integral over the moving boundary `Sigma_p`:

| Term | Integral | Circuit analog (eqs 7-9) |
| --- | --- | --- |
| I | `d/dt integral_Omega (1/2 mu0^-1 B^2)` | `d/dt(1/2 L I^2)` stored magnetic |
| II | `integral_Sigma_p dS.v (1/2 mu0^-1 B^2)` | `1/2 I^2 dL/dt` motional ("motional impedance") |
| III | `d/dt integral_Omega (1/2 eps0 E^2)` | `d/dt(1/2 C V^2)` stored electric |
| IV | `- integral_Sigma_p dS.v (1/2 eps0 E^2)` | `1/2 V^2 dC/dt` motional |
| V | `mu0^-1 closed_integral_Sigma_p dS.(eta J x B)` | `R I^2` resistive |
| VI | `- mu0^-1 closed_integral_Sigma_p dS.B(B.v)` | none — "anomalous impedance" |

Auluck **excludes** the electrode/power-source interface from `Omega` (PDF p.6);
the Poynting flux across that interface **is** the LHS `I(t)V(t)` — the input,
not a balance term. There is no "electrode contact work" integrand in Auluck and
none anywhere else in `KnowledgeReference/`.

Consequence: the WP-N1B path to an *independent* fifth/sixth term is to compute
Auluck terms **II, IV, V, VI** as independent `Sigma_p` surface integrals using
the eq. (5) integrands — not to invent an "electrode work" term and not to leave
it as the closure residual `terminal - volume - wall - stored`. This is
source-backed (eq. 5 gives the integrands verbatim) and is the implementation
proposed in `WP_N1B_POWER_PORT_ACCEPTANCE_PROPOSAL.md`.

## Verified correction to the existing source packet

`docs/ssr_audit_2026_05_18/WP-N1_power_port_source_packet.md:45` transcribes
eq. (1) as `V_12(t) = (1/I(t)) integral_Omega (J.E) d^3r` — **without the leading
minus sign**. The PDF (p.6) shows `V_12(t) = -(1/I(t)) integral_Omega d^3r (J.E)`.
The minus is load-bearing (consistency check in the verified extract). WP-N1B
implementation must verify whether `power_port.py` inherited this sign error.

## Source-status classification

- Auluck eq. (1)-(14): `local_authority_packet_complete` —
  `KnowledgeReference/auluck-2021-dpf-circuit-element-EQUATIONS-VERIFIED.md`.
- Independent term-4/"electrode work" integrand: **does not exist** in Auluck;
  the audit's premise is withdrawn. The source-faithful replacement (Auluck
  terms II/IV/V/VI) is `local_authority_packet_complete`.
- Residual tolerance: `blocked_no_source` — see
  `WP_N1B_RESIDUAL_TOLERANCE_SOURCE_STATUS.md`.
- Time-centering: `blocked_no_source` for accuracy order — see
  `WP_N1B_TIME_CENTERING_PROPOSAL.md`.
