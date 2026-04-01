# Physics Audit Report — 2026-03-31

Two-pass audit. First pass: specialist agents verified equations against cited papers.
Second pass: non-expert agents fetched actual paper text via web search and compared
equation-by-equation with zero domain assumptions.

## Methodology

Pass 1: Six specialist agents audited 46 modules against 68+ cited works.
Pass 2: Six general-purpose agents fetched actual paper PDFs/HTML from arXiv,
journals, and textbook sources. Every coefficient verified against published text.
Only peer-reviewed sources accepted as ground truth.

## Critical Finding: HLLS Solver Misattribution

The code cites Popovas et al. (2025, arXiv:2211.02438) as the HLLS reference.
After fetching and reading the actual paper:

**The code is NOT a faithful implementation of Popovas (2025).** Specific differences:

| Feature | Popovas (2025) paper | Code |
|---------|---------------------|------|
| Solver structure | HLLD (4-wave) | HLL (2-wave) |
| Entropy variable | `rho * S` where `S = c_v * ln(p/rho^gamma)` (logarithmic) | `Srho = p * rho^(1-gamma)` (power-law, Enzo/Bryan 2014 style) |
| Pressure recovery | `p = rho^gamma * exp(S/c_v)` (exponential) | `p = Srho * rho^(gamma-1)` (multiplicative) |
| Shock switching | Yes (Section 3.2) | No (dual-energy at output only) |
| Boris correction | Not mentioned | Applied (Gombosi 2002) |

The code is better described as: **"HLL Riemann solver with Enzo-style power-law
entropy tracer and Boris wave-speed capping."** The entropy pressure recovery avoids
the same catastrophic cancellation that Popovas addresses, but the mathematical
formulation is distinct. The power-law form `Srho = p * rho^(1-gamma)` is
self-consistent and the pressure recovery is correct for that definition.

**Recommendation**: Change the citation from "Popovas (2025) HLLS" to
"HLL with entropy-based pressure recovery (Bryan et al. 2014 dual-energy
entropy tracer), Boris correction (Gombosi et al. 2002)."

## Critical Finding: Lee Model Implementation Gaps

After fetching Lee & Saw (2014, Phys. Plasmas 21:072501) and Lee (2014,
J. Fusion Energy 33:319), three significant gaps were found:

### 1. Missing momentum correction in axial EOM

**Paper (Lee 2014, above eq. 1)**: The axial equation of motion is:
```
d/dt(m * dz/dt) = F_mag - p0*A
```
where `m = fm * rho0 * A * z` (swept mass). Expanding:
```
m * d^2z/dt^2 + (dm/dt) * (dz/dt) = F
d^2z/dt^2 = F/m - (dz/dt)^2 / z
```
The `-(dz/dt)^2/z` term represents deceleration from sweeping fresh gas.

**Code** (`mlx_snowplow.py:128`): Uses `a = F/m` without the `-(dz/dt)^2/z` term.

**Impact**: Sheath accelerates too fast. Arrives at anode end too early.

### 2. Missing fc factor on plasma inductance in circuit equation

**Paper (Lee 2014, eq. 2)**: Circuit equation uses `fc * L_p` and `fc * dL_p/dt`:
```
(L0 + fc*L_p) * dI/dt = V - r0*I - fc*(dL_p/dt)*I
```

**Code**: Passes raw `L_p` and `dL_p/dt` to circuit solver without `fc` factor.

**Impact**: Circuit sees wrong effective inductance and back-EMF.

### 3. Radial phase: snowplow instead of Lee slug model

**Paper (Lee 2014, eqs. 14-19)**: Radial phase has 4 coupled ODEs with separate
shock front (rs), piston (rp), column elongation (zf), and circuit current (I).

**Code**: Uses 2-variable snowplow (single radius + circuit). No shock/piston
separation, no adiabatic compression, no column elongation.

**Impact**: Radial phase dynamics are qualitatively different from the Lee model.

## Verified Correct (all confirmed against fetched papers)

### HLLD (Miyoshi & Kusano 2005, JCP 208:315) — ALL EQUATIONS MATCH

Verified against paper and cross-checked with Athena++ source code:
- SM contact speed (eq. 38): MATCH
- Star-state density (eq. 43): MATCH
- Star-state tangential velocity (eq. 44): MATCH
- Star-state tangential B (eq. 45): MATCH
- Star-state energy (eq. 48): MATCH
- Double-star velocities (eqs. 59-60): MATCH
- Double-star B-fields (eqs. 61-62): MATCH
- Double-star energy (eq. 63): MATCH
- Alfven wave speeds (eq. 51): MATCH
- Flux selection regions: MATCH (unconventional overwrite chain, functionally correct)

Two minor issues:
- `Bn_L` vs averaged `Bn` in `vB_L` dot product (line 764): LOW impact
- `pt_star` uses left-only formula, not L+R average (line 736): LOW impact

### WENO5-Z (Shu 2009 + Borges 2008 + Acker 2016) — ALL COEFFICIENTS MATCH

Verified every coefficient by Lagrange interpolation and cross-reference:
- FD candidate polynomials S0, S1, S2: MATCH (Shu 2009)
- Ideal weights 1/16, 10/16, 5/16: MATCH (Shu 2009)
- Smoothness indicators beta_0, beta_1, beta_2: MATCH (Jiang & Shu 1996 eq. 2.63)
- tau_5 = |beta_0 - beta_2|: MATCH (Borges 2008)
- Weight formula with p=2: MATCH (Acker et al. 2016 "WENO-Z+")
- eps = 1e-6: justified for float32

### SSP-RK3 (Shu & Osher 1988, JCP 77:439) — EXACT MATCH

Stage coefficients verified: (1, 0), (3/4, 1/4), (1/3, 2/3). Correct.

### Dedner GLM (Dedner 2002, JCP 175:645) — STRUCTURE MATCHES

- Psi equation: `dpsi/dt = -ch^2*divB - cr*psi` — MATCH
- Induction correction: `dB/dt += -grad(psi)` — MATCH
- Damping rate: Code uses `cr = ch/dx`. PLUTO default is `alpha=0.1` giving
  `cr = 0.1*ch/dx`. M&T2010 cite `alpha_p = 0.18`. Code is 10x stronger
  than PLUTO default but 5.6x weaker than M&T optimal.

### Powell 8-wave (Powell 1999, JCP 154:284) — EXACT MATCH

All source terms: `-div(B)*[0, B, v.B, v, 0]^T`. Confirmed.

### CT (Evans & Hawley 1988) — CORRECT

EMF computation, Br and Bz updates in cylindrical all match Faraday's law.
Uses simple CT (arithmetic averaging), not upwind CT despite citing
Gardiner & Stone (2005). The cite should reference Evans & Hawley (1988)
or note "simple CT" per Gardiner & Stone Section 4.2.

### Bremsstrahlung — CORRECT (Z^1 confirmed)

Rybicki & Lightman (1979) eq. 5.14a gives `P = 1.4e-27 * Z^2 * n_e * n_i * g * sqrt(T)`
in CGS. In the quasi-neutral form with `n_i = n_e/Z`:
`P = 1.4e-27 * Z * n_e^2 * g * sqrt(T)` → Z to the FIRST power.
Converting to SI: `P = 1.42e-40 * Z * n_e^2 * g * sqrt(T_K)` [W/m^3].
Code uses Z^1. **CORRECT.** (Previous audit incorrectly flagged this as Z^2.)

### Spitzer Resistivity — MATCH

`eta_perp = 1.03e-4 * Z * ln(Lambda) / T_eV^{3/2}` [Ohm*m].
Confirmed against NRL Formulary p.34 and Chen (2016) formula #23 in research DB.

## Bugs Fixed This Session

| # | Fix | Files | Verified |
|---|-----|-------|----------|
| 1 | Lee-More Coulomb log: `23 - 0.5*ln(n_e) + 1.5*ln(T_eV)` | mlx_transport.py:290 | NRL Formulary |
| 2 | Thermal diffusivity: added `(gamma-1)` factor | mlx_transport.py:667,684 | Energy conservation derivation |
| 3 | Resistive diffusion: added `-Br/r^2` sink | mlx_transport.py:549-551 | Vector Laplacian in cylindrical |
| 4 | Hall CFL: `B` instead of `B^2` | mlx_timestepper.py:330 | Whistler dispersion relation |
| 5 | Ohmic heating: `B_old^2 - B_new^2` | mlx_transport.py:553-568 | Magnetic energy conservation |
| 6 | Anomalous resistivity: `c_s` threshold | mlx_transport.py:426-438 | Sagdeev (1966) |
| 7 | PPM QR slice bounds | mlx_reconstruction.py:403 | Array indexing |
| 8 | HLLS docstring: corrected entropy formula | mlx_riemann.py:403-404 | Code inspection |
| 9 | WENO-Z+ p=2 citation: added Acker 2016 | mlx_reconstruction.py:248 | Acker et al. 2016 |

## Known Issues NOT Fixed (require architectural work)

| # | Issue | Why not fixable now | Paper reference |
|---|-------|--------------------|-----------------|
| 1 | Geometric source has rho attenuation factor | Removing it is correct (Stone 2008) but causes numerical instability on coarse grids | Stone et al. 2008 ApJS 178:137 |
| 2 | HLLS is really HLL+entropy, not Popovas HLLD+entropy | Implementing true Popovas requires embedding psi in Riemann solver | Popovas 2025 |
| 3 | Lee model missing axial momentum correction | Requires restructuring snowplow integrator | Lee 2014 eq. 1 |
| 4 | Lee model missing fc on circuit inductance | Requires audit of all Lp consumers | Lee 2014 eq. 2 |
| 5 | Lee radial phase is snowplow, not slug model | Requires 4-ODE system (major rewrite) | Lee 2014 eqs. 14-19 |
| 6 | CT cites Gardiner-Stone but implements Evans-Hawley | Simple CT is valid; citation needs correction | Evans & Hawley 1988 |
| 7 | Entropy tracer never resynced | Bryan 2014 recommends resync; causes energy test failure | Bryan et al. 2014 |
| 8 | Dedner damping not at M&T optimal | Current value stable; optimal destabilizes DPF | Mignone & Tzeferacos 2010 |
