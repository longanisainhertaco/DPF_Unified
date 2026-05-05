# MHD Lp Self-Consistent — Acceptance Gate

Status: PROPOSED (Wave-7 pre-merge gate). Owner: dpf-engine-architect.
Blocks: CRITICAL_BLOCKER.md Task 6d.

## Scope

Defines the pass/fail criteria for replacing the snowplow-derived `Lp(t)` with a flux-integral
`Lp(t)` computed from the MHD `B_theta(r, z, t)` field during the radial+pinch phases.
Hybrid handoff: snowplow remains authoritative during the axial phase; flux integral takes
over at radial-phase entry (current sheet detaches from outer electrode).

The Wave-6 design proposal (`docs/MHD_LP_SELF_CONSISTENT_PROPOSAL.md`) is referenced in
spirit but is not on disk at gate-write time; the gate is anchored to KR papers, not to
the proposal.

## KR Lp(t) Measurement Search — Result

Searched `KnowledgeReference/` for measured plasma-inductance time evolution.

| Paper | KR file | Lp(t) content | Anchor-grade |
|---|---|---|---|
| Lee & Saw, "A Course on Plasma Focus Numerical Experiments, Part 1" (2010) | `a-course-on-plasma-focus-numerical-experiments-s-lee-and-s-h-saw-part-1-basic-course.md` | Explicit Lp(t) formula AND computed curve (Fig. 7h) for NX2 device. Axial: `L_axial = (μ/2π)·ln(b/a)·z_f`. Radial: `L = L_axial_end + (μ/2π)·z_f·ln(b/r_p)`. dL/dt closure: `dL/dt = (μ/2π)·[ln(b/r_p)·dz_f/dt + z_f·(dr_p/dt)/r_p]`. Quoted values at peak piston speed: r_p~2.4 mm, z_f~15 mm, dr_p/dt~13.5 cm/μs, dz_f/dt~17 cm/μs → dL/dt~190 mΩ; ~2× rise during radial phase (Fig. 7h §p.13). | TIER 3 anchor (computed curve, fully derived from snowplow + dynamics, traceable per-formula). |
| Akel thesis (College of Graduate and Postdoctoral Studies) | `a-thesis-submitted-to-the-college-of-graduate-and-postdoctoral-studies-...md` | Static L_0 = 150 nH (capacitor) and L_0 = 143 nH (capacitor bank). Discusses plasma-column impedance rise during compression but does NOT publish a measured Lp(t) curve. | Static-L_0 anchor only. |
| 2025 "Theoretical and Numerical Studies on Motion Process of DPF" | `2025-theoretical-and-numerical-studies-on-motion-process-of-dense-plasma-focus.md` | Mentions inductance reduction for large devices but no Lp(t) curve. | Not anchor-grade. |
| Soto, "DPF: a versatile dense pinch for diverse applications" | `the-dense-plasma-focus-a-versatile-dense-pinch-for-diverse-applications.md` | Quotes pinch inductances: 4.4 nH (300-kA design), ~6 nH coaxial, 15-20 nH for long pinches, 43 nH peak case. Single-value snapshots, not Lp(t). | Sanity-check magnitudes. |

Conclusion: NO KR paper publishes a directly-measured Lp(t) curve from V-dot or Rogowski reconstruction. Lee Course Fig. 7h is a computed (but model-traceable) Lp(t) curve and is the strongest anchor available. Tier 3 falls back to "agree with Lee Course Fig. 7h shape and magnitude on NX2 parameters" rather than "match measured PF-1000 Lp(t)." Document the gap; do not fabricate.

## Three-Tier Acceptance Gate

### Tier 1 — Analytic Verification (fast, deterministic)

Goal: prove the flux-integral implementation is numerically correct on a problem with closed-form Lp.

Test fixture: uniform axial column of radius r_p, length z_f, carrying total current I, with
B_theta(r) = μ·I/(2π·r) for r >= r_p and B_theta(r) = μ·I·r/(2π·r_p²) for r < r_p (uniform j_z core).

Expected: `Lp_analytic = (μ/2π)·z_f·[ln(b/r_p) + 1/4]` (the +1/4 is internal inductance for uniform j_z).

Pass criteria:
- L1 error vs analytic < 1% on a 64×128 (r,z) grid
- Convergence rate >= 1.8 (second-order method) under grid refinement 64→128→256
- Vacuum region outside r=b contributes 0 to within float64 round-off

File: `tests/test_mhd_lp_tier1_analytic.py` (new). Runtime budget: < 30 s.

### Tier 2 — Hybrid Handoff Continuity + Monotonicity

Goal: prove the snowplow→flux-integral handoff at radial-phase entry is C^0 continuous and that Lp(t) has the right qualitative shape.

Test fixture: PF-1000 27 kV calibration run, full 2.5 μs window covering axial → radial → pinch → post-pinch.

Pass criteria (all four must hold):
- Continuity at handoff: |Lp_snowplow(t_handoff) − Lp_fluxint(t_handoff)| / Lp(t_handoff) < 2%. Discontinuity > 5% is a hard fail.
- Monotonic rise from axial start through pinch peak: Lp(t) is non-decreasing on the interval [t_0, t_pinch_peak] (allow numerical jitter <= 0.5%).
- Pinch-phase amplification: Lp(t_pinch_peak) / Lp(t_axial_end) ∈ [1.8, 2.5]. Lee Course quotes "more than a factor of 2 ... in a radial phase time interval about 1/10 the duration of the axial phase for the NX2" [KR: Lee Course §iv p.14]. PF-1000 ratio is not separately published; the 1.8-2.5 window brackets the NX2 ratio with ±20% generosity.
- Post-pinch decay: dLp/dt < 0 within 100 ns of peak. No persistent oscillation > 5% peak-to-peak.

File: `tests/test_mhd_lp_tier2_hybrid.py` (new). Runtime budget: < 5 min on Metal MPS.

### Tier 3 — KR-Anchored Lp(t) Shape Match

Goal: prove the simulated Lp(t) reproduces the only on-disk traceable Lp(t) curve.

Anchor: Lee Course Part 1, Figure 7h, NX2 device. Axes are time (μs, 0-2.5) vs Lp (nH). Curve is computable from the Lee Course formulas, so we are checking that our flux-integral implementation reproduces the snowplow-derived shape on the NX2 problem (NOT a redundant test of the snowplow — the flux integral and snowplow agree only if the MHD field is consistent with the snowplow assumption of a thin current sheath).

Test fixture: NX2 parameters (b=4.1 cm, a=1.9 cm, z_0 = 5 cm, V_0 = 14 kV, P_0 = 3.5 Torr D_2). Lee Course §2.4 quotes these directly.

Pass criteria:
- L1 error |Lp_sim(t) − Lp_LeeCourse_Fig7h(t)| / max(Lp_LeeCourse) < 15% over [0, 2.5 μs]
- Pinch-peak time t_peak agrees with Fig. 7h within ±50 ns
- Pinch-peak magnitude agrees within 20%
- Computed dL/dt at peak piston speed is in [150, 250] mΩ (Lee quotes ~190 mΩ for NX2 [KR: Lee Course Note 3 p.~13])

Tier 3 GAP (documented, NOT fabricated): no KR paper publishes a measured PF-1000 Lp(t). When/if such a paper enters KR, add Tier 3b targeting that specific curve. Until then, Lee Course Fig. 7h is the strongest available anchor and a snowplow-vs-flux-integral cross-check rather than a fully independent measurement.

File: `tests/test_mhd_lp_tier3_lee_nx2.py` (new). Runtime budget: < 10 min.

## Effort Estimate to Close All 3 Tiers

Assumes Wave-7 implementation lands the flux-integral Lp computation as a function on the existing Metal/Python MHD state.

| Tier | Tasks | Est. effort |
|---|---|---|
| Tier 1 | Build analytic fixture; implement uniform-current-column field initializer; convergence harness | 0.5 day |
| Tier 2 | Wire flux-integral into engine.py at handoff; add diagnostic dump of Lp(t); shape/continuity checks; calibration on PF-1000 27kV scenario | 1.5 days |
| Tier 3 | Add NX2 device config; cross-walk Lee Course Fig. 7h to a sampled time-series (digitize 20 points from the figure); shape-match harness | 1.0 day |
| Total | Tier 1 + Tier 2 + Tier 3 | 3.0 days |

Parallelization: Tier 1 and the Tier 3 fixture digitization can run in parallel (different files). Tier 2 must come after Tier 1 (handoff logic depends on a verified flux integral).

## Out of Scope for This Gate

- Energy conservation `∂(Lp·I²/2)/∂t = circuit Joule + plasma work`. This is a stronger test owned by dpf-mhd-physicist after Tier 1-3 pass; tracked separately.
- Radiation back-reaction on Lp via density depletion. Phase-W follow-up.
- 3D / non-axisymmetric Lp. AthenaK only.

## Failure Mode → Owner Routing

| Failure | Likely owner |
|---|---|
| Tier 1 fails | dpf-engine-architect (numerics bug in flux integration) |
| Tier 2 handoff discontinuity | dpf-engine-architect (handoff timing or formula mismatch) |
| Tier 2 monotonicity violation | dpf-mhd-physicist (B_theta field is non-physical) |
| Tier 3 shape mismatch | dpf-mhd-physicist (radial phase dynamics are off — investigate snowplow vs MHD divergence) |
