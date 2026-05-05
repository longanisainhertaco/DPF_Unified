# Wave-7 Reviewer FAQ: Ship Pipeline Q&A

Companion to `docs/PR_B_REVIEWER_FAQ.md`. Covers PRs #7/#8/#9/#10 and new additions
from the Wave-7 ship pipeline. Every answer is anchored to a KnowledgeReference
extract or a documented RCA; no training-data assertions.

---

## W7-Q1: Why is MJOLNIR L0 changing again?

The prior preset used L0 = 67.4 nH, the **1-MJ hard-short-test** value from
Petrov (Schmidt et al.) 2022 §II.A. MJOLNIR has been operating in **2-MJ
configuration** since commissioning; the paper explicitly states that the 2-MJ
lumped circuit parameters have not been confirmed by hard-short test and are
instead estimated from snow-plow model comparison.

The 2-MJ estimate is L0 = 46.7 nH (408 µF, 6.3 mΩ). Both numbers are on the
same page of the same paper; the 1-MJ value was applied to the 2-MJ device — a
preset mismatch, not a new physics decision.

Reference: [KR: petrov-2022-mjolnir-high-low-discharges.md §II.A L228-232]
(Schmidt et al. 2022, LLNL-JRNL-831591, _Physics of Plasmas_).

---

## W7-Q2: Why wire back_emf now?

`engine.py:1754` hard-coded `back_emf = 0.0`, discarding the motional EMF term
entirely (Wave-6 audit finding, `AUDIT_BRIEF.md` Appendix A item 1).

The trigger was the FAETON-I LOO analysis (PhD Debate 49, finding F4): with
L_p/L0 = 0.107 at the FAETON-I geometry, the snowplow inductance change during
rundown contributes ~13% of the total circuit back-voltage. Dropping it at
`back_emf = 0.0` misrepresents the circuit loading for all circuit-dominated
devices (L_p/L0 < 0.5).

The canonical form is:

    back_emf = I * dL/dt    [V]

where dL/dt is the time derivative of the plasma inductance during the axial
phase. This is the motional EMF of a moving current sheet — textbook
electrodynamics (any graduate E&M text; Griffiths §7.1, Jackson §6.3).

The KR extract for Sun 2025 §2.3 Eq. 15 gives the discrete form used in the
Lee-model circuit equation. Because Sun 2025 is not on-disk as a PDF, the
implementation is tagged `[KR: MEDIUM]` per `MERGE_PLAN.md §credibility`.

Reference: Wave-6 O6 RCA (`AUDIT_BRIEF.md:226`); [KR: Sun 2025 §2.3 Eq.15 —
MEDIUM, KR markdown only, PDF not on disk].

---

## W7-Q3: Why is the Sod shock tube test xfailed?

`test_regression_sod_density` in `tests/test_verification_consolidated.py:3253`
is marked `xfail(strict=False)` for two compounding reasons:

1. **Non-conservative pressure solver.** The Python MHD engine uses a
   non-conservative pressure formulation. At discontinuities (shocks, contact
   surfaces), non-conservative schemes do not satisfy the Rankine-Hugoniot
   conditions exactly. The Sod shock tube is a pure Riemann-problem benchmark
   designed to verify shock-speed and density-jump accuracy — precisely the
   regime where this engine fails by design.

2. **Banks 2008 convergence-rate context.** Banks & Henshaw (2008) report
   ~0.7 convergence rate for Godunov conservative finite-volume schemes on
   the Sod problem. That rate applies to schemes that solve the conservative
   form. The Python engine's non-conservative pressure path does not meet the
   preconditions for that rate guarantee; the xfail annotation documents this
   architectural boundary.

The fix is the conservative-form audit tracked in `docs/ARCHITECTURAL_DEBT.md`,
not parameter adjustment. Metal and Athena++ backends (conservative Godunov)
pass the analogous Sod tests.

Code reference: `metal_solver.py` conservative Godunov path; Python engine
non-conservative flag at `src/dpf/fluid/mhd_solver.py`.

---

## W7-Q4: Why did UNU-ICTP V0 change from 14 kV to 15 kV?

The prior preset carried V0 = 14 kV, which appears in the older ICTP Module 1
device table (p. 10 of the ICTP e-manual). The authoritative multi-device
comparison table in Lee & Saw 2014 (reproduced in the KR extract at p. 152,
L12725) lists UNU explicitly as **V0 = 15 kV**.

The table row verbatim:

    | UNU | 15 | 4 | 110 | 30 | 3.2 | 0.95 | 16 | 0.182 | ... |

This is the value Lee used when computing the UNU row of the
multi-device scaling comparison. A preset at 14 kV is stale relative to the
definitive published table.

Reference: [KR: a-course-on-plasma-focus-numerical-experiments-s-lee-and-s-h-saw-part-1-basic-course.md p.152 L12725].

---

## W7-Q5: Why was POSEIDON dropped from the validation suite?

POSEIDON (IPF Stuttgart) is tagged `_REFERENCE_ONLY` in
`src/dpf/validation/experimental_devices.py:277-279`. The primary geometry
source — Herold et al., _Nucl. Fusion_ 29:33 (1989) — is **not in the
KnowledgeReference corpus** and the PDF is not on disk under
`references/papers/`.

Per `feedback/paper-on-disk-not-hearsay.md`: a device cannot enter the
validation suite without a KR-verified citation for every pinch parameter.
POSEIDON's L0 = 35 nH is explicitly marked `# UNVERIFIED: Herold 1989 not on
disk` at `presets.py:455`. Validating against parameters whose provenance
cannot be confirmed on-disk is not validation; it is false precision.

POSEIDON-60kV (IPFS-digitized waveform) remains in the suite because its
waveform source is traceable. The 40-kV / 450-µF POSEIDON variant is gated
until the Herold 1989 paper is ingested into KnowledgeReference.

Reference: `src/dpf/validation/experimental_devices.py:277-283`; [KR: gap —
Herold 1989 Nucl. Fusion 29 not on disk].

---

## W7-Q6: Why fork the Athena C++ submodule rather than patching upstream?

The Athena++ upstream (`external/athena/src/pgen/dpf_zpinch.cpp`) carries:

    C_brem = 1.69e-32    // CGS coefficient used with SI inputs

The NRL Plasma Formulary gives 1.69e-32 in CGS units (n_e in cm⁻³, T_e in
eV). Applied to SI inputs (n_e in m⁻³, T_e in K) without conversion, this
over-cools the plasma by a factor of ~10⁸. The correct SI coefficient is
1.569e-40 W m³ K⁻¹/² — a factor 1e8 smaller — as derived in
`tests/test_bremsstrahlung_nrl.py` (1% acceptance gate).

Princeton/Athena++ upstream is a read-only institutional repository. The
coefficient error is DPF-specific (upstream does not use this bremsstrahlung
form for any other problem generator); a patch upstream would require an
Athena++ maintainer pull request with no guarantee of acceptance timeline.
The fork allows the fix to ship with this wave and is tracked in
`docs/ARCHITECTURAL_DEBT.md`.

Reference: [KR: plasma-formulary.md L5099-5105 eq.(30)];
`tests/test_bremsstrahlung_nrl.py` (derivation + 1% gate);
`tests/test_verification_consolidated.py:991-992` (xfail annotation
documenting the upstream state).

---

## W7-Q7: What does the Toh 2025 ψ(n_i) limiter do and why add it?

Standard slope limiters (minmod, MC) are designed for smooth flows with
moderate density gradients. In the post-pinch vacuum region and at the
sheath-leading-edge, ion density n_i drops toward zero. In these near-vacuum
cells, the limiter's stencil spans orders-of-magnitude density jumps; minmod
and MC both under-reconstruct, producing spurious oscillations that drive
post-pinch CFL collapse and the I(t) divergence seen in unmodified PF-1000
runs.

The Toh 2025 ψ(n_i) limiter is asymptotic-preserving: the blending function ψ
transitions from the standard limiter in dense plasma cells to a more diffusive
stencil as n_i → 0, suppressing the oscillations without introducing artificial
floors elsewhere. The implementation follows KR Eq. 31.

Worktree source: W6 `28b9c75` — `fluid/cylindrical_mhd.py`,
`tests/test_toh_limiter.py` (see `WORKTREE_INDEX.md:69`).

Gate: Toh 2025 PDF is not on disk; implementation carries `[KR: MEDIUM]` per
`MERGE_PLAN.md §credibility`. Promotion to `[KR: STRONG]` requires ingesting
the PDF into `references/papers/`.

Reference: [KR: Toh 2025 Eq.31 — MEDIUM, KR markdown only];
`WORKTREE_INDEX.md W6`; `MERGE_PLAN.md:43`.

---

## Summary Table

| # | Topic | Anchor | Status |
|---|-------|--------|--------|
| W7-Q1 | MJOLNIR L0 1-MJ vs 2-MJ | Petrov 2022 §II.A L228-232 | Ready |
| W7-Q2 | back_emf wiring | Wave-6 O6 RCA; Sun 2025 §2.3 Eq.15 | [KR: MEDIUM] |
| W7-Q3 | Sod xfailed | Non-conservative pressure; Banks 2008 | xfail(strict=False) |
| W7-Q4 | UNU-ICTP V0 14→15 kV | Lee & Saw 2014 p.152 L12725 | Ready |
| W7-Q5 | POSEIDON dropped | Herold 1989 not on disk | _REFERENCE_ONLY |
| W7-Q6 | Athena fork C_brem | NRL Formulary eq.(30); 1e8× unit error | Tracked ARCH_DEBT |
| W7-Q7 | Toh 2025 ψ(n_i) limiter | KR Eq.31; asymptotic-preserving | [KR: MEDIUM] — gate: PDF |
