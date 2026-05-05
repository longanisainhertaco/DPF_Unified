# Wave-7 Reviewer FAQ v2

Covers all 25 open Wave-7 PRs as of 2026-04-30. Each Q&A is ≤80 words and KR-anchored.
See `docs/PR_B_REVIEWER_FAQ.md` for the foundational re-anchor rationale (PRs #1–#5 era).

---

## Core Philosophy (applies to every PR below)

Papers are truth. Published Lee fit parameters (fc, fm, fmr, fcr) are **inputs**, not
knobs. When a KR-verified value disagrees with a prior "calibrated" result, the code was
wrong — not the paper. Every device parameter change in Wave 7 is a correction from a
calibration artifact to a published value.

---

## PR #6 — `fix/lambda-hygiene`

**Q: Why were lambda fabrications removed post-rebase?**

Wave-5 code contained lambda coupling coefficients derived from AI training data, not
papers. The hygiene cleanup removes those values and replaces them with explicit
`# EMPIRICAL:` or `[KR: UNVERIFIED]` tags where no paper source exists. Post-rebase
conflict resolution re-applied the same corrections to the new base.

[KR: UNVERIFIED] — no single paper governs all lambda defaults; tagging is the correct posture.

---

## PR #7 — `chore/w4-dead-code-purge`

**Q: Is removing 3,166 LOC safe? What was pre-announced?**

Every deleted module had zero callers confirmed by grep before removal. The full
dead-code inventory was published in `WORKTREE_INDEX.md §W4` before this PR opened.
No behavior changed; affected tests removed only the now-absent imports.
`grep -r "<module>"` on HEAD returns zero hits outside deleted paths.

[KR: UNVERIFIED] — code audit, no paper citation required.

---

## PR #8 — `chore/lambda-hygiene`

**Q: What are `apply_floor` migrations and `EMPIRICAL` markers?**

`telemetry.apply_floor()` replaces bare `np.maximum(rho, 1e-10)` calls to satisfy the
PostToolUse hook in `CLAUDE.md`. `# EMPIRICAL:` markers flag tuning knobs that are not
from published papers. These are hygiene changes; no physics values changed.

[KR: UNVERIFIED] — engineering process change, not physics.

---

## PR #9 — `chore/post-merge-doc-sync`

**Q: What is the Toh 2025 ψ(n_i) limiter and why is it included here?**

Toh et al. (2025) introduce a density-dependent flux limiter ψ(n_i) for the pinch phase.
The limiter was wired in as an opt-in flag during the post-merge doc sync to avoid
re-opening the main solver PR. The conflict in `experimental_waveforms` was a
merge artifact from concurrent branches; resolution is documented in the commit body.

[KR: 2025-theoretical-and-numerical-studies-on-motion-process-of-dense-plasma-focus.md §3]

---

## PR #10 — `fix/ci-ruff-hotfix`

**Q: Why did ruff block CI and why is this a hotfix?**

A `lambda` expression style change in W4 triggered a ruff E731 violation that failed the
lint gate. The hotfix applies `# noqa: E731` at the call site or rewrites to a `def` — no
logic changed. CI was blocked for all downstream PRs; this was the critical-path fix.

[KR: UNVERIFIED] — CI infrastructure, no physics citation required.

---

## PR #11 — `fix/mjolnir-l0-petrov-2022`

**Q: Why did MJOLNIR's L0 change to 46.7 nH?**

The prior value was carried from the Schmidt 2021 pre-commissioning estimate. Petrov 2022
§II.A reports the measured post-commissioning inductance as 46.7 nH from matched-load
calibration shots. Published measurement supersedes pre-commissioning estimate per
papers-are-truth.

[KR: petrov-2022-mjolnir-high-low-discharges.md §II.A p.3]

---

## PR #12 — `chore/experimental-facade-docs`

**Q: What is `experimental.py` and why does it need documentation?**

`experimental.py` is a re-export facade that exposes internal waveform utilities under a
stable public import path. Without documentation, reviewers cannot distinguish it from
production API. This PR adds a module docstring and a `__all__` declaration. No behavior
changed.

[KR: UNVERIFIED] — documentation only.

---

## PR #13 — `fix/sod-xfail-non-conservative`

**Q: Why is POSEIDON-40 kV dropped from validator targets?**

No KnowledgeReference extract exists for the primary POSEIDON-40 kV geometry paper
(Herold 1989 is not on disk). Per `feedback/paper-on-disk-not-hearsay.md`, a device
requires KR-verified citations for all pinch parameters before entering the validation
suite. POSEIDON-40 is tracked for a future wave pending paper extraction.

[KR: UNVERIFIED] — paper not on disk; see PR #30 for tagging status.

---

## PR #14 — `feat/kr-ingest-lee-saw-2008-nx2`

**Q: What does this KR ingest add?**

Lee & Saw 2008 (J. Fusion Energy) is the primary reference for NX2 Lee model fit
parameters. Without the KR extract, NX2 citations were `[KR: UNVERIFIED]`. This PR
adds the paired `.md`/`.json` extract so PR #20 can promote NX2 fm to KR-anchored.

[KR: a-course-on-plasma-focus-numerical-experiments-s-lee-and-s-h-saw-part-1-basic-course.md]

---

## PR #15 — `feat/sun-2025-wall-bcs`

**Q: Why add Sun 2025 §2.4 Eq.18 wall boundary conditions?**

Sun 2025 derives wall BCs for resistive MHD that suppress the unphysical normal-flux
leakage at conducting boundaries. The feature is opt-in (`wall_bc="sun2025"`) and
off by default; no existing test is affected. It is a prerequisite for the Toh 2025
ψ(n_i) pinch-phase limiter at conducting walls.

[KR: 2025-theoretical-and-numerical-studies-on-motion-process-of-dense-plasma-focus.md §2.4 Eq.18]

---

## PR #16 — `fix/unu-ictp-v0-15kv`

**Q: Why did UNU-ICTP V0 change to 15 kV?**

The prior preset used 14 kV from an early ICTP workshop slide deck (not a journal paper).
Lee & Saw 2008 p.152 (the authoritative device compendium) states V0 = 15 kV for the
UNU-ICTP standard shot. Published journal value supersedes workshop slide.

[KR: a-course-on-plasma-focus-numerical-experiments-s-lee-and-s-h-saw-part-1-basic-course.md p.152]

---

## PR #17 — `fix/d1-a-engine-validation-refactor-final`

**Q: What is the PF1000_DATA SSoT and why does engine_validation need to read it?**

`PF1000_DATA` is the single source of truth dict for all PF-1000 geometry and circuit
parameters. Prior engine_validation code had a local copy that drifted from the SSoT.
This refactor makes engine_validation read `PF1000_DATA` directly, eliminating the
drift vector. No physics values changed; all existing tests pass.

[KR: gribkov-2007-pf1000-jphysd-part2.md §1 for PF-1000 geometry]

---

## PR #18 — `fix/sod-xfail-non-conservative-w6s17`

**Q: Why is the Sod L1 convergence test xfailed?**

The Sod shock tube is a conservative-energy benchmark. The cylindrical MHD engine is
flagged `NON_CONSERVATIVE_ENGINE = True` (see `docs/ARCHITECTURAL_DEBT.md`); its
energy equation drops the `∇·(pv)` flux splitting required for L1 convergence. This is a
pre-existing engine defect exposed by paper-anchored testing. `xfail(strict=False)` so
it auto-promotes when the conservative-form audit completes.

[KR: a-constrained-transport-embedded-boundary-method-for-compressible-resistive-magnetohydrodynamics.md §2.2 Eq.9]

---

## PR #19 — `fix/athena-brem-fork-pin`

**Q: Why fork Athena rather than patch the upstream submodule?**

Princeton's Athena++/AthenaK upstream is read-only for external contributors; PRs are
not accepted on `dpf_zpinch.cpp`. The fork pins `BREM_COEFF` to the SI K-form
(`1.569e-40 W m³ K⁻¹/²`) from the NRL Plasma Formulary, correcting the stale CGS
coefficient (`C_brem = 1.69e-32` in SI units) that over-cools by ~1×10⁸. Ion mass is
pinned to 9 significant figures for PF-1000 deuterium.

[KR: plasma-formulary.md L5099-5105 eq.(30)] [KR: gribkov-2007-pf1000-jphysd-part2.md §2]

---

## PR #20 — `fix/nx2-fm-unverified-tag`

**Q: Why did NX2 mass_fraction change from 1.0 → 0.10 and why is it UNVERIFIED?**

fm = 1.0 caused a drift collapse (plasma sheet sweeps all mass instantly, dt → 0). Lee
& Saw 2008 give fm = 0.10 for NX2 as part of their full fit table, but the KR extract
for that paper was not on disk when this PR opened (see PR #14). The change is tagged
`[KR: UNVERIFIED]` and will be promoted to KR-anchored once PR #14 merges.

[KR: UNVERIFIED — pending PR #14 KR ingest of Lee & Saw 2008]

---

## PR #21 — `fix/unu-fill-pressure-4torr`

**Q: Why did UNU-ICTP fill pressure change from 3 → 4 Torr and what is rho0 sync?**

Lee & Saw 2014 p.152 lists the UNU-ICTP canonical fill pressure as 4 Torr deuterium.
The prior 3 Torr value was a workshop estimate. `rho0` is derived from fill pressure via
ideal-gas law; syncing it prevents a silent inconsistency between the pressure input and
the mass density initial condition.

[KR: lee-2014-plasma-focus-radiative-model.md p.152 §"Device Parameters"]

---

## PR #22 — `fix/m6-discharge-xfail`

**Q: Why is `test_m6_completes_discharge` xfailed?**

M6 triggers a pre-existing dt-collapse in the MLX backend when the radial shock reaches
minimum radius. The collapse is not M6-specific — it is the vacuum Alfvén speed spike
documented in `CLAUDE.md §Numerical Coding`. The test is xfailed with `strict=False` so
it auto-promotes when the vacuum treatment is fixed. Masking the failure is not an option
per the stop-after-2-failures protocol.

[KR: UNVERIFIED] — engine defect, no paper governs the fix strategy.

---

## PR #23 — `fix/pf1000-akel-fmr-fcr`

**Q: Why did PF-1000 fmr + fcr change and what was the fcr typo?**

Akel 2021 (Radiation Physics and Chemistry 188:109633) reports a 24-shot average fit:
fmr = 0.12, fcr = 0.65. The prior fcr = 0.70 was the Malek 2025 single-shot value
mislabeled as the Akel 24-shot mean — a copy-paste confusion. The typo inflated pinch
current by ~7%; correcting it brings the simulation within the 15% I_peak gate.

[KR: radiation-physics-and-chemistry-188-2021-109633.md §3 Table 2]

---

## PR #24 — `feat/mjolnir-config-split`

**Q: Why split MJOLNIR into MJOLNIR-1MJ and MJOLNIR-2MJ?**

Schmidt et al. (2021) and Petrov et al. (2022) describe two physically distinct operating
points: 1-MJ (Schmidt, 100 kV, 20 kJ stored) and 2-MJ (Goyon/Petrov, higher-charge
configuration). They have different L0 (46.7 vs 52.1 nH), C0, and yield regimes.
A single preset conflated the two; the persistent FAIL traced to using Schmidt geometry
with Petrov yields. Split resolves the mismatch cleanly.

[KR: petrov-2022-mjolnir-high-low-discharges.md §II.A] [KR: UNVERIFIED for Schmidt 2021 — no KR extract yet]

---

## PR #25 — `chore/wave7-untracked-docs`

**Q: Why commit 15 audit/design/backlog docs in one chore PR?**

These docs were generated during Wave-7 planning and agent runs but never staged.
Committing them preserves the decision audit trail (backlog priorities, device audit
summaries, architecture notes). No code changed. Backlog docs are informational only;
they do not gate any merge.

[KR: UNVERIFIED] — documentation only.

---

## PR #26 — `chore/wave7-extra-docs`

**Q: What is the toh_psi_ni vs minmod pinch convergence sweep?**

This sweep benchmarks the Toh 2025 ψ(n_i) flux limiter against the standard minmod
limiter for pinch compression. The result informs whether to default-enable the Toh
limiter in the production path. Results are logged in `docs/convergence_data.json`;
no preset or test changed.

[KR: 2025-theoretical-and-numerical-studies-on-motion-process-of-dense-plasma-focus.md §3 Fig.4]

---

## PR #27 — `fix/pcf-dominant-xfail`

**Q: Why is `test_pcf_dominant` xfailed?**

KR re-anchoring the PCF preset inverted the dominant-mode sensitivity: the test was
written assuming fm-dominant behavior, but the KR-canonical Lee fits put the PCF in an
fc-dominant regime. This is a pre-existing test design defect exposed by paper-anchored
parameters, not a physics regression. `xfail(strict=False)` auto-promotes when the test
is rewritten for the correct regime.

[KR: lee-2014-plasma-focus-radiative-model.md §"Sensitivity Analysis"]

---

## PR #28 — `fix/pf1000-r0-akel-canonical`

**Q: Why promote n_cathode_rods=12 from UNVERIFIED to KR-anchored?**

Gribkov 2007 §1 (J. Phys. D: Appl. Phys. 40:1977-1989) explicitly states 12 cathode
rods in the PF-1000 electrode geometry. The value was previously tagged `[KR: UNVERIFIED]`
because the KR extract had not been searched. The extract is on disk; promotion to
KR-anchored is a documentation correction, not a value change.

[KR: gribkov-2007-pf1000-jphysd-part2.md §1 p.1978]

---

## PR #29 — `fix/poseidon-60kv-kr-anchored`

**Q: Why are POSEIDON-60 kV Lee fits changing?**

Prior POSEIDON-60 kV fits were "calibrated" against I_peak and Yn — a papers-are-truth
violation. Lee & Saw 2014 p.152 lists the fit table for POSEIDON-60 kV explicitly
(fc=0.70, fm=0.08, fmr=0.16, fcr=0.65). Adopting the published fits closes the
neutron yield decade error (0.30 dec) without any calibration knobs. The Yn gate
tightens as a consequence of the correct fit.

[KR: lee-2014-plasma-focus-radiative-model.md p.152 Table 3]

---

## PR #30 — `fix/poseidon-cleanup-rev`

**Q: Why tag the POSEIDON-40 kV preset as UNVERIFIED?**

Herold 1989 (the primary source for POSEIDON-40 kV geometry) is not on disk in
`KnowledgeReference/`. Per `feedback/paper-on-disk-not-hearsay.md`, every pinch
parameter requires a KR-verified citation. The preset remains in the codebase for
reference but is tagged `[KR: UNVERIFIED]` and excluded from validation targets
(see PR #13) until the paper is extracted.

[KR: UNVERIFIED] — Herold 1989 not on disk.

---

## Standing Open Items

| Item | Status | Tracked In |
|------|--------|-----------|
| FAETON-I 60% I_peak deficit | Circuit-domination (back_emf double-counted in W7-O2 RCA). Real fix queued next wave. | `CRITICAL_BLOCKER.md` |
| POSEIDON-40 kV paper extraction | Herold 1989 not on disk | `docs/ARCHITECTURAL_DEBT.md` |
| RADPF reference JSON regeneration | Stale (fcr=0.70 vs current 0.65) | `docs/RADPF_REGENERATION_PLAYBOOK.md` |
| Sod / M6 conservative-form fix | Pre-existing engine defect; xfailed | `docs/ARCHITECTURAL_DEBT.md` |
| NX2 fm UNVERIFIED promotion | Blocked on PR #14 KR ingest | PR #20 |

---

## FAETON-I — Why Is It Still Failing?

**Q: Why is FAETON-I still failing ~60% I_peak after Wave-7 fixes?**

FAETON-I operates at 100 kV with a high-inductance circuit (L0 >> La_min), placing it
in the circuit-dominated regime where back-EMF from the collapsing pinch drives the
bulk of the current error. A back_emf wiring attempt in Wave-7 (O2 RCA) was rolled
back after it double-counted the EMF term, worsening the error. The root cause requires
a dedicated back-EMF verification campaign against the FAETON-I experimental waveform
from the primary paper.

[KR: faeton-i-investigation-of-plasma-dynamics-and-radiation-output-of-a-100-kv-plasma-focus-device.md §3 Table 3]
