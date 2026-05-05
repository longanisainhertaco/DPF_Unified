# Task 7: Higher Resolution Convergence Study — Acceptance Scope

Status: UNBLOCKED (Task 6 sheath physics improvements complete)
Date scoped: 2026-04-30

---

## What Kind of Convergence

Task 7 is a **grid-resolution sweep** (spatial convergence of MHD fields), NOT a
physics-resolution sweep (more sheath model fidelity). The distinction matters:

- The existing `docs/CONVERGENCE_STUDY.md` ran `--mode lee` (circuit + snowplow ODE only).
  Lee mode is grid-independent by design — the snowplow computes sheath position
  analytically. Observed order = 0.00, GCI = 100%. That result is correct and useless
  as an MHD convergence test.

- Task 7 requires `--mode mhd` runs: the MLX axisymmetric MHD solver on increasing
  Nr x Nz grids, with the same KR-canonical inputs (Akel 2021, Malek 2025 Lee fits).
  Grid-sensitive quantities are the density profile, B_theta structure, sheath
  thickness, and pinch column radius — not I_peak (which is circuit-driven).

Resolution ladder: **32x64 -> 64x128 -> 128x256 -> 256x512**

Baseline is 32x64 (used by Task 3 acceptance tests and Task 4 ShinkaEvolve).
Target fine grid is 256x512 (8x refinement in each axis, 64x total cell count increase).
512x1024 is excluded from Wave-7 scope — runtime is prohibitive without AMR (see AMR
design docs). Flag it as Wave-8 stretch goal if 256x512 wall time < 4 hours.

---

## KR Paper Search Result

Search: `grep -rn "convergence|grid resolution|256|1024" KnowledgeReference/`

Findings:
- `usimindepth-release-301-tech-x-corporation.md`: USim DPF input decks use 256x256
  structured grids as production-level resolution. No convergence table provided.
- `2025-theoretical-and-numerical-studies-on-motion-process-of-dense-plasma-focus.md`
  (Sun 2025): MHD simulation of DPF sheath and pinch; no grid convergence study
  reported. Validates current waveform and trajectory against experiment, not spatial
  field convergence.
- No KR paper provides a DPF-specific grid convergence table (L1 error vs Nr).
- Traffic-model convergence entries are irrelevant.

**KR convergence-paper gap: CONFIRMED.** No on-disk paper benchmarks DPF MHD
spatial convergence rates. Tier 3 (KR-anchored experimental waveform) must document
this gap explicitly (see below).

---

## 3-Tier Acceptance Spec

### Tier 1 — Monotonic I_peak Convergence (required for PASS)

Metric: I_peak (MA) computed from the MHD run's circuit state at t_peak.
Reference: Scholz 2006 PF-1000 27 kV = 1.87 MA [KR: radiation-physics-and-chemistry-188-2021-109633.md].

Pass criteria:
- I_peak error (% vs Scholz 1.87 MA) does NOT increase monotonically with resolution.
  Acceptable: flat or non-monotonic (I_peak is primarily circuit-driven, so resolution
  sensitivity should be small, < 2% swing across the ladder).
- I_peak at 256x512 remains within 15% of Scholz (current HEAD is +7.6% at 32x64;
  regression fence from `0958947`).
- Compute GCI between 128x256 and 256x512 using Roache (1998) safety factor Fs = 1.25.
  GCI < 20% on I_peak is acceptable (I_peak is not the primary spatially-sensitive
  quantity).

Failure: I_peak drifts > 5% between 32x64 and 256x512, indicating the MHD-circuit
coupling is resolution-sensitive (symptom: snowplow Lp mismatch vs MHD-derived Lp).

### Tier 2 — Pinch-Column Resolution Convergence (required for PASS)

Metric: r_pinch_min (m) — minimum radial extent of the pinch column at t_peak.
Computed from: radial density profile at z = z_anode, threshold at rho > 10 * rho_fill.

Pass criteria:
- r_pinch_min stabilizes (< 10% change) between 128x256 and 256x512.
- Sheath thickness (in cells) >= 4 at 32x64; >= 8 at 64x128 (sheath is physically
  resolved, not smeared over 1-2 cells).
- rho_max at pinch shows monotonically increasing trend with resolution (coarser grids
  smear the pinch; finer grids resolve higher peak density). No saturation requirement
  — documenting the trend is sufficient for Wave-7.

Failure: r_pinch_min does not stabilize by 128x256 (pinch column still resolution-
dependent) — indicates sheath cells < 4 even at 128x256 and AMR is required to proceed.
This is a go/no-go for Wave-8 AMR activation, NOT a Wave-7 blocker.

Secondary metric: B_theta_max at pinch (T). Same monotonic-increase expectation.
Record but not a pass/fail gate.

### Tier 3 — KR-Anchored Experimental Waveform Reproduction (stretch goal, not required for Wave-7 PASS)

Target: reproduce Lp(t) or hard X-ray (HXR) pulse waveform from a published PF-1000
27 kV experimental shot.

KR paper gap status: **NO on-disk paper provides digitized Lp(t) or HXR waveform
data for PF-1000 27 kV.** Sun 2025 shows trajectory and current waveform plots but
does not provide tabulated spatial field data. Akel 2021 provides I(t) only.

Action required before Tier 3 is testable:
- Anthony must identify and add to KnowledgeReference/ a paper with digitized
  PF-1000 spatial data (candidate: Scholz et al. 2004 IAEA, or Kubes et al. PF-1000
  interferometry papers).
- Until that paper is on disk with digitized data in `tests/reference_data/`, Tier 3
  is formally UNVERIFIABLE under papers-are-truth policy.

Tier 3 is documented here to define what "done" looks like at full validation maturity,
not to gate Wave-7 completion.

---

## Effort Estimate

| Step | Effort | Notes |
|------|--------|-------|
| Fix `run_convergence_study.py` to use `--mode mhd` | 1h | Script exists; needs mode flag and output schema update |
| Run 4-resolution ladder (32x64 to 256x512) | 4-8h wall | 256x512 is the long pole; overnight job |
| Compute GCI, r_pinch_min, rho_max per run | 2h | Analysis script; Roache 1998 method already in codebase |
| Write results to `docs/CONVERGENCE_STUDY.md` (update) | 0.5h | Append MHD section; keep Lee-mode section |
| Tier 1+2 pass/fail verdict | 0.5h | |
| **Total** | **8-12h** | Dominated by 256x512 wall time |

Tier 3 prerequisite (paper acquisition) is Anthony's decision, not counted here.

---

## What This Is Not

- Not a physics-model improvement. Task 7 measures the existing solver at higher
  resolution. Physics changes (Hall MHD, two-fluid, radiation transport) are Wave-8+.
- Not a sheath-model refinement. The sheath physics improvements from Task 6 are
  inputs to this study, not outputs.
- Not a calibration exercise. Lee fits (Malek 2025) and device geometry (Akel 2021)
  are frozen inputs. If higher resolution degrades I_peak agreement, the physics
  is wrong — not the parameters.

---

## References

- Akel et al. (2021), Radiat. Phys. Chem. 188:109633 — PF-1000 device params
  [KR: radiation-physics-and-chemistry-188-2021-109633.md]
- Malek et al. (2025), Plasma Phys. Tech. 12(1):9 — Lee fits fc=0.7, fm=0.13,
  fmr=0.35, fcr=0.65 [KR: plasma-physics-and-technology-1211-9-2025.md §3]
- Roache, P.J. (1998), Verification and Validation in Computational Science and
  Engineering, Hermosa Publishers — GCI method [TRAINING, standard reference]
- Sun et al. (2025), Acta Physica Sinica 74:115201 — DPF MHD waveform/trajectory
  [KR: 2025-theoretical-and-numerical-studies-on-motion-process-of-dense-plasma-focus.md]
  No grid convergence table. Cited to confirm gap.
- USim In-Depth 3.0.1 (Tech-X Corp) — 256x256 production grid for DPF
  [KR: usimindepth-release-301-tech-x-corporation.md] No convergence table.
