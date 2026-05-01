# ARCHITECTURAL_DEBT.md

Architectural debt ledger for `dpf-unified`. Each item documents a structural problem
surfaced by an audit, the production impact, the recommended refactor, and an effort
estimate. This file is for future-session triage; it is **not** a sprint plan and it
does **not** authorize work. Treat it as a queue ordered by ROI, not by listing order.

- Repo root: `/Users/anthonyzamora/dpf-unified`
- Authored: 2026-04-27 by `dpf-engine-architect`
- Provenance for each item: file paths and line numbers cited inline. Verified against
  the working tree (excluding `.worktrees/` and `.claude/worktrees/` mirrors).

---

## D1. Three Competing PF-1000 Definition Paths

### Description
PF-1000 device parameters are independently defined in three places, with overlapping
but non-identical fields:

- `src/dpf/presets.py:65` — `pf1000` preset (also `pf1000_akel:117`, `pf1000_20kv:168`) [POST-PR-B: line 174→168]
- `src/dpf/validation/experimental_devices.py:31` — `PF1000_DATA` (also
  `PF1000_16KV_DATA:182`, `PF1000_GRIBKOV_DATA:227`, `PF1000_20KV_DATA:371`) [POST-PR-B: 175→182, 220→227, 363→371]
- `src/dpf/validation/engine_validation.py:83` — `run_rlc_snowplow_pf1000()` with
  inline circuit parameters

Each path was authored for a different consumer (CLI presets, calibration suite,
acceptance test harness) and silently diverged. Lee fit parameters (`fc`, `fm`, `fmr`,
`fcr`), R0, L0, anode length, and waveform pointers are not guaranteed to match across
the three paths.

### Impact
- **High.** Calibration tests (`tests/test_jax_lee_model.py`,
  `tests/test_mlx_pf1000.py`) and acceptance tests (`engine_validation.py`) can hit
  three different parameter sets while all claiming to validate "PF-1000 at 27 kV / 3.5
  Torr." Anthony's Apr 18 regression to 11.5% I_peak error vs the Apr 10 baseline of
  2.8% is consistent with a divergent-source-of-truth bug profile.
- Violates "Published parameters are inputs, not knobs" (CLAUDE.md). When three call
  sites quote three slightly different numbers for the same physical device, the next
  shot at calibration is a coin flip.

### Recommended Refactor
1. Promote `experimental_devices.py:PF1000_DATA` to the **single source of truth** —
   it already carries waveform_t/waveform_I and a `lee_reference` field, which is the
   strongest schema of the three.
2. Rewrite `presets.py:pf1000*` to construct `SimulationConfig` **from** `PF1000_DATA`
   rather than redeclaring fields.
3. Rewrite `engine_validation.run_rlc_snowplow_pf1000()` to accept an
   `ExperimentalDevice` dataclass argument (default `PF1000_DATA`) instead of inlining
   numbers.
4. Add an integration test that asserts numerical equality of all key fields across
   the three call sites (or, better, asserts they call the same constructor).

### Effort Estimate
**M (1-2 days).** Mechanical refactor; the risk is in finding every implicit consumer.
Recommend running `tests/` end-to-end before and after with a fixed seed and
asserting bit-for-bit identical outputs on the PF-1000 paths.

---

## D2. `compute_current_density` HL/SI Unit Ambiguity

### Description
`src/dpf/metal/mlx_sources.py` exposes three related functions:

- `compute_curl_B_squared_HL` (line 231) — returns `|curl B|^2` in
  Heaviside-Lorentz units, no `mu_0`
- `compute_current_density_si` (line 335) — wraps the HL curl, divides by `mu_0` (or
  `mu_0^2` if B is already SI), returns `|J|^2` in SI
- `compute_current_density_components` (line 381) — returns `(Jr, Jz, Jt)` in SI

The docstring at line 356 explicitly flags the ambiguity: *"this matches the SI-typed
`compute_resistivity` API but contradicts the historical comment chain that used
`* mu_0`."* Line 707 of `mlx_solver.py` now uses `compute_current_density_si` directly
(the HL→SI TODO was partially resolved; no bare replacement target comment remains). [POST-PR-B: 706→707, TODO resolved] This means the codebase currently has two implementations
of "current density" with different unit conventions, and a third name
(`compute_current_density`) that matches neither but appears in caller docstrings and
test names.

Callers (verified):
- `src/dpf/metal/mlx_solver.py:707` (uses HL form, comments admit it should be SI) [POST-PR-B: 706→707]
- `src/dpf/metal/mlx_sources.py:463` (uses `_components` form)
- `tests/test_mlx_sources.py:160`
- `mlx_solver.py:706` references the eventual replacement target

### Impact
- **High and physics-relevant.** A factor-of-`mu_0` (1.257e-6 SI) error in J^2 propagates
  directly into Joule heating `eta * J^2`, anomalous resistivity threshold
  `J/(n*e*v_th)`, and any Lorentz-force diagnostic. This is the kind of bug that produces
  plausible-looking simulations that disagree with experiment by a constant factor and
  sends calibration agents chasing knobs.
- The "fix-anom-mu0-retry agent owns the final sign" comment at line 357 means the
  ownership of the conversion convention is ambiguous between modules.

### Recommended Refactor
1. **Pick a convention and document it once.** The codebase already standardized on SI
   in `compute_resistivity`; standardize J-related helpers on SI too.
2. Rename `compute_curl_B_squared_HL` to make the HL nature unmistakable (it already is,
   but the existence of `compute_current_density_*` siblings tempts misuse). Keep it
   private (`_compute_curl_B_squared_HL`) and force callers through `*_si`.
3. Delete or alias the bare name `compute_current_density` so it cannot be imported. All
   four call sites must explicitly say HL or SI.
4. Add a unit test that constructs a known B field, computes J via both helpers, and
   asserts the SI/HL ratio equals `mu_0` to float32 tolerance.

### Effort Estimate
**S (4-6 hours).** Three call sites + one test file + two docstrings. Risk is low if
the unit test is added first.

---

## D3. `mlx_engine.py:294` Hardcoded `back_emf=0.0`

### Description
`src/dpf/metal/mlx_engine.py:300` calls
`circuit.step(Lp=Lp_sp, dLp_dt=dLp_dt_sp, R_plasma=R_plasma, back_emf=0.0, dt=dt)`.

The lines immediately above (283-291) document why: the Auluck (2021) Poynting coupling
`V = -(1/I) * integral(J·E d^3r)` is implemented in `compute_voltage_poynting()` but
**not wired** into the circuit. The MHD B-field in the buffer zone propagates too
slowly relative to the vacuum solution, producing a wrong-sign `J·E` (energy INTO the
plasma) that would corrupt the circuit. The comment lists the two known fixes:
(a) faster B propagation via lower density floor or resistive diffusion, (b)
initializing B correctly at t=0 when I>0 (Beresnyak's approach).

The mission brief flags this as a **Known Bug #3 unblock dependency**.

[POST-PR-B: line confirmed at 300 (original note "294" was stale; 300 is current HEAD)]

### Impact
- **High but contained.** As long as `back_emf=0.0`, the circuit sees only the
  `dLp/dt * I + R_plasma * I` voltage drop and is blind to MHD-self-consistent
  back-EMF. Energy conservation across circuit + plasma cannot be closed; circuit
  energy accounting is correct only as long as the plasma is approximated as a moving
  inductor. Any physics that depends on self-consistent V (Townsend breakdown,
  voltage spikes at pinch, sheath dynamics) is gated on this fix.
- Blocks Known Bug #3 per the mission brief.

### Recommended Refactor
1. **Land Beresnyak-style B initialization first.** Initialize `B_theta(r, t=0) =
   mu_0 * I_0 / (2*pi*r)` so the plasma carries the bank current from t=0 and the
   MHD curl returns the correct sign in the buffer zone.
2. Once (1) is verified by an independent test (compute J·E on a static initial
   condition, assert it integrates to ~0), wire `compute_voltage_poynting()` to feed
   `back_emf` to `circuit.step()`.
3. Add an energy-conservation test: `dE_circuit/dt + dE_plasma/dt + radiation_loss
   ≈ 0` to ~1e-3 over a full pinch cycle.
4. Until (1) and (2) are done, **do not** flip `back_emf` from 0.0 — wrong-sign
   feedback will produce a runaway oscillation that masks the real bug.

### Effort Estimate
**L (3-5 days).** This is not a refactor; it is the implementation of the two physics
fixes the existing comment block already lays out. Effort is dominated by the
energy-conservation acceptance test and the inevitable second-order corrections (CFL,
buffer-zone width).

---

## D4. `_safe_inv_r` 1/r Fudge Pattern (Multiple Implementations)

### Description
`src/dpf/geometry/cylindrical.py:48` defines `_safe_inv_r(r)` with a Numba `@njit`
decorator. At `r < 1e-30`, it falls back to `1/r[i+1]` (one cell out). The function is
called once internally (line 101: `self.inv_r = _safe_inv_r(self.r)`).

Independently, `src/dpf/metal/mlx_sources.py:299` open-codes the same operation:
`inv_r = 1.0 / mx.maximum(r, 1e-30)`. This is a **different formula**: it clamps the
denominator rather than substituting from the next cell.

The mission brief lists six consumers; verified consumers in non-worktree code:
- `src/dpf/geometry/cylindrical.py:101` (calls `_safe_inv_r`)
- `src/dpf/metal/mlx_sources.py:299` (open-coded, different formula); also `:408` [POST-PR-B: second open-coded site added at 408]
- Five additional sites flagged in `app_mhd.py`, `app_plots.py`,
  `tests/test_neutron_yield.py`, `src/dpf/athenak_wrapper/athenak_io.py`,
  `src/dpf/validation/sedov_exact.py`, `src/dpf/validation/dynamic_zpinch.py` — each
  uses some variant of `1/r` with axis protection (formulas not yet reconciled).

### Impact
- **Medium.** `_safe_inv_r` returns `1/r[i+1]` at the axis cell, which is roughly
  `1/(1.5 * dr)` for cell-centered grids. The MLX version returns `1/1e-30 = 1e30`,
  effectively a NaN-firewall but not a physically motivated regularization. These two
  formulas agree when `r[0] >= dr/2` (true for cell-centered grids) but disagree on
  the axis-touching case and on convergence behavior as `dr -> 0`.
- The real concern is not the formulas (each is defensible) but that **six consumers
  are each picking their own**, which means convergence-study results
  (`tests/test_convergence_study.py`) may show different orders depending on which
  module a given quantity flows through.

### Recommended Refactor
1. Promote `_safe_inv_r` to `src/dpf/geometry/axis_bc.py` as the single source of
   truth. Document the L'Hopital justification in the docstring (already partially
   present at line 51).
2. Provide both NumPy and MLX implementations with **identical** formulas so the
   Python-vs-Metal parity tests cannot diverge at the axis.
3. Replace the open-coded `1.0 / mx.maximum(r, 1e-30)` in `mlx_sources.py:299` with
   the canonical helper.
4. Audit the five additional consumers and either replace or document a per-call
   override with a `# AXIS-BC:` tag explaining why this site needs a different formula.
5. Add an axis-cell convergence test: refine `dr` and assert the value at `r[0]`
   converges to the L'Hopital limit, not to a clamped constant.

### Effort Estimate
**S (1 day).** Six call sites, mostly mechanical. The hard part is deciding the
canonical formula and persuading the test suite that the change is benign.

---

## D5. `engine/core.py` Fractured Backend Dispatch

### Description
`src/dpf/engine/core.py` has 13 verified `self.backend ==` (or `in`) sites at lines:
112, 114, 124, 129, 139, 182, 443, 469, 603, 610, 667, 1002, 1096. Each is an `if/elif`
chain branching on `{"python", "athena", "athenak", "metal", "mlx", "hybrid"}`. The
mission brief calls this "9 fractured backend dispatch sites with mirror-image bugs";
the actual count is higher and several of the conditions are subtly inconsistent:

- Line 667: `self.backend not in ("mlx",)` — only mlx is excluded from electrode BC
- Line 1002: `self.backend != "metal"` — only metal excluded from Nernst
- Line 610: `self.backend == "python"` — initial-step diagnostic only on Python

Each backend-conditional behaves differently from the others, and there is no central
contract document for which backends support which physics knobs. The
`engine_tier(self.backend)` call on line 491 is the closest thing to a registry, but
it is read once and not used to gate features.

### Impact
- **High.** This is the canonical "backend fallback cascade fails" failure mode from
  the agent's `When Things Go Wrong` table. A new backend (e.g., a CUDA port) requires
  audit-and-edit at all 13 sites, and there is no single test that asserts feature
  parity across the matrix.
- Mirror-image bugs are guaranteed: when one backend learns a new feature, the
  conditional at site N gets updated and site M is missed.

### Recommended Refactor
1. Define a `BackendCapabilities` dataclass with explicit boolean fields:
   `supports_electrode_bc`, `supports_nernst`, `supports_anisotropic_conduction`,
   `requires_geom_attachment`, `provides_hybrid_handoff`, etc.
2. Each backend module declares its own `CAPABILITIES = BackendCapabilities(...)` at
   import time.
3. `core.py` dispatches on `self.fluid.capabilities.supports_X` instead of
   `self.backend == "X"`. The 13 string comparisons collapse to ~5 capability checks.
4. Add a parametrized parity test that, for each backend, asserts the declared
   capabilities match the actual behavior on a 1D Sod tube, a Brio-Wu, and a static
   diffusion test.
5. The `_resolve_backend()` chain at line 112 stays as the entry point but its output
   becomes a `Backend` enum, not a string.

### Effort Estimate
**L (3-4 days).** This is the highest-leverage refactor in this ledger because every
future backend change benefits from it. Risk is medium — the test suite must catch
silent feature regressions during the rewrite. Recommend doing this **after** D1
(presets) and D2 (units) so the tests are reliable first.

---

## D6. Self-Baselines Calibrated to Past Code State

### Description
`tests/baselines/` contains five JSON files used as "truth" data:

- `pf1000_peak_current.json` (2 lines)
- `sod_density_profile.json` (129 lines)
- `saha_curve.json` (11 lines)
- `spitzer_resistivity.json` (11 lines)
- `braginskii_coefficients.json` (6 lines)

Consumers (verified):
- `tests/test_anomalous_resistivity_validation.py`
- `tests/test_mlx_boris_leermore_fluxlim.py`
- `tests/test_physics.py`
- `tests/test_verification_consolidated.py`
- `tests/test_validation_ci.py`
- `tests/test_circuit_consolidated.py`
- `src/dpf/turbulence/subgrid.py`
- `src/dpf/metal/mlx_transport.py`
- `src/dpf/engine/physics_operators.py`
- `src/dpf/engine/core.py`
- `src/dpf/collision/spitzer.py`

The CLAUDE.md rule states: *"NEVER generate expected test values from your own
derivation. Truth data comes from RADPF reference traces in `tests/reference_data/`."*
However, the `baselines/` files are **not** RADPF traces; their provenance is unstated
and they look like outputs captured from a past Python implementation of the same code
they now validate. This is the "self-baseline" antipattern: the test passes because the
code agrees with itself, not because it agrees with physics.

`pf1000_peak_current.json` is especially suspect — a 2-line file claiming to define
peak current "truth" for a 1 MJ device.

### Impact
- **High and silent.** A regression that breaks both the code and the baseline in the
  same way produces a green test. The 2.8% -> 11.5% I_peak regression noted in
  MEMORY.md may have slipped past CI for this reason. This is the failure mode the
  CLAUDE.md rule was written to prevent.

### Recommended Refactor
1. **Audit each baseline file.** For each entry, document the source — paper figure,
   RADPF trace, analytic formula. Files whose source cannot be traced get **deleted**,
   not regenerated.
2. Move RADPF-derived traces into `tests/reference_data/` per CLAUDE.md.
3. For analytic baselines (Sod tube, Saha, Spitzer), keep the JSON but replace each
   value with an inline computation in the test that calls the published formula
   directly. The test then asserts agreement between the implementation and the formula,
   not between the implementation and a serialized version of itself.
4. `pf1000_peak_current.json` should either be replaced by the `PF1000_DATA.waveform_I`
   peak from `experimental_devices.py` (D1) or deleted.
5. Add a CI check that fails if any test imports from `baselines/` without a sibling
   comment of the form `# BASELINE_SOURCE: <citation>`.

### Effort Estimate
**M (2 days).** Audit work is irreducible. The Sod baseline (129 lines) is the largest
single artifact and may justify keeping if its source is the published Sod 1978
paper.

---

## D7. Lee 2014 Misattribution Pattern

### Description
The mission brief reports **~26 sites identified and fixed** for the
"Lee 2014 misattribution" pattern — a recurring bug where formulas, parameters, or
fits are credited to "Lee 2014" when the actual source is Lee & Saw 2010, Lee 1988,
the AAAPT device survey, or a different publication entirely.

Verified remaining citations to "Lee 2014" in the active source tree (excluding
`docs/`, `archive_reference_OLD/`, and worktree mirrors):

- `src/dpf/presets.py:267, 277, 525` — three sites [POST-PR-B: 273→267, 283→277, 531→525]
- `src/dpf/metal/mlx_snowplow.py:92, 113, 118, 122, 134, 170, 191, 230, 240` — nine
  sites (line numbers confirmed unchanged)
- `src/dpf/fluid/snowplow.py:192, 453, 491` — three sites (confirmed post-PR-B)
- `src/dpf/validation/experimental_devices.py:548, 552` — two sites [POST-PR-B: 541→548, 551→552]

That's 17 remaining occurrences in `src/`, plus another ~14 across `app_*.py` and
`docs/*.md`. Without paper-on-disk verification (per CLAUDE.md "Physics Integrity
Rules"), each citation is suspect — it is exactly the failure mode the rule exists to
prevent.

### Impact
- **Medium and corrosive.** Misattributed citations don't break tests, but they
  contaminate the audit trail. When a future calibration disagrees with a paper, the
  diagnostic path is "what does Lee 2014 actually say?" — and if the docstring is
  wrong about which paper, the rabbit hole is deeper than necessary.
- This is the kind of debt that a single contaminated AI session created and that
  multiplies across the codebase via copy-paste. The 26-fix burst suggests the
  pattern is well-understood; the question is whether it's been fully extinguished.

### Recommended Refactor
1. Run a campaign across the 17 remaining `src/` sites: for each, open the actual
   paper PDF (must be on disk under `references/papers/`), confirm the equation
   number, and either fix the citation or fix the formula. Per CLAUDE.md, do **not**
   verify from training data.
2. Many of these are likely "Lee model" generic references that should cite Lee &
   Saw 2010 (the canonical model paper) or Lee 1988 (the original equations). The
   "Lee 2014 Review" is a *review* paper and should only be cited when reviewing,
   not when implementing.
3. Add a pre-commit hook (the
   `cortana-formula` system has the necessary infrastructure) that flags new "Lee
   2014" strings and requires a paper-on-disk citation tag.
4. Update `memory/dpf-papers/` with a note: *"Lee 2014 is a review; cite Lee & Saw
   2010 or Lee 1988 for primary equations."*

### Effort Estimate
**S (1 day).** 17 sites, mostly docstrings. The work is not the editing — it's the
discipline of opening each PDF before each edit. Time-box at 1 day; if it runs over,
file a follow-up debt item rather than skipping the verification.

---

## Priority Ordering (Recommended)

This is the order I would tackle these if forced to rank — not authorization to
proceed, just an opinion:

1. **D2 (HL/SI ambiguity)** — small, high-impact, blocks confidence in any
   J^2-dependent diagnostic. Do this first because D5 and D6 both lean on
   J-related test results.
2. **D6 (self-baselines)** — until baselines are trustworthy, the tests can't
   guard the rest of the work. This is the foundation.
3. **D1 (PF-1000 fragmentation)** — explains the I_peak regression. Mid-effort,
   high payoff.
4. **D4 (`_safe_inv_r` fudge)** — small, removes a class of axis-cell parity bugs.
5. **D7 (Lee 2014 cleanup)** — small, but only valuable if D2/D6 are done first
   so the new citations land on a stable base.
6. **D5 (backend dispatch)** — high effort, high leverage. Do **after** the test
   suite is trustworthy (D6) so the rewrite has a real safety net.
7. **D3 (`back_emf=0.0`)** — physics work, not a refactor. Schedule it as the
   Known Bug #3 unblock task it already is, not as architectural debt.

---

## Out of Scope for This Ledger

The following surfaced during the audit but are not architectural debt:

- The 11.5% vs 2.8% I_peak regression is a **bug**, not debt. Likely caused by D1
  (multiple PF-1000 paths) or D6 (drifting baselines), but the regression itself
  needs a rerun + bisect, not a refactor.
- The `back_emf=0.0` hardcode (D3) is technically a known physics bug, listed here
  only because the mission brief asked for it. Real ownership is `dpf-mhd-physicist`,
  not `dpf-engine-architect`.
- ARES OS / ShinkaEvolve integration is healthy per the most recent CLAUDE.md and is
  not flagged in this audit.
