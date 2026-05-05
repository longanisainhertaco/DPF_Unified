# SOD_CONVERGENCE_CHAIN.md

Decision-chain record for the Sod shock-tube convergence saga in `dpf-unified`.
Scope: `tests/test_verification_consolidated.py::TestSodDPFConvergence::test_l1_decreases_with_resolution`.

---

## 1. Original Intent

**Target:** TVD-PLM on a contact discontinuity converges at order ~0.7-0.8 (Banks et al. 2008,
"A high-resolution Godunov method for compressible multi-material flow on overlapping grids").
The test was written to guard this property: given resolutions [64, 128, 256], each resolution
doubling must reduce the L1 density error monotonically, and every per-doubling rate must be >= 0.7.

**Fixture:** `_run_sod_dpf(N)` — Cartesian Sod tube, `gamma=5/3`, PLM reconstruction, HLL Riemann
solver, `float64`, no CT. Canonical end-time `t_end = 0.2 * L / c_s` (0.2 sound-crossing times
across the unit domain).

---

## 2. Step-Cap Bug (Wave-6 W11/W12 context)

**Root cause:** `t_end` was set to the bare float `0.2` interpreted as 0.2 **seconds**. At PF-1000
conditions (`c_s ~ 8.3e4 m/s`), one sound crossing is `~1.2e-5 s`, so `t_end = 0.2 s` corresponds
to ~16,000 sound crossings. A hard step cap (`for _ in range(n_steps)`) fired long before the
solver reached `t_end`, silently exiting the loop with the state barely evolved off the initial
discontinuity.

**Effect:** The under-evolved state produced identical (or near-identical) density profiles at all
resolutions. The L1 differences between resolutions were dominated by initial-condition noise, not
by the convergence hierarchy. Result: flat L1 plateau — the test either passed trivially or failed
depending on random floating-point ordering. This masked the real engine behavior.

**Wave-6 W11 cherry-pick:** Removed the step cap and rewrote the loop as a physics-termination
loop (`while t < t_end`). Set `t_end = 0.2 / _CS` (correct physical units). Added a post-loop
guard:

```python
assert t >= 0.95 * t_end, "step-cap exit silently masking unevolved states"
```

Reference: inline comment `# sod-engine-rca, 2026-04-27` at
`tests/test_verification_consolidated.py:495-496`.

---

## 3. Post-W11 State: Non-Conservative Pressure Plateau (S17 RCA)

After W11 the fixture correctly evolved to `t_end`. The convergence study now ran against the
self-similar Sod solution. Outcome: **convergence rate < 0.7 across all resolution doublings.**

**RCA (Wave-6 S17):** Traced to `src/dpf/metal/metal_solver.py:1882`. The SSP-RK2 stage-2
combination updates pressure as a **primitive variable**:

```python
p_new = 0.5 * p_n + 0.5 * (p_1 + dt * rhs2["pressure"])
```

The `rhs2["pressure"]` term comes from `mhd_rhs_mps`, which computes the chain-rule form
`dp/dt = -v . grad(p) - gamma * p * div(v) + [heating/cooling]`. This is exact for smooth
flows, but **fails at shocks**: `p` is discontinuous, `div(v)` carries a delta function, and
their product is undefined in the weak sense. The Rankine-Hugoniot post-shock state is not
recovered.

**Conservative replacement exists but is not wired for this path:**
- `metal_solver.py:1735` — `_recover_pressure_de(rho, vel, B, E, s_rho)` recovers pressure
  from conservatively-evolved total energy via dual-energy switching.
- `_de_active` (the guard for this path) is only `True` for cylindrical coordinates +
  electrode BC (`metal_solver.py:1304-1310`). The cartesian Sod path never sets it.
- Result: `_recover_pressure_de` is called at lines 1873 and 1916 for the cylindrical path;
  the cartesian Sod path reaches line 1882 and updates `p_new` directly from the primitive RHS.

The dual-energy conservative path at `metal_solver.py:1743` (`_recover_pressure_de` docstring)
is the documented fix. It is not a missing implementation — it is a routing gap.

See also: `docs/PHASE1_ASSESSMENT_REPORT.md §C4` and `docs/METAL_V2_DOD.md §4.6` for prior
audit records of the same root cause in the Python engine path.

---

## 4. Wave-7 O10: xfail with Engine-Defect Reason

**Decision (2026-04-30):** Mark `test_l1_decreases_with_resolution` as `xfail(strict=False)`.
This preserves the test (it keeps running and will un-fail automatically when the defect is
fixed) without blocking the CI pipeline on a known engine limitation.

**xfail reason string** (verbatim from
`tests/test_verification_consolidated.py:2379-2390`):

> Non-conservative pressure evolution at shocks (metal_solver.py:1882) violates
> Rankine-Hugoniot. Banks 2008 ~0.7 rate is for conservative Godunov scheme; the
> primitive dp/dt + gamma*p*div(v) form used here cannot achieve TVD-PLM convergence on
> shocks. Engine defect tracked as ARCHITECTURAL_DEBT D-PRESSURE-RECOVERY; will resolve when
> conservative total-energy evolution is wired (use_conservative_energy=True path exists at
> metal_solver.py:1743 _recover_pressure_de but is not wired for cartesian Sod path).

`strict=False` means: if the test starts passing (i.e. the defect is resolved), the suite
reports a bonus pass, not a failure. The test becomes a free upgrade detector.

---

## 5. What Needs to Happen to Un-xfail

Three ordered steps, all in `src/dpf/metal/metal_solver.py`:

1. **Wire `_de_active` for the cartesian Sod path.** Either expose a `use_conservative_energy`
   constructor flag (the path referenced in the xfail string) or activate dual-energy
   unconditionally outside cylindrical-only guards. The infrastructure at lines 1302-1310 and
   1865-1916 is already correct for cylindrical; reuse it.

2. **Thread `E` and `s_rho` through `_run_sod_dpf`.** The fixture passes no initial entropy
   tracer. `enable_dual_energy(rho_n, p_n)` initializes `s_rho = p / rho^gamma`; this call
   must happen before the time loop. See `metal_solver.py:1301-1303`.

3. **Remove the xfail and confirm rate >= 0.7 across [64, 128, 256].** Once conservative energy
   is wired and `_recover_pressure_de` fires at line 1916 instead of the primitive update at
   line 1882, the Banks 2008 rate is structurally reachable. Run `pytest -m slow
   tests/test_verification_consolidated.py::TestSodDPFConvergence` and verify.

Optionally audit `TestBrioWuDPFConvergence` and `TestMagnetizedNoh` for the same primitive-
pressure pattern — they pass stability gates but have not been confirmed against published
convergence orders (`docs/ARCHITECTURAL_DEBT.md §D-PRESSURE-RECOVERY` recommends this
follow-up).

---

## 6. Cross-References

| Item | Location |
|------|----------|
| Engine defect (primitive pressure) | `src/dpf/metal/metal_solver.py:1882` |
| Conservative recovery (exists, unwired) | `src/dpf/metal/metal_solver.py:1735-1743, 1916` |
| Dual-energy activation guard | `src/dpf/metal/metal_solver.py:1302-1310` |
| xfail mark (O10) | `tests/test_verification_consolidated.py:2379-2390` |
| Step-cap fix + t_end guard (W11) | `tests/test_verification_consolidated.py:489-528` |
| sod-engine-rca inline citation (S17) | `tests/test_verification_consolidated.py:495-496` |
| PHASE1 audit record (C4) | `docs/PHASE1_ASSESSMENT_REPORT.md:88-94` |
| METAL_V2_DOD audit record (4.6) | `docs/METAL_V2_DOD.md:466-481` |
| Banks 2008 reference | `tests/test_verification_consolidated.py:496-498, 2383` |

---

*Authored 2026-04-30. Read-only record — no code changes in this file.*
