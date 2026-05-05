# D2 — `experimental_devices.py` Refactor Proposal

**Status:** DESIGN ONLY — do not implement
**Date:** 2026-04-30
**Author:** dpf-engine-architect (Cortana)
**Trigger:** Wave-5 hot-file analysis flagged 9 commits in 14 days (KR re-anchor cascade churn)
**Constraint:** papers-are-truth — refactor must preserve all KR citations and numeric values byte-identically.
**Backward-compat:** every existing symbol importable from `dpf.validation.experimental_devices` must remain importable from the same path.

---

## 1. Current File Analysis

| Metric | Value |
|---|---|
| Path | `src/dpf/validation/experimental_devices.py` |
| LOC | 616 |
| `ExperimentalDevice(...)` instances | 11 (one is a duplicate `POSEIDON` — see below) |
| Active devices in `DEVICES` registry | 9 |
| `_REFERENCE_ONLY` registry | 2 (POSEIDON 40 kV, AECS-PF2) |
| Helper functions | 1 (`get_devices_by_provenance`) |
| External importers | 4 files: `experimental.py`, `experimental_comparison.py`, `experimental_diagnostics.py`, `tests/test_validation_ci.py` |
| Symbols re-exported via `experimental.py` | 13 (every `*_DATA`, `DEVICES`, `_REFERENCE_ONLY`, helper) |

**Structural notes:**
- File is a flat sequence of `ExperimentalDevice(...)` literal blocks. No classes, no inheritance, no dispatch logic.
- Each device block is 30-90 lines; `measurement_notes` strings dominate (10-30 lines each, multi-line concatenated string literals).
- KR citations are inline in three forms: trailing comments (`# [KR: ...]`), `reference=` field, and `measurement_notes=` body. Same KR file is cited up to 7 times within a single device block (UNU-ICTP, POSEIDON-60kV).
- **No physics constants live in this file.** Lee-fit coefficients (`lee_fc`, `lee_fm`, `lee_fmr`, `lee_fcr`) are device-specific published values, not global constants. Resistance/inductance/geometry are device-specific. The "coefficient cascade" churn is *device data* shifting, not physics-constant churn.
- Waveform arrays are already extracted to `experimental_waveforms.py` (good prior factoring).

**Why it's a hot file:** every KR re-anchor that touches *any single device* requires editing this monolith. The Wave-5 commits are not coupled changes — they are independent edits to PF-1000, POSEIDON-60kV, UNU-ICTP, MJOLNIR, etc., that happen to share a file.

---

## 2. Three Options

### Option A — Per-device modules + `__init__.py` registry

```
src/dpf/validation/devices/
  __init__.py              # registry + re-exports for backward-compat
  pf1000.py                # PF1000_DATA, PF1000_GRIBKOV_DATA, PF1000_16KV_DATA, PF1000_20KV_DATA
  nx2.py                   # NX2_DATA
  unu_ictp.py              # UNU_ICTP_DATA
  poseidon.py              # POSEIDON_60KV_DATA, POSEIDON_DATA
  faeton.py                # FAETON_DATA
  mjolnir.py               # MJOLNIR_DATA
  aecs_pf2.py              # AECS_PF2_DATA
src/dpf/validation/experimental_devices.py   # shim: re-exports from devices/
```

| Pros | Cons |
|---|---|
| Each device edit isolates to ~80-line file | 8 new files |
| Per-family grouping (4 PF-1000 variants together) | Registry must be assembled in `__init__.py` |
| Trivial backward-compat via shim re-exports | Slight import-time overhead (negligible) |
| Git blame becomes useful per-device | Cross-device edits (rare) now span files |
| Zero external API change | None breaking |

**Migration cost:** 1 PR, ~3 hours. Pure mechanical move + re-export.
**Test churn:** zero (imports unchanged).
**Breaking changes:** none.

---

### Option B — YAML/JSON data files + Python loader

```
src/dpf/validation/data/devices/
  pf1000.yaml
  nx2.yaml
  ...
src/dpf/validation/experimental_devices.py   # loader: parses YAML -> ExperimentalDevice
```

| Pros | Cons |
|---|---|
| Non-coders can edit device parameters | Multi-line `measurement_notes` strings awkward in YAML |
| Schema-validatable | Requires schema definition + validation layer |
| | Waveform arrays must stay in Python (NumPy literals don't YAML well) |
| | KR citations as YAML strings lose Python-string concatenation idiom |
| | Loader becomes new failure surface; test churn moderate |
| | Float precision risk: `1.332e-3` round-trip through YAML parsers historically lossy |
| | Violates "papers-are-truth byte-identical" if any float reformats |

**Migration cost:** 1-2 PRs, ~12-16 hours (loader + schema + round-trip tests).
**Test churn:** high — every device test now depends on YAML parser behavior.
**Breaking changes:** internal only (data format), external API preserved via loader.

**Verdict:** rejected. Byte-identical float preservation through YAML is fragile and the human-editing benefit is illusory (the editors are physicists who already read Python literals fine; the friction is *which file*, not *which syntax*).

---

### Option C — Status quo + extract `_REFERENCE_ONLY` only

Move `POSEIDON_DATA` and `AECS_PF2_DATA` into `experimental_devices_reference_only.py`. Keep the 9 active devices in the current monolith.

| Pros | Cons |
|---|---|
| Minimal change (~10 min) | Active-device file still 500+ LOC, still hot |
| Removes "do not use" devices from the active read path | Solves the wrong problem — churn is in active devices, not REFERENCE_ONLY |
| | KR re-anchor cascade still hits one big file |

**Migration cost:** 1 PR, ~30 min.
**Test churn:** zero.
**Breaking changes:** none (re-export shim).

**Verdict:** insufficient. Doesn't address the actual hot-spot.

---

## 3. Recommendation

**Option A (per-device modules).**

**Rationale:**
1. The churn pattern is *per-device KR re-anchoring*, which Option A localizes precisely.
2. Zero behavioral change, zero test churn, zero external API change — pure mechanical refactor.
3. Preserves Python-literal floats (papers-are-truth byte-identical guarantee trivially satisfied — `git mv` + cut/paste).
4. Aligns with prior good factoring (`experimental_waveforms.py` already split out arrays).
5. Lowest reviewer burden: diff is a sequence of moves, not logic changes.

**What it does NOT solve (and that's correct scope):**
- It does not change the `ExperimentalDevice` schema. That's a separate refactor if needed.
- It does not extract `measurement_notes` to external prose files. The notes are tightly coupled to the parameter choices and benefit from co-location.
- It does not address physics-coefficient changes — there are no physics coefficients in this file. The naming "decouples device data from coefficient cascades" is satisfied trivially because the cascades *are* device data.

**Future-optional add-on (defer):** if KR citation strings continue to dominate diffs, a separate `kr_citations.py` mapping `device_name -> {field: kr_locator}` could be split out later. Do not bundle with Option A.

---

## 4. Migration Plan (when greenlit)

| Step | Action | Cost |
|---|---|---|
| 1 | `mkdir src/dpf/validation/devices/` | 1 min |
| 2 | Create 8 per-device files; cut/paste each `ExperimentalDevice(...)` block | 60 min |
| 3 | Move `get_devices_by_provenance` to `devices/__init__.py` | 5 min |
| 4 | Build `DEVICES` and `_REFERENCE_ONLY` dicts in `devices/__init__.py` | 10 min |
| 5 | Replace `experimental_devices.py` body with `from .devices import *` re-export shim | 10 min |
| 6 | Run `pytest tests/test_validation_ci.py -x` | 5 min |
| 7 | Run full validation suite, confirm byte-identical numeric outputs vs pre-refactor | 30 min |
| 8 | Diff `git show` to verify zero numeric value changed | 10 min |
| **Total** | | **~2.5-3 hours, 1 PR** |

**Risk:** trivial. Only mechanical text moves. The verification step (8) is the safety net — any float that changed by even one digit aborts the PR.

---

## 5. Doc Path

`/Users/anthonyzamora/dpf-unified/docs/D2_EXPERIMENTAL_DEVICES_REFACTOR.md` (this file, untracked)
