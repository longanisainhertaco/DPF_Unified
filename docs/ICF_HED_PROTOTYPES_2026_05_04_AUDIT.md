# ICF/HED Prototypes Audit — 2026-05-04
Wave-7 hygiene check. Verifying O7 claim: "none integrated."

## Code Integration Audit

```
grep -rn "from.*icf_hed|import icf_hed|docs/icf-hed-prototypes" src/dpf/ tests/ --include="*.py"
```

Result: **ZERO hits. Clean.**

## Per-Prototype Status Table

| File | Lines (README) | Status Header | Integrated? | Notes |
|---|---|---|---|---|
| tabular_eos.md | 1110 | Standalone prototype — not integrated | No | Wave-5 marked "Deleted 2026-04-30" per O7 audit; file still present on disk |
| multigroup_radiation.md | 958 | Standalone prototype — NOT integrated | No | Status explicit in header |
| grmhd.md | 1113 | Standalone prototype — NOT integrated | No | Valencia formulation; DPF v/c ~1e-3, GR irrelevant |
| laser_plasma.md | 795 | Standalone prototype — NOT integrated | No | Explicitly: "Irrelevant to DPF (see Section 7)" |
| multi_material_ale.md | 1174 | Standalone; NOT integrated | No | Single D2 fill gas; no material interfaces |
| nuclear_burn.md | 1078 | Not integrated into production code | No | Beam-target vs TN yield doc; shelf ref |
| self_gravity.md | 862 | Standalone prototype — NOT integrated | No | Gravity negligible at DPF scales |
| wire_array_dynamics.md | 1155 | Standalone prototype — NOT integrated | No | Z-machine / MagLIF; not DPF coaxial geometry |

Total: 8 prototypes, ~8245 LOC. All shelf-only.

## Discrepancy Note

`tabular_eos.md` was recorded as "Deleted 2026-04-30" in the Wave-5 audit trail, but the file is present on disk at `docs/icf-hed-prototypes/tabular_eos.md`. Either the deletion was not executed or was reverted. No integration impact — the file is still unlinked from `src/dpf/`.

## BLOCKERS

None. "None integrated" claim holds. All 8 prototypes are shelf-only research docs with no Python import paths, no wiring into `engine.py`, and no test references.

## Doc Path

`/Users/anthonyzamora/dpf-unified/docs/icf-hed-prototypes/`
