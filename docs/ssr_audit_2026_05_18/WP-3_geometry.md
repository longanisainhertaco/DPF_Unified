# WP-3 / SSR-005 Audit — Reviewed Geometry And Material Boundaries

Date: 2026-05-18
Auditor scope: WP-3 / SSR-005 (Reviewed Geometry And Material Boundaries)
Repo: `/Users/anthonyzamora/dpf-unified` — branch `codex/corpus`
Audit mode: static, read-only. No `.py` or existing doc modified. No pytest/CLI run.
Python: `.venv312/bin/python` 3.12.

Files audited:
- `src/dpf/fields/source_geometry.py`
- `src/dpf/fields/particle_boundaries.py`
- `src/dpf/fields/maxwell_3d.py` (boundary/PML semantics only)
- `src/dpf/first_principles/runner.py` (geometry/boundary helpers, `boundary_policy`,
  conductor mask packets, `_deck_source_diff_packet`)
- `src/dpf/first_principles/deck.py` (PF-1000/Akel deck preset — geometry inputs only)

---

## (a) Verdict And Reasoning

### Verdict: `request_changes`

The implementation is **honest at the status level** and is **not an overclaim**:
every geometry/boundary packet carries `status: candidate_*_not_validation` and
`can_support_first_principles_acceptance: False`. The conductor-mask code does
project 12 discrete rods plus an anode cylinder, the boundary policy is wired
into the Maxwell core, and the `_deck_source_diff_packet` does emit a
source-locked deck-diff for PF-1000/Akel runs. That is genuine WP-3 progress and
clears the `reject_overclaim` bar.

However, SSR-005 requires that the geometry packet "Emit geometry packet with
mask hash, grid spacing, projected dimensions, source values, and **error from
source dimensions**." Static inspection of the entire `runner.py` confirms:

- **No mask hash** is emitted anywhere. `grep -i "hash"` over `runner.py` returns
  zero hits. The conductor mask is a `np.ndarray` that is never digested.
- **No projection error / error-from-source-dimensions** field exists.
  `grep -i "projection_error|error_from|projected_"` over `runner.py` returns
  zero hits. The mask packet reports `device_*` source values and
  `conductor_cells_active`, but never the discretization error between the
  reviewed source dimension and the voxelized mask.
- **No insulator mask, no electrode-backplate / source-interface mask, no
  vacuum-chamber mask** is built. The conductor mask covers only anode +
  cathode-rods. The insulator is carried as a string label
  (`device_insulator_material="alumina"`) and the packet itself states
  `insulator_material_surface_resolved: False`.

Separately, the default grid presets (`coarse=5^3`, `medium=7^3`, `fine=9^3` —
`cli/main.py:438-442`) place **0.73 to 1.45 cells across one 80 mm rod
diameter** (numerically confirmed below). At those resolutions the
"rod-resolved" `pf1000_rod_hollow_projection` mask degenerates to single-voxel
blobs. The packet does **not** flag this — it reports
`cathode_rods_projected: True` and `coordinate_interpretation:
centered_cartesian_full_azimuth_projection` with no resolution gate, so a reader
of the artifact would believe rods are resolved when they are not. This is the
axisymmetry/rod-fidelity honesty gap that Audit Phase 6 ("Geometry") and the
Rejection Criteria target.

Because SSR-005's required packet fields (mask hash, projection error) are
absent and the geometry currently cannot honestly distinguish a resolved rod
mask from an unresolved blob, the submission needs changes. It is not a reject —
status discipline is intact — but it cannot advance past `request_changes`
until the projection-error/hash packet and a resolution gate exist.

---

## (b) Source Evidence Table

All cited KR line ranges were opened and verified during this audit.

| KR source path:lines | Claim in code | Verified? | Notes |
| --- | --- | --- | --- |
| `experimental-study-...-pf-1000-facility-705bcc83.md:340-356` | 12 cathode rods 80 mm dia; CE (anode) radius 115.5 mm; OE (cathode) radius 200 mm; CE length 460 mm; alumina insulator extends 85 mm; vacuum chamber 1400 mm dia x 2500 mm; bank 1332 uF | **TRUE** | Lines 344-351 state rods/radii/length/insulator verbatim; 342-343 state chamber dims; 353 states 1332 uF. Exact match to the claim. |
| `radiation-physics-and-chemistry-188-2021-109633.md:108-142` | PF-1000/Akel geometry, bank, pressure, diagnostics | **TRUE** | 111-117: 480 mm electrodes, copper anode tube dia 231 mm, twelve 8-cm cathode tubes, C0=1332 uF, 16 kV, 170.5 kJ, 1.05/1.2 Torr. Match. |
| `radiation-physics-and-chemistry-188-2021-109633.md:262-270` | shot-12581 deck: L0=25 nH, C0=1332 uF, r0=6.1 mOhm, b=16 cm, a=11.55 cm, z0=48 cm, V0=16 kV, p0=1.2 Torr | **TRUE** | 263-265 state Bank/Tube/Operational lines verbatim; 266-268 define b=cathode radius, a=anode radius, z0=anode length. Match. |
| `runner.py:2543-2545` conductor-mask packet cites `"Krauz 2012 PF-1000 geometry lines 344-351 and 453-454"` | rod/electrode/insulator geometry | **TRUE** | 705bcc83:344-351 verified above. 705bcc83:453-454 is the Fig.3 caption naming anode/cathode-12-rods/insulator/vacuum-lock — supports the geometry-feature labels. Citation accurate. |
| `runner.py:2544` conductor-mask packet cites `"PF1000 geometry lines 111-117, 262-268"` | electrode/circuit geometry | **TRUE** | 109633:111-117 and :262-268 verified above. Accurate. |
| `runner.py:2542-2543, 2607-2608` boundary packets cite `HYBRID_PIC_3D_SOURCE` lines `613-619, 625-628, 640-641` | particles entering conductor/PML are absorbed; PML on axial boundary; conductor reflects | **TRUE** | `fully-electromagnetic-hybrid-pic-fluid-...-acb71fa9.md:613-619` states open/PML/conductor boundaries and "Particles entering either the conductor region or the PML region are absorbed and deleted." Citation accurate. |
| `source_geometry.py:15-17` `HybridPICSourceGeometry` declares `source_lines="632-740"`, scope `llnl_like_180ka_axisymmetric_hybrid_pic` | LLNL-like 2D-RZ setup, anode r=0.01 m, L=0.05 m | **TRUE but out-of-scope** | acb71fa9 around line 641 states "A 5-cm-long anode ... radius 1.0 cm. The cathode measures 10 cm." This is the **LLNL device, not PF-1000**. The citation is accurate to its own source, but this packet is not a PF-1000 geometry packet — see (d). |

### Source contradiction the deck must not silently absorb

The two PF-1000 KR papers **disagree on geometry** and the deck must treat this
as a documented transfer/scope decision, not a silent pick:

| Quantity | `705bcc83:344-351` (Krauz 2012) | `109633:111-117 / 262-268` (Akel 2021) | Deck value (`deck.py:842-848`) |
| --- | --- | --- | --- |
| Anode (CE / "a") radius | 115.5 mm | a = 11.55 cm = 115.5 mm | 0.1155 m — match both |
| Cathode (OE / "b") radius | 200 mm | b = 16 cm = 160 mm | 0.16 m — matches Akel, **NOT Krauz** |
| Anode length (CE / "z0") | 460 mm | 480 mm / z0 = 48 cm | 0.48 m — matches Akel, **drifts +4.3% from Krauz 460 mm** |
| Rod diameter | 80 mm | 8 cm = 80 mm | 0.080 m — match both |
| Rod count | 12 | twelve | 12 — match both |
| Anode wall | (not stated) | "tube of diameter 231 mm" -> hollow, inner≈outer 115.5 mm | inner radius **not set** -> rendered solid |

The deck picks the Akel-2021 values (b=160 mm, L=480 mm) and cites BOTH papers
as geometry sources. That is internally defensible (Akel is the shot-12581
demonstrator), but `device_cathode_rod_count`/`rod_diameter` come from one paper
and `cathode_radius` from another's interpretation. The `_deck_source_diff_packet`
only checks against `PF1000_AKEL_SOURCE_LOCKED_DECK` (`runner.py:92-104`), whose
`device_cathode_radius_m=0.16` and `device_anode_length_m=0.48` are themselves
Akel-derived — so the diff packet will report `source_locked_match` and **never
surface the 4.3% disagreement with Krauz 705bcc83**, even though 705bcc83 is
cited as the rod-geometry authority. This is a real but bounded honesty gap:
the deck-diff "match" is a match against a self-consistent Akel lock, not against
the Krauz rod paper.

---

## (c) Mask Coverage Table

SSR-005 expected masks (spec lines 293-299) vs implementation:

| Mask | Present? | Source-cited? | Status emitted | Notes |
| --- | --- | --- | --- | --- |
| 12 cathode rods | **Yes** | Yes (705bcc83:344-351, 109633:111-117) | `candidate_geometry_mask` / `candidate_pf1000_rod_hollow_projection` | `_pf1000_rod_hollow_conductor_mask` (`runner.py:2477-2527`) loops 12 rods at `angle = 2*pi*i/12`, centered Cartesian. Geometrically correct construction. **But unresolved at default grids** — see (d). |
| Hollow / copper anode | **Partial** | Yes (109633:112 "tube of diameter 231 mm") | `candidate_*` | `_pf1000_rod_hollow_conductor_mask` supports `anode_inner_radius`, but the PF-1000 deck never sets `device_anode_inner_radius_m` (`deck.py:840-851`), so `anode_inner_radius=0.0` (`runner.py:2501-2505`) -> **anode rendered solid, not hollow.** Packet honestly reports `hollow_anode_inner_radius_supplied: False` and limitation "Hollow-anode bore is not resolved." Honest, but the mask does not match the source. |
| Alumina insulator | **Missing** | Label only | n/a | No insulator voxel/material region is built. `device_insulator_material="alumina"` and `device_insulator_length_m=0.085` are carried as scalars; mask packet states `insulator_material_surface_resolved: False`. SSR-005 explicitly lists "alumina insulator" as a required mask. Gap is disclosed but the mask is absent. |
| Electrode backplate / source interface | **Missing** | No | n/a | No backplate mask and no `source_interface` label. 705bcc83:352 ("the back plate of the OE") supports a backplate; it is not modeled. The power-port "source interface" exists conceptually elsewhere but is not a geometry label here. |
| Vacuum chamber / wall | **Missing** | Source exists, unused | n/a | 705bcc83:342-343 gives chamber 1400 mm dia x 2500 mm. No chamber/wall mask is built. The grid outer boundary is treated as open/PML, not as a reviewed chamber wall. Spec says "if active" — acceptable to omit, but then the artifact must not imply a chamber boundary. |
| Material surfaces (ablation / electrode work) | **Missing** | No | n/a | No per-surface material regions. SSR-005 says "if modeled" — not modeled, so omission is allowed; must stay out of any acceptance claim. |

Candidate vs accepted: **every** mask path is `candidate`. `conductor_mask_status`
is validated against `{not_supplied, candidate_geometry_mask,
reviewed_same_scope_geometry_mask}` (`runner.py:1479-1484`) and the PF-1000 deck
sets `candidate_geometry_mask`. No path currently sets
`reviewed_same_scope_geometry_mask`. **No mask is marked accepted.** Status
discipline here is correct.

---

## (d) Projection-Error / Axisymmetry Honesty Check

### CRITICAL FINDING — rod-fidelity claim is not resolution-gated

`_pf1000_rod_hollow_conductor_mask` (`runner.py:2477-2527`) builds a genuine
3-D, full-azimuth, 12-discrete-rod mask — it is **not** an axisymmetric
projection (the axisymmetric path is the separate
`_axisymmetric_coaxial_conductor_mask`, `runner.py:2454-2474`). So the code does
not "look axisymmetric while claiming rod authority" in the construction. The
honesty problem is the opposite and subtler: **the rod mask claims rod
resolution at grid resolutions where rods cannot be resolved, with no gate.**

Numerically verified this audit (`.venv312/bin/python`, PF-1000 deck geometry,
`xy_spacing` from `deck.py:826-827` `= 2.2*(0.16+0.04)/(n-1)`):

```
shape 5^3 : 0.73 cells across one 80 mm rod diameter
shape 7^3 : 1.09 cells across one 80 mm rod diameter
shape 9^3 : 1.45 cells across one 80 mm rod diameter   (CLI "fine" preset)
```

At <= ~2 cells per rod diameter the 12 rods are sub-grid features:
`(rod_radius_field <= rod_radius)` (`runner.py:2525`) selects 0 or 1 voxel per
rod, and the 12-fold azimuthal structure is lost — the mask is
indistinguishable from a coarse ring. The CLI default for
`first-principles-3d` / `experimental-whole-shot` is `coarse=5^3`
(`cli/main.py:439`), i.e. **0.73 cells per rod — rods literally cannot appear.**

Yet the emitted packet reports, with no resolution dependence:
- `pf1000_geometry_features.cathode_rods_projected: True` (`runner.py:2564-2568`)
  — true only that the *deck declares* rods, but reads as "rods are in the mask";
- `coordinate_interpretation: centered_cartesian_full_azimuth_projection`;
- limitation text "PF-1000 cathode rods are projected onto a Cartesian
  engineering grid" — describes intent, not the achieved resolution.

There is **no field reporting cells-per-rod, no minimum-resolution gate, and no
projection error**. A downstream reader of `boundary_policy.conductor_mask`
cannot tell a 0.73-cell blob from a properly resolved rod cage. That is exactly
the failure Audit Phase 6 ("Verify mask dimensions and projection error") and
the Rejection Criteria ("reject geometry that looks axisymmetric while claiming
rod-level PF-1000 authority") are written to catch.

Mitigating facts (why this is `request_changes`, not `reject_overclaim`):
- `can_support_first_principles_acceptance: False` on every packet;
- `conductor_mask_status` stays `candidate_geometry_mask`, never
  `reviewed_same_scope_geometry_mask`;
- the packet limitations honestly say "No reviewed same-scope electrode mask ...
  is attached." Nothing claims rod-resolved *validation*.

So the status is honest; the **resolution adequacy** is unreported. The fix is
to compute and emit the projection error and a `rod_resolution_sufficient`
boolean, and to forbid `reviewed_same_scope_geometry_mask` whenever the rod
projection is under-resolved.

### Missing SSR-005 packet fields (hard requirement gap)

| SSR-005 required field | Emitted? | Where it should live |
| --- | --- | --- |
| mask hash | **No** | `_conductor_mask_packet` (`runner.py:2530`) |
| grid spacing | Yes | `boundary_policy`/manifest carry `grid_spacing_m` |
| projected dimensions | Partial | mask packet has `conductor_cells_active`, `grid_shape`; no physical projected extents (e.g. realized rod radius/length in voxels x dx) |
| source values | Yes | mask packet echoes all `device_*` |
| error from source dimensions | **No** | `_conductor_mask_packet` |

---

## (e) Proposed Patch Text (authored, NOT applied)

All patches are presented as text only. They harden honesty; they do not relax
any gate. The intent is: (1) make the mask self-describe its discretization
error, (2) make rod under-resolution visible and block accepted status when the
rods are unresolved.

### Patch 1 — add mask hash + projection-error block to `_conductor_mask_packet`

Insert a helper near `_conductor_mask_packet` in `runner.py`:

```python
import hashlib

def _conductor_mask_projection_error(
    *,
    deck: "FirstPrinciples3DDeck",
    grid: "Maxwell3DGrid",
    mask: "np.ndarray | None",
    source: str,
) -> dict[str, Any]:
    """Discretization error of a voxel conductor mask vs source dimensions.

    Honest engineering telemetry only. Not a validation packet. Reports how
    coarsely the reviewed source geometry is resolved so a reader cannot mistake
    an under-resolved blob for a resolved rod cage.
    """
    if mask is None:
        return {
            "status": "no_mask",
            "mask_sha256": None,
            "can_support_first_principles_acceptance": False,
        }
    digest = hashlib.sha256(
        np.ascontiguousarray(mask, dtype=np.uint8).tobytes()
    ).hexdigest()
    dx, dy, dz = grid.spacing
    min_xy = min(float(dx), float(dy))
    rod_d = deck.device_cathode_rod_diameter_m
    cells_per_rod_diam = (
        None if not rod_d else float(rod_d) / min_xy
    )
    rod_projection = source == "candidate_pf1000_rod_hollow_projection"
    # rods are only meaningfully resolved at >= ROD_MIN_CELLS cells per diameter
    ROD_MIN_CELLS = 4.0  # EMPIRICAL: engineering minimum, not a source value
    rod_resolution_sufficient = bool(
        rod_projection
        and cells_per_rod_diam is not None
        and cells_per_rod_diam >= ROD_MIN_CELLS
    )
    # axial cell error vs reviewed anode length
    anode_len = deck.device_anode_length_m
    axial_cells_in_anode = (
        None if not anode_len else float(anode_len) / float(dz)
    )
    return {
        "status": "candidate_conductor_mask_projection_error_not_validation",
        "mask_sha256": digest,
        "grid_spacing_m": [float(dx), float(dy), float(dz)],
        "cells_per_rod_diameter": cells_per_rod_diam,
        "rod_min_cells_threshold": ROD_MIN_CELLS,
        "rod_resolution_sufficient": rod_resolution_sufficient,
        "axial_cells_in_anode_length": axial_cells_in_anode,
        "max_radial_discretization_error_m": 0.5 * min_xy,
        "max_axial_discretization_error_m": 0.5 * float(dz),
        "note": (
            "Voxel masks resolve source dimensions to within +/- half a cell. "
            "When rod_resolution_sufficient is False the 12-rod azimuthal "
            "structure is sub-grid and the mask is a coarse ring, not a "
            "rod-resolved cage."
        ),
        "can_support_first_principles_acceptance": False,
    }
```

Then, inside `_conductor_mask_packet` (`runner.py:2540-2595`), add to the
returned dict:

```python
        "projection_error": _conductor_mask_projection_error(
            deck=deck, grid=grid, mask=mask, source=source
        ),
```

### Patch 2 — block `reviewed_same_scope_geometry_mask` when rods are unresolved

In `_validate_first_principles_3d_deck` (`runner.py`, the block at 1479-1498),
after the existing `conductor_mask_status` / `conductor_mask_mode` checks, add:

```python
    if (
        deck.conductor_mask_status == "reviewed_same_scope_geometry_mask"
        and deck.conductor_mask_mode == "pf1000_rod_hollow_projection"
    ):
        # a reviewed rod mask must actually resolve the rods
        grid_for_check = deck.resolved_grid()  # existing helper at runner.py:284-289
        dx = min(grid_for_check.dx, grid_for_check.dy)
        rod_d = deck.device_cathode_rod_diameter_m
        if not rod_d or (float(rod_d) / dx) < 4.0:
            raise ValueError(
                "reviewed_same_scope_geometry_mask with pf1000_rod_hollow_"
                "projection requires >= 4 cells across a rod diameter"
            )
```

This makes it structurally impossible to label an under-resolved rod mask as
`reviewed`, which is the SSR-005 / Rejection-Criteria requirement.

### Patch 3 — surface the Krauz-vs-Akel anode-length disagreement in the deck-diff

`_deck_source_diff_packet` (`runner.py:2242-2305`) currently only diffs against
the self-consistent Akel lock. Add a secondary, non-blocking cross-source note
so the 4.3% disagreement with the cited Krauz rod paper is visible:

```python
    # cross-source advisory: the rod-geometry paper (Krauz 2012, 705bcc83)
    # gives CE length 460 mm and OE radius 200 mm; the Akel lock uses the
    # shot-12581 values 480 mm / 160 mm. Surface the delta, do not block.
    cross_source = {
        "krauz_705bcc83_anode_length_m": 0.460,
        "krauz_705bcc83_cathode_radius_m": 0.200,
        "akel_locked_anode_length_m": PF1000_AKEL_SOURCE_LOCKED_DECK[
            "device_anode_length_m"
        ],
        "akel_locked_cathode_radius_m": PF1000_AKEL_SOURCE_LOCKED_DECK[
            "device_cathode_radius_m"
        ],
        "note": (
            "PF-1000 geometry sources disagree. Akel shot-12581 values are "
            "used as the demonstrator lock; the Krauz rod paper differs. This "
            "is a scope choice, not a validated reconciliation."
        ),
    }
```

and include `"cross_source_geometry_advisory": cross_source` in the returned
packet for the PF-1000/Akel branch.

### Patch 4 — make `cathode_rods_projected` resolution-aware

In `_conductor_mask_packet`'s `pf1000_geometry_features` block
(`runner.py:2563-2578`), change `cathode_rods_projected` from a deck-declaration
boolean to a resolution-aware one by reusing the Patch-1 result:

```python
            "cathode_rods_declared_by_deck": bool(
                pf1000_rod_projection
                and deck.device_cathode_rod_count
                and deck.device_cathode_rod_diameter_m
            ),
            "cathode_rods_resolved_on_grid": False,  # set True only when
            # projection_error["rod_resolution_sufficient"] is True
```

(Wire the actual value from the Patch-1 `projection_error` dict; the literal
`False` above is a placeholder showing intent.)

---

## (f) Negative Tests — Present vs Missing

### Present

- `tests/test_source_geometry_packet.py` — asserts `HybridPICSourceGeometry`
  stays `candidate` and `can_support_first_principles_acceptance is False`, and
  that the candidate evidence does not satisfy the readiness gate. Good fail-
  closed coverage **for the LLNL packet**, but that packet is not PF-1000
  geometry.
- `tests/test_first_principles_runner.py::test_first_principles_runner_projects_candidate_conductor_mask_from_package_deck`
  — asserts the PF-1000 mask is `candidate_*`, `mask_source` is the rod
  projection, `conductor_mask_status == candidate_geometry_mask`,
  `can_support_first_principles_acceptance is False`. Confirms status, **not**
  resolution adequacy.
- `tests/test_first_principles_runner.py::test_first_principles_runner_applies_candidate_boundary_policy`
  — asserts boundary policy stays candidate and is wired to Maxwell.
- `tests/test_first_principles_runner.py::test_pf1000_runner_emits_source_locked_deck_diff_packet`
  — asserts the deck-diff is emitted and matches the lock.
- `_validate_first_principles_3d_deck` (`runner.py:1485-1498`) rejects unknown
  `conductor_mask_status` / `conductor_mask_mode` and rejects a projection mode
  with `conductor_mask_status == not_supplied`. This is a real structural
  negative control.

### Missing — required by SSR-005 / Rejection Criteria

No test fails when:
1. a coarse / under-resolved geometry is marked
   `reviewed_same_scope_geometry_mask` or otherwise `accepted`;
2. the rod projection runs at a resolution where rods are sub-grid (the packet
   silently still says `cathode_rods_projected: True`);
3. the mask packet omits a mask hash;
4. the mask packet omits projection error / error-from-source-dimensions;
5. PF-1000/Akel geometry is mixed with PF-1000U values (no geometry-scope
   negative control — only circuit/gas deck-diff exists).

There is **no `tests/test_first_principles_geometry.py`**.

### Proposed `tests/test_first_principles_geometry.py` (authored, NOT created)

```python
"""WP-3 / SSR-005 negative controls for reviewed PF-1000 geometry masks.

These tests fail closed: a coarse or under-resolved geometry must never be
marked accepted/reviewed, and the geometry packet must always carry a mask hash
and a projection-error block.
"""

from __future__ import annotations

import pytest

from dpf.first_principles import (
    pf1000_akel_16kv_engineering_deck,
    run_first_principles_3d_deck,
)


def _conductor_mask_packet(shape: tuple[int, int, int]) -> dict:
    deck = pf1000_akel_16kv_engineering_deck(n_steps=1, shape=shape)
    result = run_first_principles_3d_deck(deck)
    return result.telemetry["boundary_policy"]["conductor_mask"]


def test_conductor_mask_packet_emits_mask_hash() -> None:
    """SSR-005: geometry packet must emit a mask hash."""
    packet = _conductor_mask_packet((9, 9, 9))
    proj = packet.get("projection_error")
    assert proj is not None, "projection_error block missing"
    assert proj.get("mask_sha256"), "mask hash missing or empty"
    assert len(proj["mask_sha256"]) == 64


def test_conductor_mask_packet_emits_projection_error() -> None:
    """SSR-005: geometry packet must report error from source dimensions."""
    packet = _conductor_mask_packet((9, 9, 9))
    proj = packet["projection_error"]
    assert proj["max_radial_discretization_error_m"] > 0.0
    assert proj["max_axial_discretization_error_m"] > 0.0
    assert proj["cells_per_rod_diameter"] is not None


def test_coarse_grid_reports_rods_unresolved() -> None:
    """A 5^3 grid puts <1 cell across a rod: must NOT claim rods resolved."""
    packet = _conductor_mask_packet((5, 5, 5))
    proj = packet["projection_error"]
    assert proj["cells_per_rod_diameter"] < 4.0
    assert proj["rod_resolution_sufficient"] is False
    feats = packet["pf1000_geometry_features"]
    # the resolution-aware feature must agree the rods are not resolved
    assert feats.get("cathode_rods_resolved_on_grid") is not True


def test_coarse_geometry_cannot_be_marked_accepted() -> None:
    """Geometry packets must never claim first-principles acceptance."""
    for shape in [(5, 5, 5), (7, 7, 7), (9, 9, 9)]:
        packet = _conductor_mask_packet(shape)
        assert packet["can_support_first_principles_acceptance"] is False
        assert packet["status"].startswith("candidate_")
        assert packet["conductor_mask_status"] != (
            "reviewed_same_scope_geometry_mask"
        )


def test_reviewed_rod_mask_requires_resolved_rods() -> None:
    """A reviewed rod mask on an under-resolved grid must be rejected."""
    deck = pf1000_akel_16kv_engineering_deck(n_steps=1, shape=(5, 5, 5))
    payload = deck.to_dict()
    payload["boundaries"]["conductor_mask_status"] = (
        "reviewed_same_scope_geometry_mask"
    )
    with pytest.raises(ValueError, match="cells across a rod diameter"):
        run_first_principles_3d_deck(payload)


def test_insulator_is_declared_but_not_resolved() -> None:
    """Until an insulator material mask exists, the packet must say so."""
    packet = _conductor_mask_packet((9, 9, 9))
    feats = packet["pf1000_geometry_features"]
    assert feats["insulator_material_surface_declared"] is True
    assert feats["insulator_material_surface_resolved"] is False
```

(Tests 1, 2, 3, 5 require Patches 1, 2, 4. They are written to **fail today**,
which is the intent — they encode the SSR-005 gap as red tests. Test 4 and
test 6 pass against current code and lock in the existing honest behavior.)

---

## (g) Remaining Blockers

1. **No mask hash** in any geometry/conductor packet. SSR-005 requires it.
   Blocking for WP-3 sign-off. Patch 1.
2. **No projection error / error-from-source-dimensions**. SSR-005 requires it.
   Blocking. Patch 1.
3. **Rod-fidelity claim not resolution-gated.** The rod mask reports
   `cathode_rods_projected: True` even at 0.73 cells/rod. Honest at the
   *status* level but misleading at the *fidelity* level. Patches 1, 2, 4.
4. **Alumina insulator mask absent.** SSR-005 lists it as a required mask;
   only a string label exists. Blocker for a "reviewed geometry" claim.
5. **Hollow anode not realized.** Source 109633:112 states a 231 mm-diameter
   anode tube; the PF-1000 deck never sets `device_anode_inner_radius_m`, so the
   anode is solid. Disclosed in limitations, but the mask does not match the
   cited source.
6. **Electrode backplate / source-interface mask absent.** 705bcc83:352 supports
   a backplate; not modeled and not labeled.
7. **Vacuum-chamber wall mask absent.** Source dimensions exist (705bcc83:342-343)
   but no chamber boundary; outer grid is open/PML. Acceptable only if no
   artifact implies a chamber wall.
8. **Krauz-vs-Akel geometry disagreement is invisible in the deck-diff.** The
   deck-diff matches a self-consistent Akel lock and never surfaces the 4.3%
   anode-length / 20% cathode-radius disagreement with the cited Krauz rod
   paper. Patch 3.
9. **`source_geometry.py` is the LLNL device, not PF-1000.** `HybridPICSourceGeometry`
   (anode r=0.01 m, L=0.05 m, 2D-RZ axisymmetric) is a separate
   `llnl_like_180ka` packet. It is honestly scoped and not claimed as PF-1000,
   but a reader scanning `fields/source_geometry.py` for the PF-1000 geometry
   will not find it there — the PF-1000 geometry lives entirely in
   `runner.py` + `deck.py`. Not a defect, but a structural note: SSR-005's
   "reviewed PF-1000 geometry mask packet" has no single home module.
10. **No dedicated geometry test file.** `tests/test_first_principles_geometry.py`
    does not exist. Negative controls for accepted/under-resolved geometry are
    missing. See (f).

### What is already correct (not blockers)

- Every geometry/boundary packet is `candidate_*` with
  `can_support_first_principles_acceptance: False`.
- `conductor_mask_status` is enum-validated; `reviewed_same_scope_geometry_mask`
  is defined but never set by any current deck.
- The 12-rod mask construction (`runner.py:2477-2527`) is geometrically
  correct: discrete rods at `2*pi*i/12`, centered Cartesian, full azimuth.
- Boundary policy is genuinely wired into the Maxwell core: conductor cells zero
  the adjacent E-edges (`maxwell_3d.py:337-340`), PML damping factors are built
  per face/edge (`maxwell_3d.py:182-189, 488-500`), and particle absorption
  deletes conductor/PML/outside particles (`particle_boundaries.py:64-113`).
- All KR line-range citations checked in this audit are accurate — no
  fabricated or wrong-line citations were found.

---

## Audit Provenance

- KR sources opened and verified:
  `experimental-study-...-pf-1000-facility-705bcc83.md:340-356, 450-456`;
  `radiation-physics-and-chemistry-188-2021-109633.md:100-149, 255-279`;
  `fully-electromagnetic-hybrid-pic-fluid-...-acb71fa9.md:613-641`.
- Code read read-only:
  `source_geometry.py` (full), `particle_boundaries.py` (full),
  `maxwell_3d.py` (boundary/PML lines 82-99, 179-340, 402-500),
  `runner.py` (lines 85-405, 1465-1539, 2242-2648),
  `deck.py` (lines 475-527, 775-971), `cli/main.py` (lines 437-447, 2491).
- Numerical check: `.venv312/bin/python` one-liner, rod cells-per-diameter at
  default grid presets. No pytest, no CLI run.
- No `.py` file or existing doc was modified. The only file created is this
  report.
