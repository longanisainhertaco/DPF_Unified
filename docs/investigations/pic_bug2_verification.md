# PIC Bug 2 Verification: Esirkepov dt Mismatch

**Date**: 2026-03-26
**Investigator**: dpf-validation-engineer
**Verdict**: BUG CONFIRMED -- real, but at a different location than agents claimed

## Agent Claims vs Reality

Three agents claimed `self.dt` is "hardcoded at line 1561 of hybrid.py." A readiness
check agent found the public API takes `dt` as an explicit parameter. Both statements
are partially correct but paint an incomplete picture.

## Findings

### 1. The public `deposit_current_esirkepov()` function (line 1162)

YES, it takes `dt` as an explicit parameter:

```python
def deposit_current_esirkepov(
    positions_old: np.ndarray,
    positions_new: np.ndarray,
    weights: np.ndarray,
    charge: float,
    grid_shape: tuple[int, int, int],
    dx: float, dy: float, dz: float,
    dt: float,                          # <-- explicit parameter
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
```

This is a standalone module-level function. No bug here.

### 2. The `HybridPIC.deposit()` method (line 1525)

This method calls `deposit_current_esirkepov` with `self.dt`:

```python
# hybrid.py line 1559-1561
jx, jy, jz = deposit_current_esirkepov(
    sp.positions_old, sp.positions, sp.weights, sp.charge,
    self.grid_shape, self.dx, self.dy, self.dz, self.dt,  # <-- self.dt
)
```

`self.dt` is set ONCE in `__init__` (line 1322) and NEVER updated.

### 3. The `HybridPIC.push_particles()` method (line 1402)

Takes `dt` as an optional parameter, defaults to `self.dt`:

```python
def push_particles(self, E, B, dt: float | None = None) -> None:
    if dt is None:
        dt = self.dt
    # ... uses dt for Boris push at line 1452
```

### 4. How the engine calls these methods

In `engine/core.py` line 876-878:

```python
self.kinetic.step(dt, self.time, E_fld, B_fld)     # passes MHD dt
# ...
Jx, Jy, Jz = self.kinetic.get_current_density()    # calls deposit()
```

In `kinetic/manager.py` line 123:

```python
self.driver.push_particles(E_field, B_field, dt=dt)  # passes MHD dt <<<
```

In `kinetic/manager.py` line 134:

```python
def get_current_density(self):
    _, Jx, Jy, Jz = self.driver.deposit()            # no dt arg -- uses self.dt <<<
```

### 5. Where `self.dt` comes from

`KineticManager.__init__` (manager.py line 38):

```python
self.driver = HybridPIC(
    ...
    dt=1e-9,  # initial dt; overridden each step() call   <-- COMMENT IS WRONG
)
```

The comment says "overridden each step() call" but **nothing ever updates
`self.driver.dt`**. It stays at `1e-9` forever.

## The Bug

The call sequence each engine step is:

1. `kinetic.step(dt=mhd_dt)` --> `driver.push_particles(dt=mhd_dt)` -- Boris push uses **mhd_dt**
2. `kinetic.get_current_density()` --> `driver.deposit()` -- Esirkepov uses **self.dt = 1e-9**

Esirkepov current deposition computes `J = q * (x_new - x_old) / (cell_volume * dt)`.
If `dt` in this formula differs from the `dt` used to compute `x_new - x_old` (the Boris
push), the resulting J violates charge conservation: `div(J) + drho/dt != 0`.

**Concrete impact**: With mhd_dt ~ 1e-11 to 1e-8 s (typical DPF CFL), the Esirkepov
deposit uses dt=1e-9 regardless. This means:

- If mhd_dt = 1e-11: J is **100x too small** (denominator 100x too large)
- If mhd_dt = 1e-8: J is **10x too large** (denominator 10x too small)
- If mhd_dt = 1e-9: J is correct by coincidence

## Location

The bug is NOT at line 1561 (that line just passes `self.dt`, which is correct syntax).
The bug is the **absence of an update to `self.dt`** combined with the separation between
`push_particles()` (receives dt) and `deposit()` (does not).

**Root cause site**: `kinetic/manager.py` lines 123+134 -- `step()` passes dt to push
but `get_current_density()` calls `deposit()` which uses stale `self.dt`.

## Is `_last_push_dt` already implemented?

NO. `_last_push_dt` appears only in documentation files:
- `docs/PIC_VALIDATION_SCAFFOLD.md`
- `docs/PIC_PROTOTYPE_CODE.md`

It does not exist in any source code.

## Fix (5 lines)

### Option A: Store push dt, use in deposit (recommended)

**hybrid.py, push_particles method (after line 1431):**

```python
# BEFORE (line 1431-1432):
if dt is None:
    dt = self.dt

# AFTER:
if dt is None:
    dt = self.dt
self._last_push_dt = dt
```

**hybrid.py, __init__ method (after line 1322):**

```python
# ADD after line 1322:
self._last_push_dt = dt
```

**hybrid.py, deposit method (line 1561):**

```python
# BEFORE:
self.grid_shape, self.dx, self.dy, self.dz, self.dt,

# AFTER:
self.grid_shape, self.dx, self.dy, self.dz, self._last_push_dt,
```

### Option B: Pass dt through deposit (cleaner API but breaks callers)

Change `deposit()` to accept `dt` parameter and update `get_current_density()` in
`manager.py` to pass it. More invasive, not recommended as first fix.

## Impact on Dependency Graph

The bug IS real. The 3 agents' dependency graphs remain valid in structure. However:
- The fix location is `hybrid.py` lines 1322, 1432, 1561 (not "line 1561 hardcoded")
- The fix is 3 insertions + 1 edit = 4 changed lines, not a redesign
- No other bugs depend on this one being fixed first -- it's leaf-level
