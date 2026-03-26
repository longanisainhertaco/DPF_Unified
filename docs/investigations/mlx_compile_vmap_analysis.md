# mx.compile() and mx.vmap() Analysis for DPF-Unified MLX Solver

**Date**: 2026-03-26
**MLX Version**: 0.31.0
**Hardware**: M3 Pro, 36GB unified memory
**Scope**: Empirical testing of compile/vmap on actual DPF solver functions

---

## Executive Summary

Both `mx.compile()` and `mx.vmap()` work with our solver code patterns. Key findings:

| Feature | Works? | Speedup | Recommendation |
|---------|--------|---------|----------------|
| `mx.compile(_hlls_flux_gpu)` | Yes (per-dim closures) | **1.82-2.26x** | Deploy immediately |
| `mx.compile(cons_to_prim)` | Yes (already wired) | **1.54x** | Already deployed |
| `mx.vmap` for AMR batching | Yes | **2.41x** vs loop | Use for AMR blocks |
| `compile(vmap(f))` | Yes | **2.44x** vs loop | Preferred composition order |
| `vmap(compile(f))` | Yes | Same | Both orders work |
| Shape caching | Efficient | 1.06x overhead | Safe for variable grids |

---

## 1. mx.compile on _hlls_flux_gpu

### Question
`_hlls_flux_gpu` has Python control flow: `if dim==0`, `elif dim==1`, `else`. Can it be compiled?

### Answer: YES, with per-dim closures

`mx.compile` traces Python at compile time. When Python `if/else` branches depend on
a Python integer (not an `mx.array`), the tracer evaluates the branch at trace time and
bakes the taken path into the compiled graph. This means:

- A single `mx.compile(_hlls_flux_gpu)` call with `dim=0` traces the `dim==0` branch.
- Calling the same compiled function with `dim=1` **re-traces** (MLX detects the changed
  argument and recompiles). This is correct but incurs recompilation cost on first call.

**Recommended pattern** (already in MLX_OPT_PHASE1_PROTOTYPE.md):

```python
_HLLS_COMPILED = {}

def get_hlls_compiled(dim: int):
    if dim not in _HLLS_COMPILED:
        def _fn(QL, QR, gamma):
            return _hlls_flux_gpu(QL, QR, gamma, dim=dim)
        _HLLS_COMPILED[dim] = mx.compile(_fn)
    return _HLLS_COMPILED[dim]
```

This creates 3 permanently cached compiled functions (dim=0,1,2). No re-tracing after
first call per dim.

### Correctness verification

Per-dim compiled closures vs uncompiled reference:

| dim | Max absolute error | NaN? |
|-----|--------------------|------|
| 0 | 3.05e-05 | No |
| 1 | 6.10e-05 | No |
| 2 | 3.05e-05 | No |

Errors are from floating-point reassociation during fusion (compile reorders
elementwise ops). 6e-05 is well within float32 tolerance for MHD fluxes.

### Performance

| Grid | Compiled (ms) | Uncompiled (ms) | Speedup |
|------|---------------|-----------------|---------|
| 32x64 (2K cells) | 0.415 | 0.794 | 1.91x |
| 64x128 (8K cells) | 0.388 | 0.797 | 2.06x |
| 128x256 (32K cells) | 0.557 | 1.081 | 1.94x |
| 256x512 (131K cells) | 1.733 | 3.913 | 2.26x |

Speedup increases with grid size (more fusion opportunity, less launch overhead
relative to compute). At 256x512, compile saves 2.2 ms per call x 6 calls/step
= **13 ms/step**.

### Inner closure concern

The `_pflux_mlx` inner function references `ib_n`, `ib_t1`, etc. from the outer scope.
These are Python ints determined by the `dim` argument. Since we use per-dim closures,
these are constants at trace time. **No issue.**

The list comprehension `[F_out[i:i+1] if i != ib_n else F_zero for i in range(NVAR)]`
is also Python-level control flow evaluated at trace time. **No issue.**

---

## 2. mx.compile on cons_to_prim

### Question
Does `cons_to_prim` have any compile-blocking patterns?

### Answer: NO — it compiles cleanly

`_cons_to_prim_impl` is pure elementwise: `mx.maximum`, `mx.reciprocal`, multiply,
subtract, index. No Python control flow, no dynamic shapes, no side effects.

The codebase already has the compile wiring in `mlx_primitives.py:57-67`:

```python
_COMPILED: dict[str, object] = {}

def _compile_if_available(fn):
    try:
        return mx.compile(fn)
    except Exception:
        return fn
```

And `cons_to_prim()` at line 141-143 lazily compiles on first call.

### Performance

| | Time (ms/call) | Speedup |
|---|---|---|
| Uncompiled | 0.242 | - |
| Compiled | 0.157 | **1.54x** |

Called once per RK stage (3x/step). Saves ~0.25 ms/step.

### prim_to_cons compilability

`prim_to_cons` has `if e_electron is None:` — Python control flow on a Python value.
This traces correctly when compiled via a closure that fixes `e_electron` presence.
In practice, DPF always passes `e_electron`, so a single compiled variant suffices.

---

## 3. mx.vmap for AMR Block Batching

### Question
Can `mx.vmap(f, in_axes=0)` handle functions that internally use `mx.maximum`,
`mx.where`, `mx.sqrt`? Are there ops that don't support vmap?

### Answer: All common MHD ops work with vmap

Tested operations that PASS under vmap:

| Operation | Works? |
|-----------|--------|
| `mx.maximum(a, scalar)` | Yes |
| `mx.where(cond, a, b)` | Yes |
| `mx.sqrt(x)` | Yes |
| `mx.reciprocal(x)` | Yes |
| `mx.minimum(a, b)` | Yes |
| `mx.stack([...], axis=0)` | Yes |
| `mx.power(x, scalar)` | Yes |
| `mx.abs(x)` | Yes |
| `mx.isnan(x)` | Yes |
| Array indexing `x[0]`, `x[i:j]` | Yes |
| Arithmetic (`+`, `-`, `*`, `/`) | Yes |
| Comparison (`>`, `<`, `>=`) | Yes |

### Known vmap limitations (from MLX docs + GitHub issues)

Operations that do NOT support vmap:

1. **`mx.convolve`** — [Issue #2085](https://github.com/ml-explore/mlx/issues/2085). Not used in our solver.
2. **Double vmap with advanced indexing** — [Issue #1517](https://github.com/ml-explore/mlx/issues/1517). We only need single vmap.
3. **`mx.random.split`** — [Discussion #1108](https://github.com/ml-explore/mlx/discussions/1108). Not used in solver.
4. **Functions returning constants** — Fixed in PR #1524. Not an issue in v0.31.0.

All operations in `_hlls_flux_gpu`, `cons_to_prim`, WENO5-Z reconstruction, and
geometric source terms are vmap-compatible.

### Multi-input vmap

Riemann solvers take two arrays (QL, QR). Tested:

```python
vmapped_riemann = mx.vmap(riemann_like, in_axes=(0, 0))
result = vmapped_riemann(batch_UL, batch_UR)  # (8, 10, 16, 32)
```

Works correctly. Both inputs are batched along axis 0.

### Performance: vmap vs manual loop

8 blocks of (10, 32, 64):

| Method | Time (ms) | Speedup vs loop |
|--------|-----------|-----------------|
| Manual Python loop | 0.541 | 1.00x |
| `mx.vmap` | 0.224 | **2.41x** |
| `mx.compile(mx.vmap(...))` | 0.222 | **2.44x** |

The vmap advantage comes from building a single fused graph instead of 8 separate
evaluations. The compile-on-top-of-vmap adds marginal benefit here because vmap
already produces a single graph.

---

## 4. mx.compile + mx.vmap Composition

### Question
Can you `compile(vmap(f))`? Or `vmap(compile(f))`? Which order?

### Answer: BOTH orders work, prefer `compile(vmap(f))`

| Composition | Works? | Notes |
|-------------|--------|-------|
| `mx.compile(mx.vmap(f))` | Yes | Single compiled graph for the entire batched operation |
| `mx.vmap(mx.compile(f))` | Yes | Each vmap lane runs a compiled function |

From the [MLX docs](https://ml-explore.github.io/mlx/build/html/usage/compile.html):

> "A transformation of a compiled function will not by default be compiled.
> To compile the transformed function simply pass it through compile()."

**Recommended**: `compile(vmap(f))` — this gives the compiler the most opportunity to
fuse across the batch dimension. The inner-compile variant `vmap(compile(f))` means
the vmap transform sees an opaque compiled function and cannot optimize across lanes.

In practice, both produced identical results and near-identical performance in our
benchmarks, but `compile(vmap(f))` is architecturally cleaner and recommended by MLX.

---

## 5. mx.compile Shape Caching Behavior

### Question
Does mx.compile cache across calls with different input shapes?

### Answer: YES — MLX caches multiple shape traces efficiently

Tested: alternating between shapes `(64, 128)` and `(32, 256)` for 1000 calls.

| Pattern | Time (ms) | Ratio |
|---------|-----------|-------|
| Same shape, 1000 calls | 103.3 | 1.00x |
| Alternating shapes, 1000 calls | 109.6 | **1.06x** |

Only 6% overhead for shape alternation. MLX maintains an internal cache of compiled
graphs keyed by input shapes/dtypes. Recompilation only happens on first encounter
of a new shape.

### Recompilation triggers (from docs)

| Change | Behavior |
|--------|----------|
| Shape changes (same ndim) | Partial recompilation (fast) |
| Dimensionality changes | Full recompilation |
| Dtype changes | Full recompilation |
| Number of inputs changes | Full recompilation |
| Python control flow args change | Re-trace (re-evaluates Python branches) |

### `shapeless=True` parameter

MLX 0.31 supports `mx.compile(fn, shapeless=True)` which avoids ANY recompilation
on shape changes. However, this fails for shape-dependent computations like
`mx.reshape(x, (x.shape[0], -1))` where the shape is used as a value.

Our solver functions use shapes only for slicing (`Q[0:1]`, `Q[:, start:end]`),
which are shape-dependent. **Do not use `shapeless=True`** for our solver functions.
The default shape-caching behavior (6% overhead) is sufficient.

---

## 6. Compile Purity Requirement

`mx.compile` requires pure functions (no side effects). Key implications for DPF:

| Pattern | Pure? | Action |
|---------|-------|--------|
| `_hlls_flux_gpu` | Yes | Compile directly |
| `cons_to_prim` | Yes | Already compiled |
| `_weno5z_left_biased` | Yes | Already compiled |
| `MLXMHDSolver.step()` | **No** (modifies self) | Do NOT compile |
| Functions calling `mx.eval()` | **No** (side effect) | Do NOT compile |
| Functions calling `np.asarray()` | **No** (CPU sync) | Do NOT compile |

The compiled functions must be the leaf-level numerical kernels, not the orchestrator.

---

## 7. Recommendations for DPF-Unified

### Immediate (OPT Phase 1)

1. **Wire per-dim compiled HLLS** into `compute_fluxes`:
   ```python
   _HLLS_COMPILED = {}
   def _get_hlls_compiled(dim):
       if dim not in _HLLS_COMPILED:
           def fn(QL, QR, gamma):
               return _hlls_flux_gpu(QL, QR, gamma, dim=dim)
           _HLLS_COMPILED[dim] = mx.compile(fn)
       return _HLLS_COMPILED[dim]
   ```
   Expected: **1.8-2.3x** on Riemann portion, **13 ms/step** saved at 256x512.

2. **Compile `fast_magnetosonic_boris`** (same per-dim pattern). 1.05x on CFL.

3. **Compile `_bremsstrahlung_logspace`** when it lands. 1.3x on that source term.

### Medium-term (AMR)

4. **Use `mx.compile(mx.vmap(rhs_block))` for AMR block batching**:
   ```python
   def rhs_single_block(U, grid_params):
       # Full MHD RHS for one block
       ...
       return dU_dt

   rhs_batch = mx.compile(mx.vmap(rhs_single_block, in_axes=(0, 0)))
   dU_dt_all = rhs_batch(U_blocks, grid_params_batch)
   ```
   Expected: **2.4x** vs sequential block processing.

### Do NOT do

- Do NOT compile `step()` or any method that calls `mx.eval()`.
- Do NOT use `shapeless=True` for solver kernels (shape-dependent slicing).
- Do NOT compile functions with `np.asarray()` calls (the bremsstrahlung NumPy detour
  must be replaced with log-space MLX first).
- Do NOT use `mx.vmap` with `mx.convolve` (not supported).

---

## Appendix: Raw Benchmark Data

### Test environment
- MLX 0.31.0
- macOS 15.x, M3 Pro
- Python 3.11
- All benchmarks: warmup 5 iterations, then N timed iterations with `mx.eval()` sync

### _hlls_flux_gpu compile benchmark (128x256, N=200)

```
Uncompiled: 1.104 ms/call
Compiled:   0.608 ms/call
Speedup:    1.82x
```

### cons_to_prim compile benchmark (128x256, N=200)

```
Uncompiled: 0.242 ms/call
Compiled:   0.157 ms/call
Speedup:    1.54x
```

### vmap block batching (8 blocks x 10x32x64, N=200)

```
Manual loop:      0.541 ms
vmap:             0.224 ms  (2.41x)
compile(vmap):    0.222 ms  (2.44x)
```

### Shape caching (1000 calls)

```
Same shape:        103.3 ms
Alternating:       109.6 ms  (1.06x overhead)
```

---

## Sources

- [MLX Compilation Documentation](https://ml-explore.github.io/mlx/build/html/usage/compile.html)
- [MLX Function Transforms Documentation](https://ml-explore.github.io/mlx/build/html/usage/function_transforms.html)
- [mx.vmap API Reference](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.vmap.html)
- [Cannot vmap mx.convolve — Issue #2085](https://github.com/ml-explore/mlx/issues/2085)
- [vmap constant output bug — Issue #1516](https://github.com/ml-explore/mlx/issues/1516)
- [Double vmap indexing bug — Issue #1517](https://github.com/ml-explore/mlx/issues/1517)
- [Compile behavior discussion — Issue #712](https://github.com/ml-explore/mlx/issues/712)
- [mx.compile for class methods — Discussion #837](https://github.com/ml-explore/mlx/discussions/837)
