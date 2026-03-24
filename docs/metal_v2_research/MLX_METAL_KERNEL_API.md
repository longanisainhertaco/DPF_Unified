# MLX Custom Metal Kernel API Reference

## Confirmed: `mx.fast.metal_kernel()` EXISTS

Verified on MLX 0.30.6 (March 2026). The API is stable and production-ready.

## Function Signature

```python
mx.fast.metal_kernel(
    name: str,                          # Kernel function name
    input_names: list[str],             # Parameter names for inputs
    output_names: list[str],            # Parameter names for outputs
    source: str,                        # MSL function body (NOT full function)
    header: str = '',                   # Header code (includes, helper functions)
    ensure_row_contiguous: bool = True, # Auto-contiguify inputs
    atomic_outputs: bool = False,       # Use device atomic<float> for outputs
) -> Callable
```

## Calling the Returned Kernel

```python
kernel = mx.fast.metal_kernel(...)
outputs = kernel(
    inputs: list[mx.array],         # Input arrays (must match input_names order)
    template: list[tuple] = [],     # Template parameters, e.g. [("T", mx.float32)]
    grid: tuple[int, int, int],     # Total threads (x, y, z)
    threadgroup: tuple[int, int, int],  # Threads per threadgroup
    output_shapes: list[tuple],     # Shape of each output array
    output_dtypes: list[mx.Dtype],  # Dtype of each output array
    verbose: bool = False,          # Print compiled MSL source
)
# Returns: list[mx.array]
```

## Auto-Generated Variables in MSL Source

For each input named `foo`:
- `foo` — pointer to the data (`const device float*` or `device float*` for outputs)
- `foo_shape` — shape array (`const constant int*`), access dimensions as `foo_shape[0]`, `foo_shape[1]`, etc.
- `foo_strides` — stride array (`const constant size_t*`)
- `foo_ndim` — number of dimensions (`const constant int&`)

Built-in Metal variables:
- `thread_position_in_grid` — `uint3`, absolute thread position
- `threadgroup_position_in_grid` — `uint3`, which threadgroup
- `thread_position_in_threadgroup` — `uint3`, position within threadgroup
- `threads_per_threadgroup` — `uint3`
- `grid_size` — `uint3`

## Key Patterns Discovered

### 1. Shape Access
```metal
uint nr = state_shape[0];   // First dimension
uint nz = state_shape[1];   // Second dimension
```

### 2. Row-Major Indexing for (nvar, nr, nz)
```metal
uint stride = nr * nz;
uint idx = r * nz + z;
float val = state[var * stride + idx];
```

### 3. Bounds Checking
```metal
if (r >= nr || z >= nz) return;  // Essential for non-power-of-2 grids
```

### 4. Scalar Parameters via Array Input
Pass scalars as 1-element arrays:
```python
gamma_param = mx.array([5.0/3.0], dtype=mx.float32)
# In MSL: float gamma = gamma_param[0];
```

### 5. No Template Required for float32
Templates are optional. For pure float32 kernels, pass `template=[]`.

### 6. Grid/Threadgroup Sizing
Grid must be >= total work items. Threadgroup is threads per group.
```python
tg = (32, 8, 1)  # 256 threads per group (good for M3 Pro)
grid = (ceil(nr/32)*32, ceil(nz/8)*8, 1)  # Round up
```

## Limitations

1. **No shared memory syntax** — threadgroup memory must be declared in MSL source directly
2. **No printf/debugging** — use `verbose=True` to inspect generated code
3. **Row-contiguous inputs only** (default) — non-contiguous arrays are copied
4. **No dynamic shapes in source** — use shape arrays for dimension queries
5. **Single compilation per kernel** — JIT compiled on first call, cached after

## Thread Group Recommendations (M3 Pro, 14 GPU cores)

| Grid Size | Threadgroup | Rationale |
|-----------|-------------|-----------|
| Small (<64x64) | (16, 16, 1) = 256 | Full occupancy per core |
| Medium (128x256) | (32, 8, 1) = 256 | Favor radial dimension for coalescing |
| Large (512x1024) | (32, 8, 1) = 256 | Same; Metal handles group scheduling |
| 1D (N,) | (256, 1, 1) | Standard 1D |

M3 Pro GPU: max 1024 threads per threadgroup, 32KB threadgroup memory, 14 cores.
Optimal occupancy at 256-512 threads/group for compute-bound kernels.

## Comparison with PyTorch MPS

| Feature | MLX metal_kernel | PyTorch MPS |
|---------|-----------------|-------------|
| Custom MSL code | Direct | Not possible |
| Kernel launch overhead | ~5 us | ~50 us |
| Memory model | Unified (zero-copy from numpy) | Copy to GPU-visible |
| float64 | Not available | Not available |
| Debugging | verbose=True | None |
| Compilation | JIT, cached | N/A (uses built-in ops) |
