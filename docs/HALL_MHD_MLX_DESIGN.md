# Hall MHD — MLX Backend Port Design

Status: DESIGN ONLY — do not implement without review.
Date: 2026-03-26

## 1. Current State Assessment

**Hall MHD already exists in the MLX backend** — `mlx_sources.py:280-345` implements
`apply_hall_mhd()` for cylindrical coordinates, and `mlx_solver.py:715-724` wires it
into the solver step at position 6.6 (after Braginskii viscosity, before PIC feedback).

This document covers the **gaps** between the MLX and Metal (PyTorch) implementations,
not a from-scratch port.

### What MLX has (mlx_sources.py)
| Function | Lines | Purpose |
|----------|-------|---------|
| `compute_current_density_components` | 233-277 | J = curl(B)/mu_0, cylindrical, central diff |
| `apply_hall_mhd` | 280-345 | E_H = (JxB)/(n_e*e), Faraday update, cylindrical |

### What Metal has that MLX lacks
| Feature | Metal location | Gap severity |
|---------|---------------|-------------|
| Cartesian 3D curl_B | `metal_transport.py:48-85` | Low — MLX is cylindrical-only |
| Cartesian 3D apply_hall | `metal_transport.py:153-215` | Low — MLX is cylindrical-only |
| HL-to-SI unit conversion | `metal_transport.py:195-211` | **Critical** |
| NaN sanitization | `metal_transport.py:214` | Medium |
| Whistler CFL constraint | `metal_solver.py:523-541` | **Critical** |
| Nernst advection | `metal_transport.py:636-731` | Future scope |
| Hall CFL sub-cycling | Not in either backend | Needed |
| Boundary stencil corrections | `metal_transport.py` uses torch.gradient | Medium |

## 2. Critical Bug: Missing HL-to-SI Unit Conversion

The MLX `apply_hall_mhd` in `mlx_sources.py` operates directly on code-unit B-fields
**without converting to SI**. The Metal version explicitly converts:

```python
# Metal (correct):
sqrt_mu0 = math.sqrt(MU_0)
B_si = B * sqrt_mu0                          # HL -> SI
J = curl_B_mps(B_si, dx, dy, dz, mu_0=MU_0) # J in A/m^2
E_Hall = hall_electric_field_mps(J, B_si, rho, ion_mass)
dB_si = -dt * curl_E
B_new = B + dB_si / sqrt_mu0                 # SI -> HL
```

```python
# MLX (current — WRONG for SI constants):
Jr, Jz, Jt = compute_current_density_components(U, dr, dz, r_cell)
# compute_current_density_components divides by MU_0 at line 276-277
# but the input B is in HL units (mu_0=1), so J = curl(B_HL)/MU_0
# when it should be J = curl(B_SI)/MU_0 = curl(B_HL * sqrt(MU_0))/MU_0
```

The `compute_current_density_components` function divides by `_MU0` (line 277),
but operates on HL-unit B-fields. This means J is off by a factor of `sqrt(MU_0)`,
and E_Hall is off by `sqrt(MU_0)` as well. The two errors partially cancel in the
induction equation (curl(E_Hall) updates B), but the magnitude is still wrong by
`MU_0` overall.

**Fix**: Either (a) convert B to SI before calling `compute_current_density_components`,
or (b) keep everything in HL units where mu_0=1 and remove the `/MU_0` division.
Option (b) is cleaner for the MLX solver since it already uses HL units throughout.

### Proposed HL-native approach (option b)

In HL units, mu_0 = 1. The Hall E-field becomes:
```
J_HL = curl(B_HL)                        # no mu_0 division
E_Hall_HL = (J_HL x B_HL) / (n_e * e)   # same cross product
dB_HL/dt = -curl(E_Hall_HL)              # Faraday
```

But `n_e * e` uses SI values. Since `B_HL = B_SI / sqrt(mu_0)`:
```
E_Hall_HL = curl(B_HL) x B_HL / (n_e * e)
          = [curl(B_SI)/sqrt(mu_0)] x [B_SI/sqrt(mu_0)] / (n_e * e)
          = curl(B_SI) x B_SI / (mu_0 * n_e * e)
```

This is NOT the correct SI Hall field `(J x B)/(n_e * e) = curl(B) x B / (mu_0 * n_e * e)`.
Wait — it is exactly that. So HL-native works if we:
1. Compute `J_HL = curl(B_HL)` (no mu_0 factor)
2. `E_H = (J_HL x B_HL) / (n_e * e)` — note the implicit mu_0 is absorbed
3. But the Faraday update `dB_HL = -curl(E_H) * dt` needs dimensional consistency

The correct HL approach requires `E_H = (J_HL x B_HL) * mu_0 / (n_e * e)` to
account for the HL-to-SI conversion. **This factor of mu_0 is currently missing.**

## 3. Function Inventory: metal_transport.py Hall Functions

| Function | Signature | torch API calls | MLX equivalent |
|----------|-----------|----------------|----------------|
| `_safe_gradient` | `(field, dim, spacing) -> Tensor` | `torch.gradient`, `torch.zeros_like` | `mx.roll` central diff (already in mlx_sources.py) |
| `curl_B_mps` | `(B, dx, dy, dz, mu_0) -> Tensor` | `torch.gradient` x6, `torch.zeros_like` | `compute_current_density_components` (cylindrical) |
| `_cross_product` | `(a, b) -> Tensor` | `torch.zeros_like`, element ops | Inline in `apply_hall_mhd` already |
| `_curl_field` | `(F, dx, dy, dz) -> Tensor` | `torch.gradient` x6 | `mx.roll` central diff (already in mlx_sources.py) |
| `hall_electric_field_mps` | `(J, B, rho, ion_mass, e_charge) -> Tensor` | `torch.clamp`, div, `_cross_product` | Inline in `apply_hall_mhd` already |
| `apply_hall_mhd_mps` | `(B, rho, dt, dx, dy, dz, ion_mass) -> Tensor` | All above + `math.sqrt`, `torch.isfinite`, `torch.where` | `apply_hall_mhd` exists but has unit bug |

## 4. torch-to-MLX Translation Table

| torch operation | MLX equivalent | Notes |
|----------------|----------------|-------|
| `torch.gradient(f, dim=d, spacing=s)[0]` | `(mx.roll(f,-1,axis=d) - mx.roll(f,1,axis=d)) / (2*s)` | MLX has no `gradient`; use roll-based central diff |
| `torch.roll(t, n, dims=d)` | `mx.roll(a, n, axis=d)` | Direct equivalent |
| `torch.clamp(t, min=v)` | `mx.maximum(a, v)` | MLX uses `maximum`/`minimum` |
| `torch.clamp(t, max=v)` | `mx.minimum(a, v)` | Same |
| `torch.where(cond, a, b)` | `mx.where(cond, a, b)` | Direct equivalent |
| `torch.isfinite(t)` | `mx.isnan(a)` negated; or `(a == a) & (mx.abs(a) < mx.inf)` | MLX lacks `isfinite`; use composite |
| `torch.zeros_like(t)` | `mx.zeros_like(a)` | Direct equivalent |
| `t.unsqueeze(0)` | `a[None]` or `mx.expand_dims(a, 0)` | Indexing preferred |
| `torch.sqrt(t)` | `mx.sqrt(a)` | Direct equivalent |
| `float(t.max().item())` | `float(mx.max(a).item())` | Need `mx.eval()` before `.item()` |
| `t.clone()` | `mx.array(a)` | MLX arrays are immutable; reassign |
| Device management (`t.to('mps')`) | N/A | MLX has no device concept; all on Metal |

## 5. Float32 Risk Analysis

### DPF Pinch Conditions
- B ~ 50 T, rho ~ 1e-3 kg/m^3 (deuterium, n_e ~ 3e23 m^-3)
- J ~ curl(B)/mu_0 ~ B/(mu_0 * dx) ~ 50/(1.26e-6 * 1e-3) ~ 4e10 A/m^2
- E_Hall = |J x B| / (n_e * e) ~ 4e10 * 50 / (3e23 * 1.6e-19) ~ 4.2e4 V/m

### Float32 range check
| Quantity | Magnitude | float32 range? | Risk |
|----------|-----------|---------------|------|
| B (HL units) | ~4.5e4 (= 50T / sqrt(mu_0)) | OK (max 3.4e38) | None |
| J (SI) | ~4e10 A/m^2 | OK | None |
| J (HL, no mu_0 div) | ~4e10 * sqrt(mu_0) ~ 4.5e7 | OK | None |
| n_e * e | 3e23 * 1.6e-19 = 4.8e4 | OK | None |
| 1/(n_e * e) | 2.1e-5 | OK | None |
| E_Hall | ~4.2e4 V/m | OK | None |
| curl(E_Hall) | ~E/dx ~ 4.2e7 | OK | None |
| dB/dt * dt | ~4.2e7 * 1e-10 ~ 4.2e-3 | OK | None |

### Cancellation risks
1. **J x B cross product**: Each term is ~O(J*B) ~ 2e15. The cross product
   subtracts two such terms. If J is nearly parallel to B, cancellation occurs.
   In DPF, J is primarily toroidal while B has all 3 components, so the cross
   product is well-conditioned. **Risk: LOW**.

2. **curl(E_Hall) via central differences**: Subtracts E at adjacent cells.
   If E varies smoothly, the difference is small relative to E. At shocks,
   E jumps discontinuously, so no cancellation. **Risk: LOW**.

3. **1/(n_e * e) at vacuum cells**: n_e -> 0 gives 1/0. Current code uses
   `mx.maximum(rho, 1e-12)` floor. At rho=1e-12, n_e = 3e14, and
   1/(n_e*e) = 2.1e4. E_Hall then scales as J*B*2.1e4, which for
   vacuum-level J,B is small. **Risk: LOW with existing floor**.

4. **Accumulation over many steps**: Hall term is O(dt * curl(JxB)/(n_e*e)).
   With dt ~ 1e-10 s and 1e5 steps, accumulated error ~ 1e5 * eps_float32 *
   |dB| ~ 1e5 * 1.2e-7 * 4e-3 = 5e-9. Negligible. **Risk: NONE**.

**Conclusion**: Float32 is safe for Hall MHD at DPF conditions. No float64
fallback needed for the Hall term itself.

## 6. Integration Point

The existing integration at `mlx_solver.py:715-724` is correct in placement:
operator-split after ideal MHD RHS (step 6.0-6.4), after resistive diffusion (6.5),
after Braginskii viscosity (6.55), and before PIC feedback (6.65).

The Hall term should remain operator-split (not embedded in `mhd_rhs`) because:
1. The whistler CFL is much more restrictive than ideal MHD CFL — sub-cycling needed
2. Operator splitting allows independent testing
3. Consistent with Athena++ approach (`field/field_diffusion/`)

### Sub-cycling requirement

Whistler CFL: `dt_hall = CFL * dx^2 * n_e * e * mu_0 / |B|`

At pinch (B=50T, n_e=3e23, dx=1e-3):
```
dt_hall = 0.4 * (1e-3)^2 * 3e23 * 1.6e-19 * 1.26e-6 / 50
        = 0.4 * 1e-6 * 6.05e-2
        = 2.4e-8 s
```

Ideal MHD CFL with c_f ~ 1e5 m/s, dx=1e-3: `dt_mhd = 0.4 * 1e-3 / 1e5 = 4e-9 s`.

At pinch, dt_hall > dt_mhd, so no sub-cycling needed there. But in the pre-pinch
rundown (lower n_e ~ 1e20, B ~ 1T, dx ~ 5e-3):
```
dt_hall = 0.4 * (5e-3)^2 * 1e20 * 1.6e-19 * 1.26e-6 / 1
        = 0.4 * 2.5e-5 * 2.02e-5
        = 2.0e-10 s
dt_mhd  = 0.4 * 5e-3 / 1e4 = 2.0e-7 s
```

Here dt_hall << dt_mhd by 1000x. **Sub-cycling is mandatory** in early phases.

## 7. Implementation Phases

### Phase H1: Fix Unit Conversion Bug (~30 LOC, 1 hour)

**File**: `src/dpf/metal/mlx_sources.py`

Fix `apply_hall_mhd` to use consistent HL units. Two options:

**Option A** (recommended): HL-native, add mu_0 correction factor.
```python
def apply_hall_mhd(
    U: mx.array,
    dt: float,
    dr: float,
    dz: float,
    r_cell: mx.array,
    ion_mass: float = 3.3435e-27,
) -> mx.array:
    _E_CHARGE = 1.602176634e-19
    _MU0 = 4.0 * 3.141592653589793 * 1e-7

    rho = mx.maximum(U[IDN], 1e-12)
    ne = rho / ion_mass

    # J_HL = curl(B_HL) — no mu_0 division in HL units
    Jr, Jz, Jt = _curl_B_hl_cylindrical(U, dr, dz, r_cell)

    Br, Bz, Bt = U[IBR], U[IBZ], U[IBT]

    # E_Hall in HL: multiply by mu_0 to convert (J_HL x B_HL) to correct SI-consistent field
    inv_ne_e = _MU0 / (ne * _E_CHARGE)
    E_r = (Jz * Bt - Jt * Bz) * inv_ne_e
    E_z = (Jt * Br - Jr * Bt) * inv_ne_e
    E_t = (Jr * Bz - Jz * Br) * inv_ne_e

    # Faraday: dB_HL/dt = -curl_cyl(E_Hall)
    # ... (existing curl code unchanged)
```

Where `_curl_B_hl_cylindrical` is the existing stencil from
`compute_current_density_components` but **without** the `/ _MU0` division.

**Option B**: Convert to SI internally (matches Metal approach).
```python
_SQRT_MU0 = math.sqrt(4.0 * math.pi * 1e-7)
B_si_r = U[IBR] * _SQRT_MU0  # etc.
# Then use existing compute_current_density_components (which divides by MU_0)
# Result dB_si -> dB_hl = dB_si / _SQRT_MU0
```

Option A avoids redundant multiply/divide and keeps everything in HL.

### Phase H2: Add Whistler CFL + Sub-cycling (~60 LOC, 2 hours)

**File**: `src/dpf/metal/mlx_sources.py`

```python
def hall_whistler_dt(
    U: mx.array,
    dr: float,
    dz: float,
    ion_mass: float = 3.3435e-27,
    cfl: float = 0.4,
) -> float:
    """Compute Hall whistler CFL timestep constraint.

    dt_hall = cfl * dx_min^2 * mu_0 * n_e_min * e / B_max

    Args:
        U: Conserved state (NVAR, nr, nz).
        dr, dz: Cell spacings [m].
        ion_mass: Ion mass [kg].
        cfl: CFL number.

    Returns:
        Maximum stable timestep for Hall term [s].
    """
    _MU0 = 4.0 * 3.141592653589793 * 1e-7
    _E_CHARGE = 1.602176634e-19
    _SQRT_MU0 = float((_MU0) ** 0.5)

    rho = mx.maximum(U[IDN], 1e-12)
    ne = rho / ion_mass
    B2 = U[IBR]**2 + U[IBZ]**2 + U[IBT]**2
    # B_HL -> B_SI: B_SI = B_HL * sqrt(mu_0)
    B_max_hl = float(mx.sqrt(mx.max(B2)).item())
    B_max_si = B_max_hl * _SQRT_MU0
    ne_min = float(mx.max(ne).item())  # use max(ne) for most restrictive

    dx_min = min(dr, dz)
    if B_max_si < 1e-30 or ne_min < 1e-10:
        return 1e10  # Hall term negligible

    # v_whistler = B / (mu_0 * n_e * e * dx)
    v_hall = B_max_si / (_MU0 * ne_min * _E_CHARGE * dx_min)
    return cfl * dx_min / max(v_hall, 1e-30)
```

**File**: `src/dpf/metal/mlx_sources.py` — modify `apply_hall_mhd`:

```python
def apply_hall_mhd(
    U: mx.array,
    dt: float,
    dr: float,
    dz: float,
    r_cell: mx.array,
    ion_mass: float = 3.3435e-27,
    max_subcycles: int = 20,
) -> mx.array:
    dt_hall = hall_whistler_dt(U, dr, dz, ion_mass)
    n_sub = max(1, min(int(math.ceil(dt / dt_hall)), max_subcycles))
    dt_sub = dt / n_sub

    for _ in range(n_sub):
        U = _hall_substep(U, dt_sub, dr, dz, r_cell, ion_mass)
        mx.eval(U)
    return U
```

### Phase H3: Add NaN Guard + Sanitization (~15 LOC, 30 min)

**File**: `src/dpf/metal/mlx_sources.py`

```python
# After computing dBr, dBz, dBt:
dBr = mx.where(mx.isnan(dBr), mx.zeros_like(dBr), dBr)
dBz = mx.where(mx.isnan(dBz), mx.zeros_like(dBz), dBz)
dBt = mx.where(mx.isnan(dBt), mx.zeros_like(dBt), dBt)

# Clamp to prevent runaway:
max_dB = 0.1 * mx.sqrt(U[IBR]**2 + U[IBZ]**2 + U[IBT]**2 + 1e-30)
dBr = mx.clip(dBr, -max_dB, max_dB)
dBz = mx.clip(dBz, -max_dB, max_dB)
dBt = mx.clip(dBt, -max_dB, max_dB)
```

### Phase H4: Wire Whistler CFL into compute_dt (~10 LOC, 15 min)

**File**: `src/dpf/metal/mlx_solver.py`

In `compute_dt` method, after ideal MHD CFL:
```python
if self.enable_hall:
    from dpf.metal.mlx_sources import hall_whistler_dt
    dt_hall = hall_whistler_dt(U, self._grid.dr, self._grid.dz, self.ion_mass, self.cfl)
    dt = min(dt, dt_hall)
```

### Phase H5: Validation Tests (~120 LOC, 3 hours)

**File**: `tests/test_mlx_hall.py`

```python
class TestHallMHD:
    def test_hall_unit_conversion_consistency(self):
        """Verify HL-native Hall matches SI-converted result."""

    def test_hall_modifies_b(self):
        """Hall term with nonzero J produces dB != 0."""

    def test_hall_zero_b_noop(self):
        """Zero B-field -> zero Hall E-field -> no update."""

    def test_hall_uniform_b_noop(self):
        """Uniform B -> curl(B)=0 -> J=0 -> no Hall effect."""

    def test_hall_whistler_cfl(self):
        """Whistler CFL < MHD CFL at low density."""

    def test_hall_subcycling_count(self):
        """Sub-cycle count matches dt/dt_hall ratio."""

    def test_hall_nan_sanitization(self):
        """NaN in vacuum cells does not propagate."""

    def test_whistler_wave_dispersion(self):
        """Whistler wave: omega = k^2 * B / (mu_0 * n_e * e).
        Initialize sinusoidal Bz perturbation, evolve, check phase speed
        matches analytical prediction within 10%."""

    def test_hall_drift_uniform_b(self):
        """Uniform B with gradient in n_e: Hall drift v_d = J/(n_e*e).
        Check B-field evolution matches expected drift direction."""

    def test_cross_backend_parity(self):
        """Metal apply_hall_mhd_mps vs MLX apply_hall_mhd on same IC.
        L1(dB) < 1e-3 relative."""
```

## 8. LOC Estimates and Dependencies

| Phase | LOC (new/modified) | Dependencies | Risk |
|-------|-------------------|-------------|------|
| H1: Unit fix | ~30 modified | None | Low — well-understood math |
| H2: CFL + subcycling | ~60 new | H1 | Medium — subcycle count tuning |
| H3: NaN guard | ~15 new | H1 | Low |
| H4: Wire CFL | ~10 modified | H2 | Low |
| H5: Tests | ~120 new | H1-H4 | Low |
| **Total** | **~235 LOC** | | |

Existing `apply_hall_mhd` is 65 LOC. Most of it survives; the changes are
surgical (add mu_0 factor, wrap in subcycle loop, add guards).

## 9. Risk Assessment

| Risk | Severity | Mitigation |
|------|----------|-----------|
| Unit conversion bug already affects production runs | High — Hall results wrong by factor of ~1.26e-6 | Fix in H1; existing tests may not catch because they test "Hall modifies B" not "by how much" |
| Whistler CFL kills performance in early rundown | Medium — 1000x subcycles | Cap at 20 subcycles (existing pattern in resistive diffusion); warn if capped |
| Sub-cycling accumulates float32 error | Low — analysis in section 5 shows negligible | Monitor energy conservation in test_whistler_wave_dispersion |
| MLX `mx.roll` boundary artifacts | Medium — roll wraps around | Existing pattern in `compute_current_density`; boundaries already handled |
| Hall + resistive diffusion interaction | Low — both operator-split | Order: resistive first (implicit), Hall second (explicit subcycled) |

## 10. Out of Scope (Future)

- Nernst advection for MLX (metal_transport.py:636-731, ~100 LOC)
- Cartesian 3D Hall (MLX solver is cylindrical-primary)
- Hall term in energy equation (standard approximation: Hall only modifies B)
- Ambipolar diffusion
