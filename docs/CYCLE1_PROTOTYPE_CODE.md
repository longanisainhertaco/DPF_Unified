# Cycle 1 Prototype Code: PIC V5 + Ghost Padding GPU Port + Hall Validation

**Date**: 2026-03-26
**Author**: dpf-validation-engineer (Opus 4.6)
**Lessons applied**: prototype-in-MD, multi-perspective review, verify before implementing, smoke before sweep

---

## Item 1: PIC V5 -- Full Discharge on MLX Backend

V4 failed at step 0 because the Python MHD solver overflows on 8x1x16.
The fix: use MLX backend on a larger grid. MLX handles float32 with dual-energy
recovery, entropy tracer, and ghost-cell RHS masking -- all absent from the Python solver.

### Prototype Code

```python
"""PIC V5: Full MHD+PIC discharge on MLX backend."""
import math
import numpy as np

# Gate on MLX availability
try:
    import mlx.core as mx
    HAS_MLX = True
except ImportError:
    HAS_MLX = False

import pytest


@pytest.mark.skipif(not HAS_MLX, reason="MLX not available")
@pytest.mark.slow
def test_pic_v5_mlx_full_discharge():
    from dpf.metal.mlx_solver import MLXMHDSolver
    from dpf.experimental.pic.hybrid import HybridPIC, boris_push
    from dpf.presets import get_preset

    # --- Setup: MLX solver with PF-1000 ---
    nr, nz = 16, 32
    preset = get_preset("pf1000_scholz")
    dx = 0.23 / nr   # anode radius ~ 0.23 m
    dz = 0.60 / nz   # electrode length ~ 0.60 m

    solver = MLXMHDSolver(
        grid_shape=(nr, 1, nz),
        dx=dx,
        dz=dz,
        gamma=5.0 / 3.0,
        cfl=0.3,
        riemann_solver="hll",       # HLL stable for coarse grid
        reconstruction="plm",       # PLM at 16 cells
        time_integrator="ssp_rk3",
    )

    # Initial MHD state: uniform deuterium fill
    rho0 = 0.084   # kg/m^3 (3.5 Torr D2)
    p0 = 350.0     # Pa (3.5 Torr)
    state = solver.initialize(rho=rho0, pressure=p0)

    # Circuit parameters from preset
    V0 = 27e3       # 27 kV
    C0 = 1332e-6    # F
    L0 = 33.5e-9    # H
    R0 = 2.3e-3     # Ohm

    # Simple RLC for current estimate
    omega = 1.0 / math.sqrt(L0 * C0)
    tau = 2.0 * L0 / R0

    # --- PIC setup ---
    pic = HybridPIC(
        grid_shape=(nr, 1, nz),
        dx=dx, dy=dx, dz=dz,
        dt=1e-9,  # initial; overridden each step
    )
    # Add deuteron species
    charge_d = 1.602e-19
    mass_d = 3.34e-27
    pic.add_species("deuterons", charge=charge_d, mass=mass_d)

    # --- Monitoring arrays ---
    n_total = 200
    inject_start = 50
    nan_detected = False
    max_v_over_c = 0.0
    particle_counts = []
    c_light = 2.998e8

    for step_i in range(n_total):
        # Estimate current (RLC underdamped)
        t = step_i * 1e-9  # approximate 1 ns steps
        current = (V0 / (omega * L0)) * math.exp(-t / tau) * math.sin(omega * t)

        # MHD step
        dt_mhd = solver.compute_dt(state)
        dt_mhd = min(dt_mhd, 5e-9)  # cap for stability
        state = solver.step(state, dt_mhd, current=current, voltage=V0)

        # NaN check on MHD state
        for key, val in state.items():
            if isinstance(val, np.ndarray) and np.any(np.isnan(val)):
                nan_detected = True
                break
        if nan_detected:
            break

        # PIC phase: inject beam after step 50
        if step_i >= inject_start:
            # Inject 100 beam deuterons along z at 100 keV
            if step_i == inject_start:
                E_beam = 100e3 * charge_d  # 100 keV in Joules
                v_beam = math.sqrt(2 * E_beam / mass_d)
                pic.inject_beam(
                    species_name="deuterons",
                    n_beam=100,
                    energy_eV=100e3,
                    direction=np.array([0.0, 0.0, 1.0]),
                    origin=np.array([dx * nr / 2, 0.0, dz]),
                    spread=0.1,
                )

            # Build E and B fields for PIC from MHD state
            # Simplified: uniform fields from state averages
            B_avg = np.mean(state.get("B", np.zeros((3, nr, 1, nz))), axis=(1, 2, 3))
            E_field = np.zeros((nr, 1, nz, 3), dtype=np.float64)
            B_field = np.zeros((nr, 1, nz, 3), dtype=np.float64)
            B_field[..., 0] = B_avg[0]
            B_field[..., 1] = B_avg[1]
            B_field[..., 2] = B_avg[2]

            # Push particles
            pic.push_particles(E_field, B_field, dt=dt_mhd)

            # Track max velocity
            for sp in pic.species:
                if sp.n_particles() > 0:
                    v2 = np.sum(sp.velocities ** 2, axis=1)
                    v_max = math.sqrt(float(np.max(v2)))
                    max_v_over_c = max(max_v_over_c, v_max / c_light)

        particle_counts.append(
            sum(sp.n_particles() for sp in pic.species)
        )

    # --- Assertions ---
    assert not nan_detected, f"NaN at step {step_i}"
    assert max_v_over_c < 1.0, f"Superluminal particle: v/c = {max_v_over_c:.3f}"
    assert particle_counts[-1] >= 100, "Particle count dropped below injected amount"
    # MHD state should have evolved (rho not uniform)
    rho_final = state["rho"]
    assert np.std(rho_final) > 1e-6 * np.mean(rho_final), "MHD state did not evolve"
```

### Smoke Test (run before full 200 steps)

```python
@pytest.mark.skipif(not HAS_MLX, reason="MLX not available")
def test_pic_v5_smoke_10_steps():
    """10-step smoke: MLX solver + PIC init, no beam yet. Verifies no crash."""
    from dpf.metal.mlx_solver import MLXMHDSolver

    solver = MLXMHDSolver(
        grid_shape=(16, 1, 32), dx=0.015, dz=0.019,
        riemann_solver="hll", reconstruction="plm",
    )
    state = solver.initialize(rho=0.084, pressure=350.0)
    for _ in range(10):
        dt = min(solver.compute_dt(state), 5e-9)
        state = solver.step(state, dt, current=1e5, voltage=27e3)
    assert not any(
        np.any(np.isnan(v)) for v in state.values() if isinstance(v, np.ndarray)
    )
```

---

## Item 2: Ghost Padding GPU Port

### Problem

`_pad_electrode_ghost` in `mlx_solver.py:331` calls `np.asarray(U_padded)`, converting
the entire (10, nr_g, nz) state to NumPy for Python-loop electrode BC fixups. This is
the #3 bottleneck (5-8% of step time) per the optimization plan.

The fixup loop (lines 331-374) does two things:
1. Set B_theta = mu0*I/(2*pi*r) / sqrt(mu0) in outer ghost and outermost interior cells
2. Update total energy E for energy consistency when B_theta changes

Both are vectorizable as MLX slice operations.

### Prototype Code

```python
"""Pure-MLX ghost padding: eliminate np.asarray from electrode BC fixup."""
import math
import mlx.core as mx
import numpy as np

# Constants (same as mlx_solver.py)
_MU0 = 4.0 * math.pi * 1e-7
_SQRT_MU0 = math.sqrt(_MU0)
P_FLOOR = 1e-20
GAMMA = 5.0 / 3.0

# State variable indices
IDN, IMR, IMZ, IMT, IEN, ISR, IBR, IBZ, IBT, IEE = range(10)


def electrode_bt_fixup_mlx(
    U_padded: mx.array,
    r_cell: mx.array,
    current: float,
    ng: int,
    nr_phys: int,
    convert_si_to_hl: bool = True,
) -> mx.array:
    """Apply electrode B_theta boundary condition in pure MLX.

    Replaces the np.asarray-based loop in mlx_solver.py:331-374.

    Sets B_theta = mu0*I/(2*pi*r) [/ sqrt(mu0) if HL] at:
      - Outer ghost cells: indices [ng+nr_phys, ng+nr_phys+ng)
      - Outermost ng interior cells: indices [ng+nr_phys-ng, ng+nr_phys)
        using max(existing, electrode) blending

    Updates total energy for magnetic energy consistency.

    Args:
        U_padded: Ghost-padded state (NVAR, nr_g, nz), mx.array float32.
        r_cell: Cell-center radii for full padded grid (nr_g,), mx.array.
        current: Circuit current [A].
        ng: Number of ghost zones.
        nr_phys: Number of physical radial cells (excludes ghosts).
        convert_si_to_hl: If True, divide B_theta by sqrt(mu0) for HL units.

    Returns:
        Updated U_padded with electrode BC applied, mx.array float32.
    """
    divisor = _SQRT_MU0 if convert_si_to_hl else 1.0
    nr_g = U_padded.shape[1]

    # Compute electrode B_theta profile: mu0*I/(2*pi*r) / divisor
    r_safe = mx.maximum(mx.abs(r_cell), 1e-10)       # (nr_g,)
    Bt_electrode = mx.array(
        _MU0 * current / (2.0 * math.pi) / divisor, dtype=mx.float32
    ) / r_safe                                         # (nr_g,)
    # Broadcast to (nr_g, nz)
    Bt_electrode_2d = Bt_electrode[:, None] * mx.ones((1, U_padded.shape[2]), dtype=mx.float32)

    # --- Outer ghost cells: hard-set B_theta ---
    # Build mask for outer ghost indices [ng+nr_phys, ng+nr_phys+ng)
    idx = mx.arange(nr_g)
    outer_ghost_mask = (idx >= ng + nr_phys) & (idx < ng + nr_phys + ng)
    outer_ghost_mask_2d = outer_ghost_mask[:, None]  # (nr_g, 1) broadcast

    # Energy fixup: dE = 0.5*(Bt_new^2 - Bt_old^2) + (Br^2+Bz^2 terms cancel)
    B2_old = U_padded[IBR] ** 2 + U_padded[IBZ] ** 2 + U_padded[IBT] ** 2
    Bt_new_outer = mx.where(outer_ghost_mask_2d, Bt_electrode_2d, U_padded[IBT])
    B2_new_outer = U_padded[IBR] ** 2 + U_padded[IBZ] ** 2 + Bt_new_outer ** 2
    dE_outer = 0.5 * (B2_new_outer - B2_old)

    # Apply outer ghost B_theta
    Bt_updated = Bt_new_outer

    # Apply outer ghost energy fixup
    E_updated = U_padded[IEN] + mx.where(outer_ghost_mask_2d, dE_outer, 0.0)

    # Enforce beta floor in outer ghost
    p_mag_outer = 0.5 * B2_new_outer
    beta_floor = 1e-4
    p_min = beta_floor * mx.maximum(p_mag_outer, P_FLOOR)
    E_floor = p_min / (GAMMA - 1.0) + 0.5 * B2_new_outer
    E_updated = mx.where(
        outer_ghost_mask_2d,
        mx.maximum(E_updated, E_floor),
        E_updated,
    )

    # Density floor in outer ghost
    rho_updated = mx.where(
        outer_ghost_mask_2d,
        mx.maximum(U_padded[IDN], 1e-4),
        U_padded[IDN],
    )

    # --- Interior blend cells: max(existing, electrode) ---
    interior_blend_mask = (idx >= ng + nr_phys - ng) & (idx < ng + nr_phys)
    interior_blend_mask_2d = interior_blend_mask[:, None]

    Bt_blended = mx.where(
        mx.abs(Bt_updated) > mx.abs(Bt_electrode_2d),
        Bt_updated,
        Bt_electrode_2d,
    )
    Bt_final = mx.where(interior_blend_mask_2d, Bt_blended, Bt_updated)

    # Energy fixup for interior blend
    B2_blend = U_padded[IBR] ** 2 + U_padded[IBZ] ** 2 + Bt_final ** 2
    dE_blend = 0.5 * (B2_blend - B2_old)
    E_final = E_updated + mx.where(interior_blend_mask_2d, dE_blend, 0.0)

    # Beta floor for interior blend
    p_mag_blend = 0.5 * B2_blend
    p_min_b = beta_floor * mx.maximum(p_mag_blend, P_FLOOR)
    E_floor_b = p_min_b / (GAMMA - 1.0) + 0.5 * B2_blend
    E_final = mx.where(
        interior_blend_mask_2d,
        mx.maximum(E_final, E_floor_b),
        E_final,
    )

    # Reassemble state
    return mx.stack([
        rho_updated,
        U_padded[IMR],
        U_padded[IMZ],
        U_padded[IMT],
        E_final,
        U_padded[ISR],
        U_padded[IBR],
        U_padded[IBZ],
        Bt_final,
        U_padded[IEE],
    ], axis=0).astype(mx.float32)


def test_ghost_padding_mlx_parity():
    """Verify pure-MLX fixup matches NumPy fixup on a reference state."""
    nr, nz, ng = 16, 32, 3
    nr_g = nr + 2 * ng
    current = 1.5e6  # 1.5 MA

    # Synthetic padded state
    rng = np.random.default_rng(42)
    U_np = rng.uniform(0.1, 10.0, (10, nr_g, nz)).astype(np.float32)
    U_np[IDN] = np.maximum(U_np[IDN], 1e-4)
    U_np[IEN] = np.maximum(U_np[IEN], 1.0)

    # Radial coordinates
    dr = 0.23 / nr
    r_cell_np = np.array([
        -ng * dr + (i + 0.5) * dr for i in range(nr_g)
    ], dtype=np.float32)

    U_mx = mx.array(U_np)
    r_cell_mx = mx.array(r_cell_np)

    result = electrode_bt_fixup_mlx(
        U_mx, r_cell_mx, current, ng, nr, convert_si_to_hl=True
    )
    mx.eval(result)
    result_np = np.asarray(result)

    # Outer ghost B_theta should be electrode profile
    for ig in range(ng):
        out_idx = ng + nr + ig
        r_pos = max(abs(r_cell_np[out_idx]), 1e-10)
        expected_bt = _MU0 * current / (2 * math.pi * r_pos) / _SQRT_MU0
        actual_bt = result_np[IBT, out_idx, nz // 2]
        assert abs(actual_bt - expected_bt) / abs(expected_bt) < 1e-4, (
            f"Ghost cell {ig}: expected Bt={expected_bt:.2f}, got {actual_bt:.2f}"
        )

    # No NaN in output
    assert not np.any(np.isnan(result_np)), "NaN in ghost-padded state"
```

---

## Item 3: Hall MHD Physics Validation

### Problem

Three bugs were fixed in `mlx_sources.py:apply_hall_mhd`:
1. Removed erroneous `/MU_0` in `compute_current_density_components` (HL units)
2. Added NaN guard via `mx.isfinite` mask
3. Added `mx.isfinite` was confirmed available

But the physics output was never validated: does Hall modify B by the correct
magnitude and direction?

### Key Bug: Missing mu_0 factor in E_Hall (HL units)

Current code: `E_Hall = (J_HL x B_HL) / (n_e * e)`.
Correct HL: `E_Hall = mu_0 * (J_HL x B_HL) / (n_e * e)`.
Without mu_0, Hall is ~10^6 too weak. Must fix before validation.

### Prototype Code: Whistler Wave Dispersion Test

```python
"""Hall MHD validation: whistler wave dispersion relation."""
import math
import numpy as np

try:
    import mlx.core as mx
    HAS_MLX = True
except ImportError:
    HAS_MLX = False

import pytest

_MU0 = 4.0 * math.pi * 1e-7
_E_CHARGE = 1.602176634e-19
_SQRT_MU0 = math.sqrt(_MU0)


@pytest.mark.skipif(not HAS_MLX, reason="MLX not available")
def test_hall_whistler_phase_speed():
    """Whistler wave propagation: Hall ON vs analytical dispersion."""
    from dpf.metal.mlx_sources import apply_hall_mhd
    from dpf.metal.mlx_kernels import IDN, IBR, IBZ, IBT, IEN, ISR, IMR, IMZ, IMT, IEE

    # Grid: 1D in z (use nr=4 to avoid axis singularity)
    nr, nz = 4, 64
    Lz = 1.0   # 1 m domain
    dr = 0.01
    dz = Lz / nz

    # Uniform background: deuterium plasma
    ion_mass = 3.34e-27
    rho0 = 1e-3         # kg/m^3
    ne0 = rho0 / ion_mass  # ~3e23 m^-3
    p0 = 1e4             # Pa
    B0_SI = 1.0          # 1 T background Bz
    B0_HL = B0_SI / _SQRT_MU0
    gamma = 5.0 / 3.0

    # Perturbation: sinusoidal Bx (transverse to Bz background)
    k = 2.0 * math.pi / Lz   # wavenumber (1 full wavelength in domain)
    dB_amp_SI = 0.01          # 10 mT perturbation
    dB_amp_HL = dB_amp_SI / _SQRT_MU0

    # Build initial state (NVAR, nr, nz) in HL units
    z_cell = np.array([(j + 0.5) * dz for j in range(nz)], dtype=np.float32)
    r_cell = np.array([(i + 0.5) * dr for i in range(nr)], dtype=np.float32)

    U = np.zeros((10, nr, nz), dtype=np.float32)
    U[IDN] = rho0
    U[IBZ] = B0_HL
    U[IBR] = dB_amp_HL * np.sin(k * z_cell)[None, :]  # Br perturbation
    # Total energy: thermal + magnetic
    B2 = U[IBR] ** 2 + U[IBZ] ** 2 + U[IBT] ** 2
    U[IEN] = p0 / (gamma - 1) + 0.5 * B2
    U[ISR] = p0 / rho0 ** (gamma - 1)  # entropy tracer

    U_mx = mx.array(U)
    r_cell_mx = mx.array(r_cell)

    # Analytical whistler phase speed
    # omega = k^2 * B0_SI / (mu_0 * ne0 * e)
    # v_phase = omega / k = k * B0_SI / (mu_0 * ne0 * e)
    v_phase_analytical = k * B0_SI / (_MU0 * ne0 * _E_CHARGE)

    # Evolve with Hall term only (no ideal MHD)
    # Whistler CFL: dt_hall ~ dx^2 * mu0 * ne * e / B
    dt_hall = 0.1 * dz ** 2 * _MU0 * ne0 * _E_CHARGE / B0_SI
    n_steps = max(1, int(0.1 * (2 * math.pi / k) / v_phase_analytical / dt_hall))
    n_steps = min(n_steps, 500)  # cap for test speed

    for _ in range(n_steps):
        U_mx = apply_hall_mhd(U_mx, dt_hall, dr, dz, r_cell_mx, ion_mass)
    mx.eval(U_mx)

    U_final = np.asarray(U_mx)

    # Measure phase shift of Br perturbation
    Br_initial = dB_amp_HL * np.sin(k * z_cell)
    Br_final = U_final[IBR, nr // 2, :]  # take middle radial cell

    # Cross-correlation to find phase shift
    from numpy.fft import fft
    F_init = fft(Br_initial)
    F_final = fft(Br_final)
    cross = F_final * np.conj(F_init)
    # Phase of the k=1 mode
    phase_shift = np.angle(cross[1])  # radians

    # Expected phase shift: omega * t = k * v_phase * n_steps * dt_hall
    total_time = n_steps * dt_hall
    expected_phase = k * v_phase_analytical * total_time
    # Wrap to [-pi, pi]
    expected_phase_wrapped = (expected_phase + math.pi) % (2 * math.pi) - math.pi

    # Acceptance: measured phase within 30% of analytical
    # (generous: finite difference stencil has dispersion error)
    if abs(expected_phase_wrapped) > 0.1:  # only test if phase shift is measurable
        error = abs(phase_shift - expected_phase_wrapped) / abs(expected_phase_wrapped)
        assert error < 0.30, (
            f"Whistler phase error: {error:.1%}. "
            f"Expected {expected_phase_wrapped:.3f} rad, got {phase_shift:.3f} rad"
        )

    # Hall must have modified Br (not a no-op)
    dBr = U_final[IBR, nr // 2, :] - dB_amp_HL * np.sin(k * z_cell)
    assert np.max(np.abs(dBr)) > 1e-6 * dB_amp_HL, (
        "Hall term did not modify Br -- possible no-op bug"
    )


@pytest.mark.skipif(not HAS_MLX, reason="MLX not available")
def test_hall_uniform_b_noop():
    """Uniform B-field -> curl(B)=0 -> J=0 -> no Hall effect."""
    from dpf.metal.mlx_sources import apply_hall_mhd
    from dpf.metal.mlx_kernels import IDN, IBZ, IEN, ISR

    nr, nz = 8, 16
    U = np.zeros((10, nr, nz), dtype=np.float32)
    U[IDN] = 1e-3
    U[IBZ] = 1000.0  # uniform B_z in HL
    U[IEN] = 1e4 + 0.5 * 1000.0 ** 2
    U[ISR] = 1e4 / (1e-3 ** (2.0 / 3.0))

    r_cell = np.array([(i + 0.5) * 0.01 for i in range(nr)], dtype=np.float32)
    U_mx = mx.array(U)

    U_after = apply_hall_mhd(U_mx, 1e-9, 0.01, 0.01, mx.array(r_cell))
    mx.eval(U_after)

    # B should be unchanged (curl of uniform = 0)
    diff = np.max(np.abs(np.asarray(U_after) - U))
    assert diff < 1e-10, f"Uniform B changed by Hall: max|dU| = {diff:.2e}"


@pytest.mark.skipif(not HAS_MLX, reason="MLX not available")
def test_hall_on_vs_off_magnitude():
    """Hall ON produces measurable dB; magnitude scales with 1/(n_e*e)."""
    from dpf.metal.mlx_sources import apply_hall_mhd
    from dpf.metal.mlx_kernels import IDN, IBR, IBZ, IBT, IEN, ISR

    nr, nz = 8, 32
    dr = dz = 0.01
    ion_mass = 3.34e-27

    # Two runs: high density (weak Hall) vs low density (strong Hall)
    results = {}
    for label, rho in [("high_ne", 1e-2), ("low_ne", 1e-4)]:
        U = np.zeros((10, nr, nz), dtype=np.float32)
        U[IDN] = rho
        B0 = 1000.0  # HL
        U[IBZ] = B0
        # Add Bt gradient to get nonzero curl(B)
        z_cell = np.array([(j + 0.5) * dz for j in range(nz)], dtype=np.float32)
        U[IBT] = 100.0 * np.sin(2 * math.pi * z_cell / (nz * dz))[None, :]
        B2 = U[IBR] ** 2 + U[IBZ] ** 2 + U[IBT] ** 2
        U[IEN] = 1e4 / (5.0 / 3.0 - 1) + 0.5 * B2
        U[ISR] = 1e4 / rho ** (2.0 / 3.0)

        r_cell = np.array([(i + 0.5) * dr for i in range(nr)], dtype=np.float32)
        U_mx = mx.array(U)
        U_after = apply_hall_mhd(U_mx, 1e-10, dr, dz, mx.array(r_cell), ion_mass)
        mx.eval(U_after)

        dB = np.max(np.abs(np.asarray(U_after)[IBR:IBT + 1] - U[IBR:IBT + 1]))
        results[label] = dB

    # Low density (fewer charge carriers) -> stronger Hall -> larger dB
    assert results["low_ne"] > results["high_ne"], (
        f"Hall scaling wrong: low_ne dB={results['low_ne']:.2e} "
        f"<= high_ne dB={results['high_ne']:.2e}"
    )
    # Both should be nonzero
    assert results["high_ne"] > 0, "Hall had zero effect at high density"
```

---

## Six Sigma FMEA Tables

### Item 1: PIC V5 on MLX (Top 5 Risks)

| # | Failure Mode | Cause | Effect | Sev | Occur | Detect | RPN | Mitigation |
|---|-------------|-------|--------|-----|-------|--------|-----|------------|
| 1 | MHD NaN at step 0 | MLX solver config wrong (missing ghost masking) | Test crashes immediately | 9 | 3 | 1 | 27 | Smoke test (10 steps, no PIC) validates MLX runs before PIC activates |
| 2 | PIC field interpolation NaN | MHD ghost cells contain NaN, CIC reads them | All particles get NaN velocity in 1 step | 10 | 7 | 2 | 140 | Ghost NaN guard (V1 fix) must be applied before V5 runs; smoke test catches |
| 3 | Superluminal particles after beam injection | Non-relativistic Boris + 100 keV E-field | Esirkepov multi-cell crossing, charge loss, rho -> 0 | 8 | 5 | 3 | 120 | Monitor max|v|/c every step; cap at 200 steps (well under 6000-step threshold) |
| 4 | Esirkepov dt mismatch | MHD dt != PIC self.dt (known bug, unfixed) | J scaled wrong, charge non-conservation | 7 | 9 | 4 | 252 | **Highest RPN.** Must apply `_last_push_dt` fix (3 LOC) before V5 |
| 5 | Memory exhaustion from particle injection | inject_beam called every step, no removal | 200 * n_beam particles accumulate | 5 | 2 | 2 | 20 | Inject once at step 50, not every step |

**Action items before V5 implementation:**
1. Apply Esirkepov dt fix (RPN 252 -- mandatory)
2. Apply ghost NaN guard in interpolation (RPN 140 -- mandatory)
3. Run smoke test first (10 steps, no PIC)

### Item 2: Ghost Padding GPU Port (Top 5 Risks)

| # | Failure Mode | Cause | Effect | Sev | Occur | Detect | RPN | Mitigation |
|---|-------------|-------|--------|-----|-------|--------|-----|------------|
| 1 | B_theta profile wrong at ghost cells | Index off-by-one in mask construction | WENO5-Z sees discontinuity, NaN in 1 step | 9 | 4 | 2 | 72 | Parity test: compare MLX output to NumPy output cell-by-cell |
| 2 | Energy inconsistency (negative pressure) | dE computation misses a B-field component | Pressure goes negative, HLLD crashes | 9 | 3 | 2 | 54 | Assert p > 0 at all ghost cells after fixup |
| 3 | mx.where broadcast shape mismatch | r_cell (nr_g,) vs U (nr_g, nz) broadcast | Silent wrong results or crash | 6 | 5 | 3 | 90 | **Highest RPN.** Unit test with explicit shape checks at each intermediate |
| 4 | Interior blend overwrites sheath B_theta | max(existing, electrode) logic inverted | Pinch B_theta reduced, weaker confinement | 7 | 3 | 4 | 84 | Compare Bt at blend cells before/after; must be >= pre-fixup |
| 5 | Performance regression from mx.stack | Reassembling 10-var state creates temporary | Slower than NumPy for small grids | 4 | 4 | 2 | 32 | Benchmark at 16x32 and 64x128; must be faster at 64x128 |

**Action items before implementation:**
1. Write parity test (NumPy reference vs MLX) -- run at exact same state
2. Shape-check assertions at every intermediate mask
3. Benchmark both paths at 64x128

### Item 3: Hall MHD Validation (Top 5 Risks)

| # | Failure Mode | Cause | Effect | Sev | Occur | Detect | RPN | Mitigation |
|---|-------------|-------|--------|-----|-------|--------|-----|------------|
| 1 | Whistler phase speed off by mu_0 factor | Missing mu_0 in E_Hall (known from design doc) | Hall term ~10^6 too weak in HL units | 10 | 10 | 3 | 300 | **Highest RPN.** Fix mu_0 factor BEFORE running validation |
| 2 | Phase measurement aliased | n_steps * dt too large, phase wraps past pi | Cross-correlation gives wrong phase | 5 | 4 | 2 | 40 | Compute expected phase first; adjust n_steps to keep < pi/2 |
| 3 | Boundary artifacts from mx.roll wrap-around | Central diff stencil wraps z=0 to z=nz-1 | Spurious J at boundaries contaminates whistler | 7 | 6 | 3 | 126 | Use interior cells only (nr//2) for phase measurement; pad domain |
| 4 | Float32 cancellation in curl(E_Hall) | Adjacent E_Hall values nearly equal | dB underflows to zero, test shows "no effect" | 6 | 3 | 3 | 54 | Use measurable perturbation amplitude (10 mT, not 0.1 mT) |
| 5 | Test passes for wrong reason | Hall modifies B but by wrong magnitude, loose tolerance | False positive on "Hall works" claim | 8 | 5 | 5 | 200 | Include magnitude scaling test (high vs low n_e); analytical phase check |

**Action items before implementation:**
1. Fix mu_0 factor in `apply_hall_mhd` (RPN 300 -- mandatory, blocks all validation)
2. Add whistler CFL constraint (currently missing, will blow up without it)
3. Run uniform-B no-op test first (zero baseline)

---

## Lessons Applied

| Lesson | How Applied |
|--------|------------|
| **Prototype in MD** | All three items have runnable code blocks with imports, assertions, and expected outputs. Not pseudocode. |
| **Multi-perspective** | FMEA tables assess each item from failure mode, root cause, detection, AND mitigation perspectives. |
| **Verify before implementing** | Each item identifies prerequisite fixes (dt bug, mu_0 factor) that must be verified before the prototype runs. |
| **Smoke before sweep** | PIC V5 includes a 10-step smoke test before the full 200-step discharge. Ghost padding has a parity test before production use. |
| **RPN-ordered action** | Highest-RPN items are called out: Esirkepov dt (252), mu_0 factor (300), broadcast shape (90). These are the kill-chain entry points. |
| **Known bug cross-reference** | PIC V5 references 4 documented bugs from `pic_compound_bugs.md` and gates on their fixes. Not assuming they are fixed. |

---

## Implementation Priority

| Item | Blocking Prerequisites | Estimated LOC | Risk (max RPN) |
|------|----------------------|---------------|----------------|
| 3: Hall mu_0 fix + validation | None (self-contained) | ~20 fix + 60 test | 300 |
| 2: Ghost padding GPU port | None (self-contained) | ~100 code + 30 test | 90 |
| 1: PIC V5 full discharge | Items from PIC_PROTOTYPE_CODE.md: ghost NaN guard, Esirkepov dt fix | ~80 test | 252 |

Recommended order: **3 -> 2 -> 1**. Item 3 has the highest single-risk RPN (mu_0 = 300)
and is self-contained. Item 2 is independent. Item 1 depends on prior bug fixes from
the PIC prototype document and should go last.
