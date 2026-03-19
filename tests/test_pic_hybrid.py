"""Tests for the hybrid PIC module (src/dpf/experimental/pic/hybrid.py).

First test coverage for the PIC module — Boris pusher, CIC deposition,
beam injection, Coulomb scattering, reflecting BCs, and full-step integration.

Units: SI throughout (m, s, kg, C, V, T).
"""

from __future__ import annotations

import numpy as np
import pytest

from dpf.constants import e as e_charge
from dpf.constants import m_d
from dpf.experimental.pic.hybrid import (
    HybridPIC,
    boris_push,
    deposit_current,
    deposit_density,
)

# Deuterium ion constants (singly ionised)
M_D = m_d
Q_D = e_charge


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_pic(n: int = 16) -> HybridPIC:
    """Minimal HybridPIC on an n³ grid with 1 cm cell spacing."""
    dx = 0.01
    return HybridPIC(grid_shape=(n, n, n), dx=dx, dy=dx, dz=dx, dt=1e-9)


def _uniform_vector_field(shape: tuple[int, int, int], vec: np.ndarray) -> np.ndarray:
    """Return (nx, ny, nz, 3) field with constant value vec at every cell."""
    field = np.zeros((*shape, 3), dtype=np.float64)
    field[..., :] = np.asarray(vec, dtype=np.float64)
    return field


# ---------------------------------------------------------------------------
# 1. Boris push — uniform B, check Larmor radius
# ---------------------------------------------------------------------------

def test_boris_push_uniform_B() -> None:
    """Particle in uniform Bz should gyrate at r_L = m*v_perp / (q*B)."""
    B_mag = 1.0      # T
    v_perp = 1.0e5   # m/s

    pos = np.array([[0.0, 0.0, 0.0]])
    vel = np.array([[v_perp, 0.0, 0.0]])
    E = np.zeros((1, 3))
    B = np.array([[0.0, 0.0, B_mag]])

    r_L_expected = M_D * v_perp / (Q_D * B_mag)
    omega_c = Q_D * B_mag / M_D
    T_gyro = 2.0 * np.pi / omega_c
    dt = T_gyro / 1000          # 1000 steps per gyration

    # Run one full gyration.
    # The guiding centre is at (0, r_L, 0), so max distance from origin = 2*r_L.
    max_r = 0.0
    for _ in range(1000):
        pos, vel = boris_push(pos, vel, E, B, Q_D, M_D, dt)
        r = np.sqrt(pos[0, 0] ** 2 + pos[0, 1] ** 2)
        max_r = max(max_r, r)

    assert abs(max_r - 2.0 * r_L_expected) / r_L_expected < 0.05


# ---------------------------------------------------------------------------
# 2. Boris push — uniform E, check final velocity
# ---------------------------------------------------------------------------

def test_boris_push_uniform_E() -> None:
    """Particle in uniform Ex should gain v = v0 + (q/m)*E*dt after one step."""
    E_mag = 1.0e4   # V/m
    dt = 1.0e-10    # s

    pos = np.array([[0.0, 0.0, 0.0]])
    vel = np.array([[0.0, 0.0, 0.0]])
    E = np.array([[E_mag, 0.0, 0.0]])
    B = np.zeros((1, 3))

    _, new_vel = boris_push(pos, vel, E, B, Q_D, M_D, dt)

    v_expected = Q_D * E_mag * dt / M_D
    assert abs(new_vel[0, 0] - v_expected) / v_expected < 1e-6


# ---------------------------------------------------------------------------
# 3. CIC density — particle count conservation
# ---------------------------------------------------------------------------

def test_cic_density_conservation() -> None:
    """sum(density * cell_volume) must equal N * weight exactly."""
    rng = np.random.default_rng(42)
    nx, ny, nz = 8, 8, 8
    dx = dy = dz = 0.01
    N = 500
    weight = 1.0e12

    positions = rng.uniform(0.0, nx * dx, (N, 3))
    weights = np.full(N, weight)

    density = deposit_density(positions, weights, (nx, ny, nz), dx, dy, dz)
    total = np.sum(density) * dx * dy * dz

    assert abs(total - N * weight) / (N * weight) < 1e-10


# ---------------------------------------------------------------------------
# 4. CIC current — single particle with known velocity
# ---------------------------------------------------------------------------

def test_cic_current_density() -> None:
    """Single particle at a grid node deposits J_x = q*w*vx / V_cell at that node."""
    nx, ny, nz = 4, 4, 4
    dx = dy = dz = 0.01
    V_cell = dx * dy * dz

    # Particle at node (2, 2, 2) — fractional offset exactly 0
    cx, cy, cz = 2, 2, 2
    positions = np.array([[cx * dx, cy * dy, cz * dz]])
    vx = 1.0e6
    velocities = np.array([[vx, 0.0, 0.0]])
    weights = np.array([1.0])

    Jx, Jy, Jz = deposit_current(
        positions, velocities, weights, Q_D, (nx, ny, nz), dx, dy, dz
    )

    J_expected = Q_D * 1.0 * vx / V_cell
    assert abs(Jx[cx, cy, cz] - J_expected) / J_expected < 1e-10
    assert abs(np.sum(np.abs(Jy))) < 1e-30
    assert abs(np.sum(np.abs(Jz))) < 1e-30


# ---------------------------------------------------------------------------
# 5. inject_beam — particle count
# ---------------------------------------------------------------------------

def test_inject_beam_count() -> None:
    """inject_beam(n_beam=1000) must add exactly 1000 macro-particles."""
    pic = _make_pic()
    pic.add_species("d+", M_D, Q_D, np.zeros((0, 3)), np.zeros((0, 3)), np.zeros(0))

    n_beam = 1000
    pic.inject_beam(
        species_idx=0,
        n_beam=n_beam,
        energy_eV=100e3,
        direction=[0.0, 0.0, 1.0],
        position=[0.05, 0.05, 0.01],
    )

    assert pic.species[0].n_particles() == n_beam


# ---------------------------------------------------------------------------
# 6. inject_beam — weight check
# ---------------------------------------------------------------------------

def test_inject_beam_weight() -> None:
    """Injected macro-particles must have physical (non-trivial) weights.

    inject_beam uses weight_total / n_beam so each macro-particle represents
    many real ions — weights must be >> 1 and equal across the beam.
    """
    n_beam = 50
    weight_total = 1e16
    pic = _make_pic()
    pic.add_species("d+", M_D, Q_D, np.zeros((0, 3)), np.zeros((0, 3)), np.zeros(0))
    pic.inject_beam(
        0, n_beam, 100e3, [0.0, 0.0, 1.0], [0.05, 0.05, 0.01],
        weight_total=weight_total,
    )

    sp = pic.species[0]
    expected_weight = weight_total / n_beam

    assert np.all(sp.weights > 1.0), "weights must be physical (> 1.0)"
    assert np.all(np.isfinite(sp.weights)), "weights must be finite"
    assert np.allclose(sp.weights, expected_weight), (
        f"expected weight {expected_weight}, got {sp.weights[0]}"
    )


# ---------------------------------------------------------------------------
# 7. Coulomb scattering — speed preservation
# ---------------------------------------------------------------------------

def test_coulomb_scatter_preserves_speed() -> None:
    """Elastic Takizuka-Abe scatter changes direction but preserves |v|."""
    rng = np.random.default_rng(0)
    pic = _make_pic()
    pic.enable_collisions(n_background=1.0e24, T_background_eV=200.0)

    N = 200
    v0 = 1.0e7
    velocities = rng.normal(0.0, v0 / np.sqrt(3.0), (N, 3))
    positions = rng.uniform(0.0, 0.16, (N, 3))
    weights = np.ones(N)

    pic.add_species("d+", M_D, Q_D, positions, velocities, weights)

    E = _uniform_vector_field(pic.grid_shape, [0.0, 0.0, 0.0])
    B = _uniform_vector_field(pic.grid_shape, [0.0, 0.0, 0.0])

    speeds_before = np.linalg.norm(pic.species[0].velocities, axis=1)
    pic.push_particles(E, B)
    speeds_after = np.linalg.norm(pic.species[0].velocities, axis=1)

    rel_err = np.abs(speeds_after - speeds_before) / (speeds_before + 1e-30)
    assert np.all(rel_err < 0.05), f"max relative speed change = {rel_err.max():.3e}"


# ---------------------------------------------------------------------------
# 8. Full step — no NaN on 16³ grid over 10 steps
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_full_step_no_crash() -> None:
    """16³ grid, 200-particle beam, 10 Boris steps with uniform B — no NaN or Inf."""
    pic = _make_pic(16)
    pic.add_species("d+", M_D, Q_D, np.zeros((0, 3)), np.zeros((0, 3)), np.zeros(0))
    pic.inject_beam(
        species_idx=0,
        n_beam=200,
        energy_eV=50e3,
        direction=[0.0, 0.0, 1.0],
        position=[0.08, 0.08, 0.01],
        spread=0.1,
    )

    E = _uniform_vector_field(pic.grid_shape, [0.0, 0.0, 0.0])
    B = _uniform_vector_field(pic.grid_shape, [0.0, 0.0, 0.5])

    for _ in range(10):
        pic.push_particles(E, B)

    sp = pic.species[0]
    assert not np.any(np.isnan(sp.positions)), "NaN in positions"
    assert not np.any(np.isnan(sp.velocities)), "NaN in velocities"
    assert not np.any(np.isinf(sp.velocities)), "Inf in velocities"


# ---------------------------------------------------------------------------
# 9. Deposit feedback — J_kin is nonzero with correct sign
# ---------------------------------------------------------------------------

def test_deposit_feedback() -> None:
    """Beam moving in +z deposits positive J_z; rho must be non-negative."""
    pic = _make_pic(8)
    pic.add_species("d+", M_D, Q_D, np.zeros((0, 3)), np.zeros((0, 3)), np.zeros(0))
    pic.inject_beam(
        species_idx=0,
        n_beam=100,
        energy_eV=100e3,
        direction=[0.0, 0.0, 1.0],
        position=[0.04, 0.04, 0.04],
    )

    rho_grid, Jx, Jy, Jz = pic.deposit()

    assert np.any(Jz != 0.0), "J_z is zero — deposition not working"
    assert np.sum(Jz) > 0.0, "J_z wrong sign for +z beam"
    assert np.all(rho_grid >= 0.0), "rho must be non-negative"


# ---------------------------------------------------------------------------
# 10. Reflecting BC — particle at boundary reflects, stays in domain
# ---------------------------------------------------------------------------

def test_reflecting_bc() -> None:
    """Particle aimed at the lower x-boundary reflects and stays inside the domain."""
    pic = _make_pic(8)
    dx = 0.01
    Lx = 8 * dx

    positions = np.array([[0.005, 0.04, 0.04]])
    velocities = np.array([[-1.0e6, 0.0, 0.0]])
    weights = np.ones(1)
    pic.add_species("d+", M_D, Q_D, positions, velocities, weights)

    E = _uniform_vector_field(pic.grid_shape, [0.0, 0.0, 0.0])
    B = _uniform_vector_field(pic.grid_shape, [0.0, 0.0, 0.0])

    for _ in range(20):
        pic.push_particles(E, B, dt=1.0e-10)

    sp = pic.species[0]
    assert np.all(sp.positions[:, 0] >= 0.0), "particle escaped lower x boundary"
    assert np.all(sp.positions[:, 0] <= Lx), "particle escaped upper x boundary"
    assert np.all(np.isfinite(sp.velocities)), "velocity became non-finite"
