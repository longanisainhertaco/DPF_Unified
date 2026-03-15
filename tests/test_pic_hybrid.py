"""Tests for the hybrid PIC module (dpf.experimental.pic.hybrid).

Covers: Boris push, CIC deposition, field interpolation,
ParticleSpecies, HybridPIC driver, and detect_instability.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from dpf.constants import e as e_charge, m_d
from dpf.experimental.pic.hybrid import (
    HybridPIC,
    ParticleSpecies,
    boris_push,
    deposit_current,
    deposit_density,
    detect_instability,
    interpolate_field_to_particles,
)

# ---------------------------------------------------------------------------
# Constants for test particles (deuterium ion)
# ---------------------------------------------------------------------------

MASS = m_d
CHARGE = e_charge  # singly-ionised deuterium


# ===========================================================================
# Boris push — uniform B field (gyration)
# ===========================================================================


@pytest.mark.slow
def test_boris_push_uniform_B_produces_larmor_gyration():
    """Single deuterium ion in uniform Bz should gyrate at the Larmor radius."""
    B_z = 1.0  # T
    v_perp = 1.0e5  # m/s  (well sub-relativistic)

    # Larmor radius: r = m*v / (|q|*B)
    r_L = MASS * v_perp / (CHARGE * B_z)

    # Initial conditions: ion at origin, moving in x
    pos = np.array([[0.0, 0.0, 0.0]])
    vel = np.array([[v_perp, 0.0, 0.0]])
    E = np.zeros((1, 3))
    B = np.array([[0.0, 0.0, B_z]])

    # Gyration period: T = 2*pi*m / (|q|*B)
    T_gyro = 2.0 * math.pi * MASS / (CHARGE * B_z)
    dt = T_gyro / 1000  # 1000 steps per cycle

    # Push for one full gyration
    n_steps = 1000
    for _ in range(n_steps):
        pos, vel = boris_push(pos, vel, E, B, CHARGE, MASS, dt)

    # After one full gyration, should be back near origin
    assert abs(pos[0, 0]) == pytest.approx(0.0, abs=r_L * 0.05)
    assert abs(pos[0, 1]) == pytest.approx(0.0, abs=r_L * 0.05)
    assert pos[0, 2] == pytest.approx(0.0, abs=1e-12)


@pytest.mark.slow
def test_boris_push_uniform_B_conserves_speed():
    """Boris push in pure B field must conserve |v| (magnetic force does no work)."""
    B_z = 2.0
    v_perp = 5.0e5

    pos = np.array([[0.0, 0.0, 0.0]])
    vel = np.array([[v_perp, 0.0, 0.0]])
    E = np.zeros((1, 3))
    B = np.array([[0.0, 0.0, B_z]])

    T_gyro = 2.0 * math.pi * MASS / (CHARGE * B_z)
    dt = T_gyro / 500
    speed_initial = np.linalg.norm(vel)

    for _ in range(500):
        pos, vel = boris_push(pos, vel, E, B, CHARGE, MASS, dt)

    speed_final = np.linalg.norm(vel)
    assert speed_final == pytest.approx(speed_initial, rel=1e-6)


def test_boris_push_uniform_E_accelerates_linearly():
    """Single ion in uniform Ex should gain momentum linearly: Δv = (q/m)*E*t."""
    E_x = 1.0e4  # V/m
    dt = 1.0e-10  # s
    n_steps = 100

    pos = np.array([[0.0, 0.0, 0.0]])
    vel = np.array([[0.0, 0.0, 0.0]])
    E = np.array([[E_x, 0.0, 0.0]])
    B = np.zeros((1, 3))

    for _ in range(n_steps):
        pos, vel = boris_push(pos, vel, E, B, CHARGE, MASS, dt)

    total_time = n_steps * dt
    expected_vx = (CHARGE / MASS) * E_x * total_time

    assert vel[0, 0] == pytest.approx(expected_vx, rel=1e-6)
    assert vel[0, 1] == pytest.approx(0.0, abs=1e-12)
    assert vel[0, 2] == pytest.approx(0.0, abs=1e-12)


def test_boris_push_zero_fields_leaves_particle_unchanged():
    """No fields — particle should drift at constant velocity."""
    v0 = np.array([[3.0e4, -1.0e4, 2.0e4]])
    pos = np.array([[0.1, 0.2, 0.3]])
    E = np.zeros((1, 3))
    B = np.zeros((1, 3))
    dt = 1.0e-9

    new_pos, new_vel = boris_push(pos, v0, E, B, CHARGE, MASS, dt)

    # Velocity must be unchanged
    np.testing.assert_allclose(new_vel, v0, rtol=1e-12)
    # Position update: x = x0 + v*dt
    expected_pos = pos + v0 * dt
    np.testing.assert_allclose(new_pos, expected_pos, rtol=1e-12)


def test_boris_push_returns_correct_shapes():
    """Output shapes must match input (N, 3)."""
    n = 7
    pos = np.random.default_rng(0).random((n, 3))
    vel = np.random.default_rng(1).random((n, 3)) * 1e4
    E = np.zeros((n, 3))
    B = np.zeros((n, 3))

    new_pos, new_vel = boris_push(pos, vel, E, B, CHARGE, MASS, 1e-10)

    assert new_pos.shape == (n, 3)
    assert new_vel.shape == (n, 3)


# ===========================================================================
# CIC density deposition
# ===========================================================================


def test_deposit_density_single_particle_at_cell_center_deposits_100_percent():
    """Particle at exact cell center (0.5*dx, 0.5*dy, 0.5*dz) should deposit
    equal weight to all 8 surrounding nodes — total equals particle weight."""
    nx, ny, nz = 8, 8, 8
    dx = dy = dz = 1.0e-2
    weight = 1.0e12

    # Cell (2,2,2) center
    pos = np.array([[(2 + 0.5) * dx, (2 + 0.5) * dy, (2 + 0.5) * dz]])
    weights = np.array([weight])

    density = deposit_density(pos, weights, (nx, ny, nz), dx, dy, dz)
    cell_volume = dx * dy * dz

    # Total deposited particle count (density * volume, summed)
    total = np.sum(density) * cell_volume
    assert total == pytest.approx(weight, rel=1e-10)


def test_deposit_density_particle_at_node_concentrates_in_one_cell():
    """Particle exactly at a grid node (ix=3, iy=3, iz=3) should deposit
    only to the cell (ix, iy, iz) — fractional offset is zero."""
    nx, ny, nz = 8, 8, 8
    dx = dy = dz = 1.0e-2
    weight = 5.0e11

    ix, iy, iz = 3, 3, 3
    pos = np.array([[ix * dx, iy * dy, iz * dz]])
    weights = np.array([weight])

    density = deposit_density(pos, weights, (nx, ny, nz), dx, dy, dz)
    cell_volume = dx * dy * dz

    # Only cell (ix, iy, iz) should be non-zero (CIC clamps to [0, N-2])
    deposited_in_cell = density[ix, iy, iz] * cell_volume
    assert deposited_in_cell == pytest.approx(weight, rel=1e-10)


def test_deposit_density_particle_at_cell_edge_splits_between_neighbors():
    """Particle at (ix+0.5)*dx in x, centered in y and z, should split
    50% into cell ix and 50% into cell ix+1 along x."""
    nx, ny, nz = 8, 8, 8
    dx = dy = dz = 1.0e-2
    weight = 1.0

    ix, iy, iz = 2, 2, 2
    # Place at x-edge: fraction fx=0.5, fy=0.5, fz=0.5
    # By symmetry all 8 corners get equal 1/8 share
    # To test x-split: put fy=0, fz=0 exactly (particle on y-z face of cell ix)
    pos = np.array([[(ix + 0.5) * dx, iy * dy, iz * dz]])
    weights = np.array([weight])

    density = deposit_density(pos, weights, (nx, ny, nz), dx, dy, dz)
    cell_volume = dx * dy * dz

    # With fy=0, fz=0: contributions are:
    #  cell (ix,   iy, iz) += w*(1-0.5)*1*1 = 0.5
    #  cell (ix+1, iy, iz) += w*0.5*1*1     = 0.5
    share_lo = density[ix, iy, iz] * cell_volume
    share_hi = density[ix + 1, iy, iz] * cell_volume

    assert share_lo == pytest.approx(0.5 * weight, rel=1e-10)
    assert share_hi == pytest.approx(0.5 * weight, rel=1e-10)


def test_deposit_density_conserves_total_particle_count():
    """Sum of deposited density times cell volume equals sum of all weights."""
    rng = np.random.default_rng(42)
    nx, ny, nz = 8, 8, 8
    dx, dy, dz = 1.0e-3, 1.0e-3, 1.0e-3

    # Random particles within domain interior (avoid boundary clamping edge cases)
    n = 200
    pos = rng.uniform(0.1 * dx, (nx - 1.1) * dx, (n, 3))
    pos[:, 1] = rng.uniform(0.1 * dy, (ny - 1.1) * dy, n)
    pos[:, 2] = rng.uniform(0.1 * dz, (nz - 1.1) * dz, n)
    weights = rng.uniform(1.0e10, 1.0e12, n)

    density = deposit_density(pos, weights, (nx, ny, nz), dx, dy, dz)
    cell_volume = dx * dy * dz

    total_deposited = np.sum(density) * cell_volume
    total_weight = np.sum(weights)

    assert total_deposited == pytest.approx(total_weight, rel=1e-10)


# ===========================================================================
# CIC current deposition
# ===========================================================================


def test_deposit_current_single_particle_direction_preserved():
    """A particle moving only in x should produce non-zero Jx, zero Jy and Jz."""
    nx, ny, nz = 8, 8, 8
    dx = dy = dz = 1.0e-2

    pos = np.array([[3.5 * dx, 3.5 * dy, 3.5 * dz]])
    vel = np.array([[1.0e5, 0.0, 0.0]])
    weights = np.array([1.0])

    Jx, Jy, Jz = deposit_current(pos, vel, weights, CHARGE, (nx, ny, nz), dx, dy, dz)

    assert np.sum(np.abs(Jx)) > 0.0
    assert np.sum(np.abs(Jy)) == pytest.approx(0.0, abs=1e-30)
    assert np.sum(np.abs(Jz)) == pytest.approx(0.0, abs=1e-30)


def test_deposit_current_conserves_total_current():
    """Sum of J * cell_volume should equal q * sum(w * v) for each component."""
    rng = np.random.default_rng(7)
    nx, ny, nz = 8, 8, 8
    dx, dy, dz = 1.0e-3, 1.0e-3, 1.0e-3
    n = 100

    pos = rng.uniform(0.1 * dx, (nx - 1.1) * dx, (n, 3))
    pos[:, 1] = rng.uniform(0.1 * dy, (ny - 1.1) * dy, n)
    pos[:, 2] = rng.uniform(0.1 * dz, (nz - 1.1) * dz, n)
    vel = rng.uniform(-1e5, 1e5, (n, 3))
    weights = rng.uniform(1.0, 10.0, n)

    Jx, Jy, Jz = deposit_current(pos, vel, weights, CHARGE, (nx, ny, nz), dx, dy, dz)
    cell_volume = dx * dy * dz

    expected_x = CHARGE * np.sum(weights * vel[:, 0])
    expected_y = CHARGE * np.sum(weights * vel[:, 1])
    expected_z = CHARGE * np.sum(weights * vel[:, 2])

    assert np.sum(Jx) * cell_volume == pytest.approx(expected_x, rel=1e-10)
    assert np.sum(Jy) * cell_volume == pytest.approx(expected_y, rel=1e-10)
    assert np.sum(Jz) * cell_volume == pytest.approx(expected_z, rel=1e-10)


# ===========================================================================
# Field interpolation
# ===========================================================================


def test_interpolate_field_uniform_scalar_returns_same_value_everywhere():
    """Interpolating a uniform scalar field must return that value at any position."""
    nx, ny, nz = 8, 8, 8
    dx = dy = dz = 1.0e-2
    field_val = 42.0

    field = np.full((nx, ny, nz), field_val)

    rng = np.random.default_rng(0)
    pos = rng.uniform(0.5 * dx, (nx - 1.5) * dx, (20, 3))
    pos[:, 1] = rng.uniform(0.5 * dy, (ny - 1.5) * dy, 20)
    pos[:, 2] = rng.uniform(0.5 * dz, (nz - 1.5) * dz, 20)

    values = interpolate_field_to_particles(field, pos, dx, dy, dz)

    np.testing.assert_allclose(values, field_val, rtol=1e-12)


def test_interpolate_field_uniform_vector_returns_same_vector_everywhere():
    """Interpolating a uniform vector field must return that vector at any position."""
    nx, ny, nz = 8, 8, 8
    dx = dy = dz = 1.0e-2
    Bx, By, Bz = 0.5, -0.3, 1.2

    field = np.zeros((nx, ny, nz, 3))
    field[..., 0] = Bx
    field[..., 1] = By
    field[..., 2] = Bz

    rng = np.random.default_rng(1)
    n = 15
    pos = rng.uniform(0.5 * dx, (nx - 1.5) * dx, (n, 3))
    pos[:, 1] = rng.uniform(0.5 * dy, (ny - 1.5) * dy, n)
    pos[:, 2] = rng.uniform(0.5 * dz, (nz - 1.5) * dz, n)

    values = interpolate_field_to_particles(field, pos, dx, dy, dz)

    assert values.shape == (n, 3)
    np.testing.assert_allclose(values[:, 0], Bx, rtol=1e-12)
    np.testing.assert_allclose(values[:, 1], By, rtol=1e-12)
    np.testing.assert_allclose(values[:, 2], Bz, rtol=1e-12)


def test_interpolate_field_raises_on_bad_shape():
    """Non 3-D or non (nx,ny,nz,3) array must raise ValueError."""
    bad_field = np.ones((4, 4, 4, 5))  # 5 components — not allowed
    pos = np.array([[0.01, 0.01, 0.01]])

    with pytest.raises(ValueError):
        interpolate_field_to_particles(bad_field, pos, 0.01, 0.01, 0.01)


def test_interpolate_field_bilinear_check_at_cell_center():
    """Value at cell center (equidistant from 8 nodes) must equal mean of node values."""
    nx, ny, nz = 4, 4, 4
    dx = dy = dz = 1.0

    rng = np.random.default_rng(3)
    field = rng.uniform(0.0, 10.0, (nx, ny, nz))

    # Cell (1,1,1) center: position (1.5, 1.5, 1.5)
    pos = np.array([[1.5, 1.5, 1.5]])
    values = interpolate_field_to_particles(field, pos, dx, dy, dz)

    # At cell center each of the 8 nodes gets weight 1/8
    expected = (
        field[1, 1, 1] + field[2, 1, 1] + field[1, 2, 1] + field[1, 1, 2]
        + field[2, 2, 1] + field[2, 1, 2] + field[1, 2, 2] + field[2, 2, 2]
    ) / 8.0

    assert values[0] == pytest.approx(expected, rel=1e-12)


# ===========================================================================
# ParticleSpecies
# ===========================================================================


def test_particle_species_creation_stores_correct_attributes():
    """ParticleSpecies should faithfully store name, mass, charge, and arrays."""
    pos = np.array([[0.1, 0.2, 0.3]])
    vel = np.array([[1.0e4, 0.0, 0.0]])
    wts = np.array([1.0e11])

    sp = ParticleSpecies(
        name="test_ion",
        mass=MASS,
        charge=CHARGE,
        positions=pos,
        velocities=vel,
        weights=wts,
    )

    assert sp.name == "test_ion"
    assert sp.mass == pytest.approx(MASS, rel=1e-12)
    assert sp.charge == pytest.approx(CHARGE, rel=1e-12)
    assert sp.n_particles() == 1


def test_particle_species_n_particles_reflects_array_length():
    """n_particles() must equal the number of rows in the positions array."""
    n = 37
    pos = np.random.default_rng(0).random((n, 3))
    vel = np.zeros((n, 3))
    wts = np.ones(n)

    sp = ParticleSpecies("ions", MASS, CHARGE, pos, vel, wts)
    assert sp.n_particles() == n


def test_particle_species_add_particles_via_concatenation():
    """Concatenating arrays to species.positions/velocities/weights works correctly."""
    pos1 = np.array([[0.0, 0.0, 0.0]])
    vel1 = np.array([[1e4, 0.0, 0.0]])
    wt1 = np.array([1.0])

    sp = ParticleSpecies("ions", MASS, CHARGE, pos1, vel1, wt1)

    pos2 = np.array([[0.1, 0.1, 0.1], [0.2, 0.2, 0.2]])
    vel2 = np.zeros((2, 3))
    wt2 = np.ones(2)

    sp.positions = np.concatenate([sp.positions, pos2])
    sp.velocities = np.concatenate([sp.velocities, vel2])
    sp.weights = np.concatenate([sp.weights, wt2])

    assert sp.n_particles() == 3


def test_particle_species_remove_particles_via_mask():
    """Masking out particles reduces n_particles correctly."""
    n = 10
    pos = np.random.default_rng(5).random((n, 3))
    vel = np.zeros((n, 3))
    wts = np.ones(n)

    sp = ParticleSpecies("ions", MASS, CHARGE, pos, vel, wts)

    # Remove first 3 particles
    keep = np.ones(n, dtype=bool)
    keep[:3] = False
    sp.positions = sp.positions[keep]
    sp.velocities = sp.velocities[keep]
    sp.weights = sp.weights[keep]

    assert sp.n_particles() == n - 3


# ===========================================================================
# HybridPIC driver
# ===========================================================================


def test_hybrid_pic_initializes_with_correct_attributes():
    """HybridPIC should store grid_shape, spacings, dt, and empty species list."""
    grid_shape = (8, 8, 8)
    dx = dy = dz = 1.0e-2
    dt = 1.0e-9

    pic = HybridPIC(grid_shape, dx, dy, dz, dt)

    assert pic.grid_shape == grid_shape
    assert pic.dx == pytest.approx(dx)
    assert pic.dy == pytest.approx(dy)
    assert pic.dz == pytest.approx(dz)
    assert pic.dt == pytest.approx(dt)
    assert pic.species == []


def test_hybrid_pic_add_species_appends_to_list():
    """add_species should return a ParticleSpecies and append it to self.species."""
    pic = HybridPIC((8, 8, 8), 1e-2, 1e-2, 1e-2, 1e-9)

    pos = np.array([[0.02, 0.02, 0.02]])
    vel = np.zeros((1, 3))
    wts = np.array([1.0])

    sp = pic.add_species("d+", MASS, CHARGE, pos, vel, wts)

    assert len(pic.species) == 1
    assert pic.species[0] is sp
    assert sp.name == "d+"


def test_hybrid_pic_add_multiple_species():
    """Multiple calls to add_species should grow the species list."""
    pic = HybridPIC((8, 8, 8), 1e-2, 1e-2, 1e-2, 1e-9)

    for name in ["d+", "he3+", "proton"]:
        pos = np.zeros((5, 3))
        vel = np.zeros((5, 3))
        wts = np.ones(5)
        pic.add_species(name, MASS, CHARGE, pos, vel, wts)

    assert len(pic.species) == 3


def test_hybrid_pic_inject_beam_increases_particle_count():
    """inject_beam should append n_beam particles to the target species."""
    pic = HybridPIC((16, 16, 16), 1e-3, 1e-3, 1e-3, 1e-10)
    pos = np.zeros((0, 3))
    vel = np.zeros((0, 3))
    wts = np.zeros(0)
    pic.add_species("d+", MASS, CHARGE, pos, vel, wts)

    n_beam = 50
    pic.inject_beam(
        species_idx=0,
        n_beam=n_beam,
        energy_eV=1.0e4,
        direction=[0.0, 0.0, 1.0],
        position=[8e-3, 8e-3, 1e-3],
    )

    assert pic.species[0].n_particles() == n_beam


def test_hybrid_pic_inject_beam_kinetic_energy_matches_specified_eV():
    """Injected beam particles should have KE = energy_eV * e_charge."""
    pic = HybridPIC((16, 16, 16), 1e-3, 1e-3, 1e-3, 1e-10)
    pic.add_species("d+", MASS, CHARGE, np.zeros((0, 3)), np.zeros((0, 3)), np.zeros(0))

    energy_eV = 1.0e4
    pic.inject_beam(
        species_idx=0,
        n_beam=20,
        energy_eV=energy_eV,
        direction=[0.0, 0.0, 1.0],
        position=[8e-3, 8e-3, 1e-3],
        spread=0.0,
    )

    sp = pic.species[0]
    speeds = np.linalg.norm(sp.velocities, axis=1)
    expected_speed = math.sqrt(2.0 * energy_eV * e_charge / MASS)

    np.testing.assert_allclose(speeds, expected_speed, rtol=1e-10)


def test_hybrid_pic_inject_beam_direction_normalised():
    """inject_beam with an un-normalized direction should still produce correct speed."""
    pic = HybridPIC((16, 16, 16), 1e-3, 1e-3, 1e-3, 1e-10)
    pic.add_species("d+", MASS, CHARGE, np.zeros((0, 3)), np.zeros((0, 3)), np.zeros(0))

    energy_eV = 5.0e3
    # Deliberately unnormalized direction
    pic.inject_beam(0, 10, energy_eV, [3.0, 4.0, 0.0], [5e-3, 5e-3, 5e-3], spread=0.0)

    sp = pic.species[0]
    speeds = np.linalg.norm(sp.velocities, axis=1)
    expected_speed = math.sqrt(2.0 * energy_eV * e_charge / MASS)

    np.testing.assert_allclose(speeds, expected_speed, rtol=1e-10)


def test_hybrid_pic_inject_beam_with_angular_spread_produces_cone():
    """Beam with angular spread > 0 should produce velocities within the cone."""
    pic = HybridPIC((16, 16, 16), 1e-3, 1e-3, 1e-3, 1e-10)
    pic.add_species("d+", MASS, CHARGE, np.zeros((0, 3)), np.zeros((0, 3)), np.zeros(0))

    energy_eV = 1.0e4
    spread = 0.1  # rad half-angle
    n_beam = 200
    direction = np.array([0.0, 0.0, 1.0])

    pic.inject_beam(0, n_beam, energy_eV, direction, [5e-3, 5e-3, 5e-3], spread=spread)

    sp = pic.species[0]
    # All speeds must equal expected_speed
    expected_speed = math.sqrt(2.0 * energy_eV * e_charge / MASS)
    speeds = np.linalg.norm(sp.velocities, axis=1)
    np.testing.assert_allclose(speeds, expected_speed, rtol=1e-10)

    # Angle between each velocity and z-axis must be <= spread
    v_norm = sp.velocities / speeds[:, None]
    cos_angles = v_norm[:, 2]  # dot with [0,0,1]
    angles = np.arccos(np.clip(cos_angles, -1.0, 1.0))
    assert np.all(angles <= spread + 1e-9)


def test_hybrid_pic_deposit_returns_correct_shapes():
    """deposit() should return arrays matching grid_shape."""
    grid_shape = (8, 8, 8)
    pic = HybridPIC(grid_shape, 1e-2, 1e-2, 1e-2, 1e-9)

    pos = np.array([[0.04, 0.04, 0.04]])
    vel = np.array([[1e4, 0.0, 0.0]])
    wts = np.array([1.0])
    pic.add_species("d+", MASS, CHARGE, pos, vel, wts)

    rho, Jx, Jy, Jz = pic.deposit()

    assert rho.shape == grid_shape
    assert Jx.shape == grid_shape
    assert Jy.shape == grid_shape
    assert Jz.shape == grid_shape


def test_hybrid_pic_deposit_empty_species_returns_zeros():
    """deposit() with no particles should return all-zero arrays."""
    grid_shape = (8, 8, 8)
    pic = HybridPIC(grid_shape, 1e-2, 1e-2, 1e-2, 1e-9)
    pic.add_species("d+", MASS, CHARGE, np.zeros((0, 3)), np.zeros((0, 3)), np.zeros(0))

    rho, Jx, Jy, Jz = pic.deposit()

    np.testing.assert_array_equal(rho, 0.0)
    np.testing.assert_array_equal(Jx, 0.0)


@pytest.mark.slow
def test_hybrid_pic_push_particles_reflecting_bc_keeps_particles_in_domain():
    """After many pushes in a random field, reflecting BC must keep all particles in [0, L]."""
    rng = np.random.default_rng(99)
    grid_shape = (8, 8, 8)
    dx = dy = dz = 1.0e-2
    dt = 1.0e-10

    pic = HybridPIC(grid_shape, dx, dy, dz, dt)

    n = 20
    pos = rng.uniform(0.01, 0.07, (n, 3))
    vel = rng.uniform(-2e5, 2e5, (n, 3))
    wts = np.ones(n)
    pic.add_species("d+", MASS, CHARGE, pos, vel, wts)

    Lx = grid_shape[0] * dx
    Ly = grid_shape[1] * dy
    Lz = grid_shape[2] * dz

    E = np.zeros((*grid_shape, 3))
    B = np.zeros((*grid_shape, 3))
    B[..., 2] = 0.5

    for _ in range(500):
        pic.push_particles(E, B)

    sp = pic.species[0]
    assert np.all(sp.positions[:, 0] >= 0.0)
    assert np.all(sp.positions[:, 0] <= Lx)
    assert np.all(sp.positions[:, 1] >= 0.0)
    assert np.all(sp.positions[:, 1] <= Ly)
    assert np.all(sp.positions[:, 2] >= 0.0)
    assert np.all(sp.positions[:, 2] <= Lz)


# ===========================================================================
# detect_instability
# ===========================================================================


def test_detect_instability_uniform_density_and_uniform_Bz_returns_false():
    """Uniform density with uniform Bz has no sign change and low compression ratio."""
    nx, ny, nz = 8, 8, 8
    rho = np.ones((nx, ny, nz))
    B_field = np.zeros((nx, ny, nz, 3))
    B_field[..., 2] = 1.0

    assert not detect_instability(rho, B_field)


def test_detect_instability_high_compression_but_no_Bz_sign_change_returns_false():
    """High density spike alone (without Bz sign change) should NOT trigger detection."""
    nx, ny, nz = 8, 8, 8
    rho = np.ones((nx, ny, nz))
    # Place a strong density spike at the center
    rho[4, 4, 4] = 100.0  # ratio > threshold of 5.0

    B_field = np.zeros((nx, ny, nz, 3))
    B_field[..., 2] = 1.0  # Uniform Bz, no sign change

    assert not detect_instability(rho, B_field)


def test_detect_instability_Bz_sign_change_but_low_compression_returns_false():
    """Bz sign change alone (without high compression) should NOT trigger detection."""
    nx, ny, nz = 8, 8, 8
    rho = np.ones((nx, ny, nz))  # max/mean = 1.0, below threshold of 5.0

    B_field = np.zeros((nx, ny, nz, 3))
    # Sign change on axis: alternate Bz along z
    mid_x, mid_y = nx // 2, ny // 2
    B_field[mid_x, mid_y, :nz // 2, 2] = 1.0
    B_field[mid_x, mid_y, nz // 2:, 2] = -1.0

    assert not detect_instability(rho, B_field)


def test_detect_instability_detects_density_compression_with_Bz_sign_change():
    """Both criteria together — density spike AND Bz sign change — must return True."""
    nx, ny, nz = 8, 8, 8
    rho = np.ones((nx, ny, nz))
    # Strong compression at multiple points (ratio >> 5)
    rho[2, 2, 2] = 50.0
    rho[5, 5, 5] = 50.0

    B_field = np.zeros((nx, ny, nz, 3))
    mid_x, mid_y = nx // 2, ny // 2
    B_field[mid_x, mid_y, : nz // 2, 2] = 1.0
    B_field[mid_x, mid_y, nz // 2 :, 2] = -1.0

    assert detect_instability(rho, B_field)


def test_detect_instability_zero_density_returns_false():
    """Zero mean density must not trigger detection (division guard)."""
    rho = np.zeros((8, 8, 8))
    B_field = np.zeros((8, 8, 8, 3))

    assert not detect_instability(rho, B_field)


def test_detect_instability_custom_threshold_respected():
    """A lower threshold should trigger instability detection at smaller compression."""
    nx, ny, nz = 8, 8, 8
    rho = np.ones((nx, ny, nz))
    rho[4, 4, 4] = 3.0  # ratio = 3 < default threshold 5, > custom threshold 2

    B_field = np.zeros((nx, ny, nz, 3))
    mid_x, mid_y = nx // 2, ny // 2
    B_field[mid_x, mid_y, : nz // 2, 2] = 1.0
    B_field[mid_x, mid_y, nz // 2 :, 2] = -1.0

    assert not detect_instability(rho, B_field, threshold=5.0)
    assert detect_instability(rho, B_field, threshold=2.0)
