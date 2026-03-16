"""Tests for beam-ion tracker (Campaign 2I lightweight PIC)."""
import numpy as np
import pytest

from dpf.diagnostics.beam_tracker import BeamTracker, BeamTrackerResult


class TestBeamTrackerInit:
    def test_default_init(self):
        bt = BeamTracker()
        assert bt.n_particles == 1000
        assert bt.positions.shape == (1000, 3)
        assert not bt._initialized

    def test_custom_init(self):
        bt = BeamTracker(n_particles=100, grid_shape=(8, 8, 16), dx=0.002)
        assert bt.n_particles == 100
        assert bt.grid_shape == (8, 8, 16)


class TestBeamInjection:
    def test_inject_sets_velocities(self):
        bt = BeamTracker(n_particles=50)
        bt.inject_beam(
            center=np.array([0.01, 0.01, 0.02]),
            direction=np.array([0, 0, 1]),
            energy_eV=100e3,  # 100 keV
        )
        assert bt._initialized
        # Velocities should be nonzero and mostly in z direction
        v_z_mean = np.mean(bt.velocities[:, 2])
        assert v_z_mean > 0

    def test_beam_energy_matches(self):
        bt = BeamTracker(n_particles=200, ion_mass=3.34e-27)
        bt.inject_beam(
            center=np.array([0.005, 0.005, 0.01]),
            direction=np.array([0, 0, 1]),
            energy_eV=50e3,
        )
        v_sq = np.sum(bt.velocities**2, axis=1)
        KE_eV = 0.5 * bt.ion_mass * v_sq / 1.602e-19
        # Mean energy should be close to 50 keV (within spread)
        assert np.mean(KE_eV) == pytest.approx(50e3, rel=0.3)


class TestBorisPush:
    def test_uniform_B_circular_motion(self):
        bt = BeamTracker(n_particles=1, grid_shape=(16, 16, 16), dx=0.01)
        bt.positions[0] = [0.08, 0.08, 0.08]
        bt.velocities[0] = [1e5, 0, 0]  # vx = 100 km/s
        bt._initialized = True

        B = np.zeros((3, 16, 16, 16))
        B[2] = 1.0  # 1 T in z
        E = np.zeros((3, 16, 16, 16))

        # Push for 100 steps
        for _ in range(100):
            bt.push(E, B, dt=1e-10)

        # Particle should still be alive (circular orbit)
        assert bt.alive[0]
        # Speed should be conserved (no E field)
        v_sq = np.sum(bt.velocities[0]**2)
        v_mag = np.sqrt(v_sq)
        assert v_mag == pytest.approx(1e5, rel=0.01)

    def test_E_field_accelerates(self):
        bt = BeamTracker(n_particles=1, grid_shape=(8, 8, 8), dx=0.01)
        bt.positions[0] = [0.04, 0.04, 0.04]
        bt.velocities[0] = [0, 0, 0]
        bt._initialized = True

        B = np.zeros((3, 8, 8, 8))
        E = np.zeros((3, 8, 8, 8))
        E[2] = 1e6  # 1 MV/m in z

        bt.push(E, B, dt=1e-10)
        # Should have gained velocity in z
        assert bt.velocities[0, 2] > 0


class TestBeamResult:
    def test_result_with_no_particles(self):
        bt = BeamTracker(n_particles=10)
        bt.alive[:] = False
        result = bt.get_result()
        assert result.n_particles == 0
        assert result.mean_energy_keV == 0

    def test_result_with_beam(self):
        bt = BeamTracker(n_particles=100)
        bt.inject_beam(
            center=np.array([0.005, 0.005, 0.01]),
            direction=np.array([0, 0, 1]),
            energy_eV=80e3,
        )
        result = bt.get_result()
        assert result.n_particles == 100
        assert result.mean_energy_keV > 0
        assert result.max_energy_keV >= result.mean_energy_keV
        assert len(result.energy_spectrum) > 0
        assert isinstance(result, BeamTrackerResult)
