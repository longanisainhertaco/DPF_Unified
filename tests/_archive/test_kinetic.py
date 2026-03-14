
import numpy as np
import pytest

from dpf.config import SimulationConfig
from dpf.constants import e
from dpf.kinetic.manager import KineticManager


class TestKineticManager:
    @pytest.fixture
    def config(self):
        conf = SimulationConfig(
            grid_shape=[10, 10, 10],
            dx=0.1,
            sim_time=1e-6,
            circuit={
                "C": 1e-6, "V0": 10e3, "L0": 10e-9,
                "anode_radius": 0.01, "cathode_radius": 0.02
            }
        )
        conf.kinetic.enabled = True
        return conf

    def test_initialization(self, config):
        km = KineticManager(config)
        assert km.kc.enabled is True
        assert km.driver is not None

    def test_gyromotion(self, config):
        """Test that a particle in a uniform B-field performs cyclotron motion."""
        km = KineticManager(config)

        # Override initial conditions for a single test particle
        # Place at center, moving in +x
        v0 = 1e5
        km.ion_species.positions = np.array([[0.5, 0.5, 0.5]])
        km.ion_species.velocities = np.array([[v0, 0.0, 0.0]])
        km.ion_species.weights = np.array([1.0])

        # Uniform B-field in +z
        B0 = 1.0 # Tesla
        nx, ny, nz = 10, 10, 10
        B_field = np.zeros((nx, ny, nz, 3))
        B_field[..., 2] = B0

        E_field = np.zeros((nx, ny, nz, 3))

        # Expected Larmor radius: r = mv / qB
        # Expected period: T = 2pi m / qB
        mass = config.ion_mass
        T_c = 2 * np.pi * mass / (e * B0)

        dt = T_c / 20.0 # Resolve gyro-period

        # Step forward 1/4 period -> should be at (x0, y0-r, z0) roughly?
        # Actually, if v is +x, F = q v x B is (v, 0, 0) x (0, 0, B) = (0, -vB, 0).
        # Force is -y direction. Particle turns right.

        for _ in range(5):
            km.step(dt, 0.0, E_field, B_field)

        vel = km.ion_species.velocities[0]

        # Verify energy conservation (Boris push should reserve E in static B)
        v_mag = np.linalg.norm(vel)
        assert np.isclose(v_mag, v0, rtol=1e-4), "Energy not conserved in B-field"

    def test_beam_injection(self, config):
        config.kinetic.inject_beam = True
        config.kinetic.start_time = 1e-9
        config.kinetic.n_particles = 100

        km = KineticManager(config)

        nx, ny, nz = 10, 10, 10
        B_field = np.zeros((nx, ny, nz, 3))
        E_field = np.zeros((nx, ny, nz, 3))

        # t=0, no beam
        km.step(1e-10, 0.0, E_field, B_field)
        assert km.ion_species.n_particles() == 0

        # t > start_time, beam injected
        km.step(1e-10, 2e-9, E_field, B_field)
        assert km.ion_species.n_particles() == 100
        assert km.beam_injected is True
